import base64
import csv
import io
import json
from datetime import datetime
from math import isfinite
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from backend.app.config import settings
from backend.app.database import ensure_database, get_deployment_status, get_prediction_history, insert_prediction, update_deployment_status
from backend.app.services.adaptation_service import get_adaptation_status, start_adaptation, stop_adaptation
from backend.app.services.analytics_service import build_analytics
from backend.app.services.degradation_service import estimate_degradation
from backend.app.services.drift_service import run_drift_detection
from backend.app.services.environment_service import apply_environmental_effects
from backend.app.services.model_service import get_model_status, predict_from_bytes
from backend.app.services.safety_service import verify_safety

router = APIRouter(prefix=settings.api_prefix)


class EnvironmentPayload(BaseModel):
    rain: float = 0
    fog: float = 0
    glare: float = 0
    blur: float = 0
    brightness: float = 0
    contrast: float = 0
    noise: float = 0
    quality: float = 100


class DriftRequest(BaseModel):
    reference_features: list[float] = Field(min_length=2)
    current_features: list[float] = Field(min_length=2)
    threshold: float = Field(default=0.2, ge=0, le=1)
    method: Literal["ks", "mmd", "combined"] = "combined"

    @model_validator(mode="after")
    def validate_feature_distributions(self) -> "DriftRequest":
        if len(self.reference_features) != len(self.current_features):
            raise ValueError("Reference and current distributions must have the same number of values.")
        if not all(isfinite(value) for value in self.reference_features + self.current_features):
            raise ValueError("Reference and current distributions must contain only finite values.")
        return self


class DegradationRequest(BaseModel):
    confidence: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0, le=10)
    margin: float = Field(ge=0, le=1)
    drift_score: float = Field(ge=0, le=1)
    thresholds: dict[str, float] = Field(default_factory=lambda: {
        "confidence": 0.7,
        "entropy": 1.1,
        "margin": 0.15,
        "drift_score": 0.25,
    })

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DegradationRequest":
        expected = {"confidence", "entropy", "margin", "drift_score"}
        if set(self.thresholds) != expected:
            raise ValueError(f"Thresholds must contain exactly: {', '.join(sorted(expected))}.")
        if any(not isfinite(value) or value < 0 for value in self.thresholds.values()):
            raise ValueError("Thresholds must contain finite non-negative values.")
        if any(self.thresholds[key] > 1 for key in ("confidence", "margin", "drift_score")):
            raise ValueError("Confidence, margin, and drift thresholds must be between 0 and 1.")
        return self


class ValidationMetrics(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class SafetyAcceptance(BaseModel):
    accuracy: float = Field(default=0.82, ge=0, le=1)
    precision: float = Field(default=0.8, ge=0, le=1)
    recall: float = Field(default=0.78, ge=0, le=1)
    f1: float = Field(default=0.8, ge=0, le=1)


class SafetyRequest(BaseModel):
    metrics: ValidationMetrics
    acceptance: SafetyAcceptance = Field(default_factory=SafetyAcceptance)


@router.get("/health")
def health_check() -> dict[str, Any]:
    model_status = get_model_status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "demo_mode": model_status["demo_mode"],
        "model": model_status,
        "database": "ready",
        "message": model_status["message"],
    }


@router.get("/model-status")
def model_status() -> dict[str, Any]:
    return get_model_status()


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 10MB limit.")
    result = predict_from_bytes(content)
    insert_prediction({
        "timestamp": datetime.utcnow().isoformat(),
        "class_name": result["predicted_class"],
        "confidence": result["confidence"],
        "entropy": result["entropy"],
        "margin": result["margin"],
        "drift_status": "stable",
        "degradation_risk": 0.1,
        "model_version": result["model_version"],
        "env_summary": "clear_weather",
        "metadata": json.dumps({"demo_mode": result["demo_mode"]}),
    })
    return {"status": "success", "prediction": result}


def _image_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@router.post("/simulate-environment")
async def simulate_environment(file: UploadFile = File(...), settings_payload: str | None = None) -> dict[str, Any]:
    content = await file.read()
    payload = json.loads(settings_payload or "{}") if settings_payload else {}
    env_settings = EnvironmentPayload(**payload).model_dump()
    modified = apply_environmental_effects(content, env_settings)
    original_prediction = predict_from_bytes(content)
    modified_prediction = predict_from_bytes(modified, drift_shift=0.25)
    mime_type = file.content_type or "image/jpeg"
    return {
        "status": "success",
        "original": original_prediction,
        "modified": modified_prediction,
        "original_image": _image_to_data_url(content, mime_type),
        "modified_image": _image_to_data_url(modified, "image/jpeg"),
        "environment": env_settings,
        "modified_image_length": len(modified),
        "message": "Environmental effects applied to the uploaded image. Compare predictions to evaluate drift impact.",
    }


@router.post("/compare-predictions")
async def compare_predictions(file: UploadFile = File(...), settings_payload: str | None = None) -> dict[str, Any]:
    content = await file.read()
    payload = json.loads(settings_payload or "{}") if settings_payload else {}
    env_settings = EnvironmentPayload(**payload).model_dump()
    modified = apply_environmental_effects(content, env_settings)
    original = predict_from_bytes(content)
    modified_result = predict_from_bytes(modified, drift_shift=0.3)
    mime_type = file.content_type or "image/jpeg"
    return {
        "original": original,
        "modified": modified_result,
        "original_image": _image_to_data_url(content, mime_type),
        "modified_image": _image_to_data_url(modified, "image/jpeg"),
        "difference": {
            "class_changed": original["predicted_class"] != modified_result["predicted_class"],
            "confidence_change": round(modified_result["confidence"] - original["confidence"], 4),
            "entropy_change": round(modified_result["entropy"] - original["entropy"], 4),
        },
    }


@router.post("/drift-detection")
def drift_detection(request: DriftRequest) -> dict[str, Any]:
    result = run_drift_detection(
        np.array(request.reference_features, dtype=float),
        np.array(request.current_features, dtype=float),
        threshold=request.threshold,
        method=request.method,
    )
    return {"status": "success", "result": result}


@router.post("/degradation-estimate")
def degradation_estimate(request: DegradationRequest) -> dict[str, Any]:
    result = estimate_degradation(
        confidence=request.confidence,
        entropy=request.entropy,
        margin=request.margin,
        drift_score=request.drift_score,
        thresholds=request.thresholds,
    )
    return {"status": "success", "risk": result}


@router.post("/adaptation/start")
async def adaptation_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config = payload or {}
    result = start_adaptation(config)
    return {"status": "success", "job": result}


@router.get("/adaptation/status/{job_id}")
def adaptation_status(job_id: str) -> dict[str, Any]:
    job = get_adaptation_status(job_id)
    if job["status"] == "not_found":
        raise HTTPException(status_code=404, detail=job["message"] if "message" in job else "Adaptation job not found.")
    return {"status": "success", "job": job}


@router.post("/adaptation/stop/{job_id}")
def adaptation_stop(job_id: str) -> dict[str, Any]:
    job = stop_adaptation(job_id)
    if job["status"] == "not_found":
        raise HTTPException(status_code=404, detail=job["message"])
    return {"status": "success", "job": job}


@router.post("/safety/verify")
def safety_verify(request: SafetyRequest) -> dict[str, Any]:
    result = verify_safety(request.metrics.model_dump(), request.acceptance.model_dump())
    return {"status": "success", "result": result}


@router.post("/deploy")
def deploy_model(request: SafetyRequest) -> dict[str, Any]:
    verification = verify_safety(request.metrics.model_dump(), request.acceptance.model_dump())
    if verification["status"] != "approved":
        return JSONResponse(status_code=400, content={"status": "rejected", "message": "Model deployment is blocked because safety verification failed.", "verification": verification})
    update_deployment_status("approved", approved_model="demo-model-v1", last_verification="research-demo")
    return {"status": "approved", "deployment": {"approved_model": "demo-model-v1", "status": "approved"}}


@router.post("/rollback")
def rollback_model() -> dict[str, Any]:
    update_deployment_status("rolled_back", approved_model=None, last_verification="manual-roll-back")
    return {"status": "rolled_back", "message": "Previously approved model remains in place. The demo system can roll back safely."}


@router.get("/deployment-status")
def deployment_status() -> dict[str, Any]:
    return {"status": "success", "deployment": get_deployment_status()}


@router.get("/history")
def prediction_history() -> dict[str, Any]:
    ensure_database()
    return {"status": "success", "items": get_prediction_history(limit=100)}


@router.get("/analytics")
def analytics() -> dict[str, Any]:
    history = get_prediction_history(limit=100)
    payload = build_analytics(history, get_model_status())
    return {"status": "success", "analytics": payload}


@router.get("/export-history/csv")
def export_history_csv() -> PlainTextResponse:
    history = get_prediction_history(limit=100)
    fieldnames = ["id", "timestamp", "class_name", "confidence", "entropy", "margin", "drift_status", "degradation_risk", "model_version", "env_summary"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in history:
        writer.writerow({key: record.get(key, "") for key in fieldnames})
    return PlainTextResponse(output.getvalue(), media_type="text/csv")
