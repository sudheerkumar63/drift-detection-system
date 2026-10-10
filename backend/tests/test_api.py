from io import BytesIO

import torch
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.ml.traffic_sign_cnn import ARCHITECTURE_VERSION, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, TrafficSignCNN
from backend.app.services.analytics_service import build_analytics
from backend.app.services import model_service
from backend.app.services.drift_service import run_drift_detection
from backend.app.services.model_service import TRAFFIC_SIGN_CLASSES, predict_from_bytes
from backend.train_model import stratified_split

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buffer, format="PNG")
    image = buffer.getvalue()
    response = client.post("/api/predict", files={"file": ("sample.png", image, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "prediction" in payload
    assert payload["prediction"]["predicted_class"]


def test_simulate_environment_returns_images():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buffer, format="PNG")
    image = buffer.getvalue()
    response = client.post(
        "/api/simulate-environment",
        files={"file": ("sample.png", image, "image/png")},
        data={"settings_payload": '{"rain": 30, "fog": 15, "brightness": 5, "quality": 80}'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["original_image"].startswith("data:image/png;base64,")
    assert payload["modified_image"].startswith("data:image/jpeg;base64,")
    assert len(payload["modified_image"]) > len(payload["original_image"]) * 0.8


def test_prediction_probabilities_change_with_input_and_drift():
    def make_image(color):
        buffer = BytesIO()
        Image.new("RGB", (32, 32), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    image = make_image((255, 0, 0))
    baseline = predict_from_bytes(image)
    shifted = predict_from_bytes(image, drift_shift=0.3)
    different_image = predict_from_bytes(make_image((0, 0, 255)))

    baseline_probabilities = baseline["probabilities"]
    shifted_probabilities = shifted["probabilities"]
    assert abs(sum(baseline_probabilities.values()) - 1.0) < 1e-6
    assert shifted_probabilities != baseline_probabilities
    assert shifted["confidence"] < baseline["confidence"]
    assert different_image["probabilities"] != baseline_probabilities


def test_compatible_checkpoint_runs_real_inference(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "gtsrb_model.pth"
    model = TrafficSignCNN(num_classes=len(TRAFFIC_SIGN_CLASSES))
    torch.save({
        "architecture": ARCHITECTURE_VERSION,
        "training_complete": True,
        "model_state_dict": model.state_dict(),
        "class_names": TRAFFIC_SIGN_CLASSES,
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
        "evaluation": {"accuracy": 0.5, "samples": 2},
    }, checkpoint_path)
    monkeypatch.setattr(model_service, "MODEL_PATH", checkpoint_path)
    model_service._load_checkpoint.cache_clear()
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(buffer, format="PNG")

    try:
        result = model_service.predict_from_bytes(buffer.getvalue())
        assert result["model_available"] is True
        assert result["demo_mode"] is False
        assert len(result["probabilities"]) == 43
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
        assert result["evaluation"]["accuracy"] == 0.5
    finally:
        model_service._load_checkpoint.cache_clear()


def test_stratified_split_preserves_every_gtsrb_class():
    targets = [class_id for class_id, count in enumerate([3, 5, 10]) for _ in range(count)]
    training, validation = stratified_split(targets, validation_fraction=0.2, seed=42)

    assert set(training).isdisjoint(validation)
    assert sorted(training + validation) == list(range(len(targets)))
    assert {targets[index] for index in validation} == {0, 1, 2}


def test_analytics_uses_active_model_records_without_fabricated_metrics():
    analytics = build_analytics([
        {"timestamp": "2026-01-01T00:00:00", "model_version": "active-v1", "confidence": 0.75, "entropy": 0.9, "margin": 0.3, "drift_status": "stable"},
        {"timestamp": "2026-01-02T00:00:00", "model_version": "demo-v1", "confidence": 0.99, "entropy": 0.1, "margin": 0.8, "drift_status": "warning"},
    ], {"model_available": True, "demo_mode": False, "version": "active-v1", "evaluation": {"accuracy": 0.8, "samples": 10}})

    assert analytics["summary"]["status"] == "real_model"
    assert analytics["summary"]["average_confidence"] == 0.75
    assert analytics["summary"]["average_entropy"] == 0.9
    assert analytics["summary"]["average_margin"] == 0.3
    assert analytics["summary"]["model_evaluation"]["accuracy"] == 0.8
    assert analytics["snapshot"]["total_predictions"] == 1
    assert analytics["snapshot"]["drift_events"] == 0


def test_degradation_endpoint():
    response = client.post("/api/degradation-estimate", json={"confidence": 0.6, "entropy": 1.5, "margin": 0.08, "drift_score": 0.35})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["risk"]["heuristic_risk_score"] > 0


def test_degradation_estimate_responds_to_metric_and_threshold_changes():
    baseline = {"confidence": 0.9, "entropy": 0.4, "margin": 0.5, "drift_score": 0.05}
    degraded = {"confidence": 0.3, "entropy": 2.5, "margin": 0.02, "drift_score": 0.8}
    low_risk = client.post("/api/degradation-estimate", json=baseline).json()["risk"]
    high_risk = client.post("/api/degradation-estimate", json=degraded).json()["risk"]

    assert high_risk["heuristic_risk_score"] > low_risk["heuristic_risk_score"]
    assert high_risk["status"] == "possible_degradation"
    assert low_risk["status"] == "stable"

    adjusted = {**baseline, "thresholds": {
        "confidence": 0.95, "entropy": 0.2, "margin": 0.9, "drift_score": 0.01,
    }}
    adjusted_risk = client.post("/api/degradation-estimate", json=adjusted).json()["risk"]
    assert adjusted_risk["heuristic_risk_score"] > low_risk["heuristic_risk_score"]


def test_degradation_endpoint_rejects_invalid_threshold_set():
    response = client.post("/api/degradation-estimate", json={
        "confidence": 0.6,
        "entropy": 1.5,
        "margin": 0.08,
        "drift_score": 0.35,
        "thresholds": {"confidence": 0.7},
    })
    assert response.status_code == 422


def test_drift_endpoint():
    probabilities = [1 / 43] * 43
    response = client.post("/api/drift-detection", json={
        "reference_features": probabilities,
        "current_features": probabilities,
        "threshold": 0.2,
        "method": "combined",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["result"]["drift_score"] == 0
    assert payload["result"]["status"] == "stable"


def test_drift_controls_change_method_and_threshold_outcome():
    reference = [0.98, 0.0055, 0.0055, 0.0055, 0.0035]
    current = [0.8, 0.05, 0.05, 0.05, 0.05]
    ks_result = run_drift_detection(reference, current, threshold=0.9, method="ks")
    mmd_result = run_drift_detection(reference, current, threshold=0.9, method="mmd")
    low_threshold_result = run_drift_detection(reference, current, threshold=0.1, method="ks")

    assert ks_result["drift_score"] == ks_result["ks_test"]["statistic"]
    assert mmd_result["drift_score"] == mmd_result["mmd"]["score"]
    assert ks_result["drift_score"] != mmd_result["drift_score"]
    assert ks_result["status"] == "stable"
    assert low_threshold_result["status"] == "warning"


def test_drift_endpoint_rejects_mismatched_distributions():
    response = client.post("/api/drift-detection", json={
        "reference_features": [0.5, 0.5],
        "current_features": [0.2, 0.3, 0.5],
    })
    assert response.status_code == 422


def test_safety_endpoint():
    response = client.post("/api/safety/verify", json={
        "metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
        "acceptance": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8},
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["result"]["status"] == "approved"


def test_safety_verification_responds_to_metrics_and_thresholds():
    metrics = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8}
    thresholds = {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7}
    passed = client.post("/api/safety/verify", json={
        "metrics": metrics, "acceptance": thresholds,
    }).json()["result"]
    failed = client.post("/api/safety/verify", json={
        "metrics": {**metrics, "accuracy": 0.6}, "acceptance": thresholds,
    }).json()["result"]
    raised_threshold = client.post("/api/safety/verify", json={
        "metrics": metrics, "acceptance": {**thresholds, "f1": 0.95},
    }).json()["result"]

    assert passed["status"] == "approved"
    assert failed["status"] == "rejected"
    assert failed["checks"]["accuracy"] is False
    assert raised_threshold["status"] == "rejected"
    assert raised_threshold["checks"]["f1"] is False


def test_safety_verification_requires_measured_metrics():
    response = client.post("/api/safety/verify", json={
        "acceptance": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8},
    })
    assert response.status_code == 422


def test_adaptation_can_be_stopped():
    start_response = client.post("/api/adaptation/start", json={"environment_data": "test"})
    assert start_response.status_code == 200
    job_id = start_response.json()["job"]["job_id"]

    progress_response = client.get(f"/api/adaptation/status/{job_id}")
    assert progress_response.status_code == 200
    progress = progress_response.json()["job"]["progress"]

    stop_response = client.post(f"/api/adaptation/stop/{job_id}")
    assert stop_response.status_code == 200
    stopped_job = stop_response.json()["job"]
    assert stopped_job["status"] == "stopped"
    assert stopped_job["progress"] == progress
    assert any("stopped by user" in line.lower() for line in stopped_job["logs"])

    status_response = client.get(f"/api/adaptation/status/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["job"]["status"] == "stopped"
    assert status_response.json()["job"]["progress"] == stopped_job["progress"]
