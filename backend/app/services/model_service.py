import hashlib
import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from backend.app.config import BASE_DIR
from backend.app.ml.traffic_sign_cnn import (
    ARCHITECTURE_VERSION,
    IMAGE_SIZE,
    LEGACY_ARCHITECTURE_VERSION,
    LEGACY_IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TrafficSignCNN,
    TrafficSignCNNV1,
)

TRAFFIC_SIGN_CLASSES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60", "Speed limit 70",
    "Speed limit 80", "Restriction ends 80", "Speed limit 100", "Speed limit 120", "No overtaking",
    "No overtaking for trucks", "Priority at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice or snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons",
]

MODEL_PATH = BASE_DIR / "model_weights" / "gtsrb_model.pth"
REFERENCE_FEATURES_PATH = BASE_DIR / "data" / "reference_features.npy"


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    stabilized = x - np.max(x)
    exps = np.exp(stabilized)
    return exps / np.sum(exps)


def _deterministic_probabilities(image_bytes: bytes, shift: float = 0.0) -> np.ndarray:
    digest = hashlib.sha256(image_bytes).hexdigest()
    seed_value = int(digest[:8], 16)
    rng = np.random.default_rng(seed_value)
    base = rng.normal(0.0, 1.0, size=len(TRAFFIC_SIGN_CLASSES)).astype(float)
    class_bias = np.linspace(-1.5, 1.8, len(TRAFFIC_SIGN_CLASSES))
    logits = base + class_bias
    temperature = max(0.1, 1.0 + shift)
    return _safe_softmax(logits / temperature)


def compute_entropy(probabilities: np.ndarray) -> float:
    probs = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_margin(probabilities: np.ndarray) -> float:
    sorted_probs = np.sort(probabilities)[::-1]
    return float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])


def prediction_metrics(probabilities: np.ndarray) -> dict:
    ordered = np.argsort(probabilities)[::-1]
    top5 = [(TRAFFIC_SIGN_CLASSES[idx], float(probabilities[idx])) for idx in ordered[:5]]
    confidence = float(probabilities[ordered[0]])
    entropy = compute_entropy(probabilities)
    margin = compute_margin(probabilities)
    return {
        "top5": top5,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
    }


def get_model_status() -> dict:
    if not MODEL_PATH.exists():
        return {
            "model_available": False,
            "demo_mode": True,
            "model_path": str(MODEL_PATH),
            "message": "Trained GTSRB checkpoint missing. The app is operating in transparent demo mode.",
            "version": "demo-v1",
        }
    checkpoint_stat = MODEL_PATH.stat()
    _, checkpoint = _load_checkpoint(str(MODEL_PATH.resolve()), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)
    return {
        "model_available": True,
        "demo_mode": False,
        "model_path": str(MODEL_PATH),
        "message": "Trained GTSRB checkpoint loaded successfully.",
        "version": checkpoint["architecture"],
        "evaluation": checkpoint.get("evaluation"),
    }


@lru_cache(maxsize=1)
def _load_checkpoint(path: str, modified_ns: int, file_size: int) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("The GTSRB checkpoint must be a dictionary saved by backend.train_model.")
    architecture = checkpoint.get("architecture")
    if architecture not in {ARCHITECTURE_VERSION, LEGACY_ARCHITECTURE_VERSION}:
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}.")
    if checkpoint.get("class_names") != TRAFFIC_SIGN_CLASSES:
        raise ValueError("Checkpoint class labels do not match the GTSRB label order expected by this app.")
    expected_image_size = IMAGE_SIZE if architecture == ARCHITECTURE_VERSION else LEGACY_IMAGE_SIZE
    if checkpoint.get("image_size") != expected_image_size:
        raise ValueError(f"Checkpoint image_size must be {expected_image_size}.")
    if not checkpoint.get("training_complete"):
        raise ValueError("Checkpoint is missing the completed-training marker; train it with backend.train_model first.")
    model_class = TrafficSignCNN if architecture == ARCHITECTURE_VERSION else TrafficSignCNNV1
    model = model_class(num_classes=len(TRAFFIC_SIGN_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _preprocess_tensor(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(NORMALIZE_MEAN, dtype=np.float32)) / np.asarray(NORMALIZE_STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def load_reference_features() -> np.ndarray:
    path = BASE_DIR / "data" / "reference_features.npy"
    if path.exists():
        return np.load(path)
    generated = np.random.default_rng(42).normal(0.0, 1.0, size=(100, 64))
    np.save(path, generated)
    return generated


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def predict_from_bytes(image_bytes: bytes, drift_shift: float = 0.0) -> dict:
    started_at = time.perf_counter()
    model_status = get_model_status()
    if model_status["model_available"]:
        checkpoint_stat = MODEL_PATH.stat()
        model, checkpoint = _load_checkpoint(str(MODEL_PATH.resolve()), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)
        with torch.inference_mode():
            logits = model(_preprocess_tensor(image_bytes, checkpoint["image_size"]))
            temperature = max(0.1, 1.0 + drift_shift)
            probabilities = torch.softmax(logits / temperature, dim=1).squeeze(0).cpu().numpy()
        inference_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
    else:
        probabilities = _deterministic_probabilities(image_bytes, shift=drift_shift)
        seed = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)
        inference_time_ms = round(float(np.random.default_rng(seed).uniform(10, 80)), 2)
        checkpoint = {}
    metrics = prediction_metrics(probabilities)
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": TRAFFIC_SIGN_CLASSES[predicted_index],
        "confidence": metrics["confidence"],
        "probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(TRAFFIC_SIGN_CLASSES)},
        "top5": metrics["top5"],
        "entropy": metrics["entropy"],
        "margin": metrics["margin"],
        "model_version": model_status["version"],
        "model_available": model_status["model_available"],
        "demo_mode": model_status["demo_mode"],
        "message": model_status["message"],
        "inference_time_ms": inference_time_ms,
        "evaluation": checkpoint.get("evaluation"),
        "reference_features": load_reference_features().shape,
    }
