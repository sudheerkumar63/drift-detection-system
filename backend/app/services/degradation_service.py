import numpy as np


def estimate_degradation(confidence: float, entropy: float, margin: float, drift_score: float, thresholds: dict | None = None) -> dict:
    if thresholds is None:
        thresholds = {"confidence": 0.7, "entropy": 1.1, "margin": 0.15, "drift_score": 0.25}

    risk_factors = [
        max(0.0, thresholds["confidence"] - confidence) / max(thresholds["confidence"], 1e-6),
        max(0.0, entropy - thresholds["entropy"]) / max(thresholds["entropy"], 1e-6),
        max(0.0, thresholds["margin"] - margin) / max(thresholds["margin"], 1e-6),
        max(0.0, drift_score - thresholds["drift_score"]) / max(thresholds["drift_score"], 1e-6),
    ]
    heuristic_risk = float(np.clip(sum(risk_factors) / 4, 0.0, 1.0))

    reasons = []
    if confidence < thresholds["confidence"]:
        reasons.append("Prediction confidence is below the configured threshold.")
    if entropy > thresholds["entropy"]:
        reasons.append("Prediction entropy increased, indicating greater uncertainty.")
    if margin < thresholds["margin"]:
        reasons.append("The margin between top and second predictions is too small.")
    if drift_score > thresholds["drift_score"]:
        reasons.append("Feature distribution drift exceeds the configured threshold.")

    return {
        "heuristic_risk_score": heuristic_risk,
        "thresholds": thresholds,
        "reasons": reasons,
        "status": "possible_degradation" if heuristic_risk >= 0.5 else "stable",
        "note": "This is a heuristic risk estimate, not a calibrated probability of model failure. It does not prove the prediction is wrong without ground-truth labels.",
    }
