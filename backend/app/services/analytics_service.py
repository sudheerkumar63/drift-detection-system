def build_analytics(history: list[dict], model_status: dict) -> dict:
    model_history = [record for record in history if record.get("model_version") == model_status["version"]]
    processed_images = len(model_history)
    drift_events = sum(1 for record in model_history if record.get("drift_status") == "warning")
    confidences = [float(record["confidence"]) for record in model_history if record.get("confidence") is not None]
    entropies = [float(record["entropy"]) for record in model_history if record.get("entropy") is not None]
    margins = [float(record["margin"]) for record in model_history if record.get("margin") is not None]
    confidence_distribution = [
        {"name": "Low (<0.5)", "value": sum(value < 0.5 for value in confidences)},
        {"name": "Medium (0.5-0.8)", "value": sum(0.5 <= value < 0.8 for value in confidences)},
        {"name": "High (>=0.8)", "value": sum(value >= 0.8 for value in confidences)},
    ]
    chronological_records = sorted(model_history, key=lambda record: record.get("timestamp", ""))
    model_label = "Real model" if model_status["model_available"] else "Demo mode"
    return {
        "snapshot": {
            "total_predictions": processed_images,
            "processed_images": processed_images,
            "drift_events": drift_events,
            "confidence_distribution": confidence_distribution,
            "model_status": model_label,
        },
        "series": {
            "confidence_over_time": [
                {"time": record["timestamp"][:19], "confidence": float(record["confidence"])}
                for record in chronological_records
                if record.get("confidence") is not None
            ],
            "entropy": [
                {"time": record["timestamp"][:19], "entropy": float(record["entropy"])}
                for record in chronological_records
                if record.get("entropy") is not None
            ],
            "margin": [
                {"time": record["timestamp"][:19], "margin": float(record["margin"])}
                for record in chronological_records
                if record.get("margin") is not None
            ],
        },
        "summary": {
            "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "average_entropy": round(sum(entropies) / len(entropies), 3) if entropies else None,
            "average_margin": round(sum(margins) / len(margins), 3) if margins else None,
            "status": "real_model" if model_status["model_available"] else "demo_mode",
            "model_version": model_status["version"],
            "model_evaluation": model_status.get("evaluation"),
        },
    }
