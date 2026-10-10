from datetime import datetime


def verify_safety(adapted: dict, acceptance: dict | None = None) -> dict:
    if acceptance is None:
        acceptance = {"accuracy": 0.82, "precision": 0.8, "recall": 0.78, "f1": 0.8}

    pass_flags = {
        metric: adapted[metric] >= acceptance[metric]
        for metric in acceptance
    }
    approved = all(pass_flags.values())
    return {
        "status": "approved" if approved else "rejected",
        "accepted_thresholds": acceptance,
        "adapted": adapted,
        "checks": pass_flags,
        "message": "All supplied validation metrics meet the configured thresholds." if approved else "One or more supplied validation metrics are below their configured thresholds.",
        "verification_time": datetime.utcnow().isoformat(),
        "note": "Verification evaluates the validation metrics supplied by the user; it does not independently validate the dataset or model. This is a research demonstration, not a certified automotive safety approval.",
    }
