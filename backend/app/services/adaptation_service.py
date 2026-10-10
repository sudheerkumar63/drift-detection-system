import random
import time
from datetime import datetime


ADAPTATION_STATE = {"status": "idle", "progress": 0, "logs": [], "job_id": None, "last_report": None}


def start_adaptation(config: dict) -> dict:
    job_id = f"adapt-{int(time.time())}"
    ADAPTATION_STATE.update({
        "status": "running",
        "progress": 0,
        "job_id": job_id,
        "logs": [
            "Initializing adaptation run",
            f"Selected environment training data: {config.get('environment_data', 'synthetic')}"
        ],
    })
    return {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "message": "Adaptation started. Training will simulate a fine-tuning run for the demo workload.",
    }


def stop_adaptation(job_id: str) -> dict:
    if ADAPTATION_STATE["job_id"] != job_id:
        return {"status": "not_found", "message": "Adaptation job not found."}
    if ADAPTATION_STATE["status"] != "running":
        return {
            "status": ADAPTATION_STATE["status"],
            "progress": ADAPTATION_STATE["progress"],
            "logs": ADAPTATION_STATE["logs"],
            "job_id": ADAPTATION_STATE["job_id"],
            "message": "Adaptation is no longer running.",
        }
    ADAPTATION_STATE["status"] = "stopped"
    ADAPTATION_STATE["logs"].append("Adaptation stopped by user.")
    return {
        "status": ADAPTATION_STATE["status"],
        "progress": ADAPTATION_STATE["progress"],
        "logs": ADAPTATION_STATE["logs"],
        "job_id": ADAPTATION_STATE["job_id"],
        "message": "Adaptation stopped.",
    }


def get_adaptation_status(job_id: str | None = None) -> dict:
    if ADAPTATION_STATE["status"] == "idle":
        return {"status": "idle", "progress": 0, "logs": [], "job_id": job_id}
    if job_id and ADAPTATION_STATE["job_id"] != job_id:
        return {"status": "not_found", "progress": 0, "logs": [], "job_id": job_id}
    progress = ADAPTATION_STATE["progress"]
    if ADAPTATION_STATE["status"] == "running" and progress < 100:
        ADAPTATION_STATE["progress"] = min(100, progress + 10)
        if ADAPTATION_STATE["progress"] >= 25 and "Fine-tuning CNN last layers" not in ADAPTATION_STATE["logs"]:
            ADAPTATION_STATE["logs"].append("Fine-tuning CNN last layers")
        if ADAPTATION_STATE["progress"] >= 50 and "Evaluating validation performance" not in ADAPTATION_STATE["logs"]:
            ADAPTATION_STATE["logs"].append("Evaluating validation performance")
        if ADAPTATION_STATE["progress"] >= 75 and "Finalizing adapted checkpoint" not in ADAPTATION_STATE["logs"]:
            ADAPTATION_STATE["logs"].append("Finalizing adapted checkpoint")
    if ADAPTATION_STATE["progress"] >= 100:
        ADAPTATION_STATE["status"] = "completed"
        ADAPTATION_STATE["logs"].append("Adaptation completed. Model comparison is ready for safety verification.")
    return {
        "job_id": ADAPTATION_STATE["job_id"],
        "status": ADAPTATION_STATE["status"],
        "progress": ADAPTATION_STATE["progress"],
        "logs": ADAPTATION_STATE["logs"],
        "timestamp": datetime.utcnow().isoformat(),
    }

