import sqlite3
from contextlib import closing
from pathlib import Path

from backend.app.config import BASE_DIR

DB_PATH = BASE_DIR / "data" / "drift.db"


def ensure_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                entropy REAL NOT NULL,
                margin REAL NOT NULL,
                drift_status TEXT NOT NULL,
                degradation_risk REAL NOT NULL,
                model_version TEXT NOT NULL,
                env_summary TEXT,
                metadata TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                phase TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deployment_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                approved_model TEXT,
                last_verification TEXT,
                approved_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        conn.commit()


def insert_prediction(record: dict) -> None:
    ensure_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, class_name, confidence, entropy, margin,
                drift_status, degradation_risk, model_version, env_summary, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["class_name"],
                float(record["confidence"]),
                float(record["entropy"]),
                float(record["margin"]),
                record["drift_status"],
                float(record["degradation_risk"]),
                record["model_version"],
                record.get("env_summary", ""),
                record.get("metadata", "{}"),
            ),
        )
        conn.commit()


def get_prediction_history(limit: int = 50, offset: int = 0) -> list[dict]:
    ensure_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, class_name, confidence, entropy, margin,
                   drift_status, degradation_risk, model_version, env_summary, metadata
            FROM predictions
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "class_name": row[2],
            "confidence": row[3],
            "entropy": row[4],
            "margin": row[5],
            "drift_status": row[6],
            "degradation_risk": row[7],
            "model_version": row[8],
            "env_summary": row[9],
            "metadata": row[10],
        }
        for row in rows
    ]


def update_deployment_status(status: str, approved_model: str | None = None, last_verification: str | None = None) -> None:
    ensure_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO deployment_state (approved_model, last_verification, approved_at, status) VALUES (?, ?, datetime('now'), ?)",
            (approved_model, last_verification, status),
        )
        conn.commit()


def get_deployment_status() -> dict:
    ensure_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT approved_model, last_verification, approved_at, status FROM deployment_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return {"approved_model": None, "last_verification": None, "approved_at": None, "status": "pending"}
    return {
        "approved_model": row[0],
        "last_verification": row[1],
        "approved_at": row[2],
        "status": row[3],
    }
