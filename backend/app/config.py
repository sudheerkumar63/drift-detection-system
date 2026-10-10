from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Deep Learning Drift Detection System"
    api_prefix: str = "/api"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    model_dir: str = "model_weights"
    data_dir: str = "data"
    db_path: str = "data/drift.db"
    max_upload_size: int = 10 * 1024 * 1024
    cors_origins: str = "*"
    secret_key: str = "change-me"

    class Config:
        env_file = BASE_DIR / ".env"


settings = Settings()
