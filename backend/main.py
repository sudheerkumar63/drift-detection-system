from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.config import settings
from backend.app.database import ensure_database

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Research dashboard for traffic sign recognition drift detection and adaptation.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    ensure_database()


@app.get("/")
def root() -> dict:
    return {"message": "Deep Learning Drift Detection System API is running."}
