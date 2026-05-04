"""FastAPI application for real-time Expresso churn prediction.

Startup loads the feature pipeline and trained model into app.state once;
every /predict request reuses those in-memory artifacts.

Run locally:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Or via config values in configs/base.yaml:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.dependencies import ModelArtifacts, load_artifacts
from src.api.schemas import (
    HealthResponse,
    InfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    SubscriberFeatures,
    SubscriberPrediction,
)
from src.utils.logging import setup_logger

# ── Column dtype maps (mirrors ingestion.py) ──────────────────────────────────

_CAT_COLS = {"REGION", "TENURE", "MRG", "TOP_PACK"}
_FLOAT_COLS = {
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2",
    "REGULARITY", "FREQ_TOP_PACK",
}


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger()
    try:
        app.state.artifacts = load_artifacts()
        logger.info("API ready")
    except FileNotFoundError as exc:
        logger.warning(f"Model artifacts not found at startup — /predict will return 503. {exc}")
        app.state.artifacts = None
    yield
    app.state.artifacts = None


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Expresso Churn Prediction API",
    description=(
        "Real-time prepaid subscriber churn scoring.\n\n"
        "**POST /predict** — score up to 10 000 subscribers per request.\n"
        "**GET  /info**    — model metadata and CV metrics.\n"
        "**GET  /ready**   — readiness probe (503 until model is loaded).\n"
        "**GET  /health**  — liveness probe (always 200)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── Dependency ────────────────────────────────────────────────────────────────


def get_artifacts() -> ModelArtifacts:
    artifacts: ModelArtifacts | None = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return artifacts


ArtifactsDep = Annotated[ModelArtifacts, Depends(get_artifacts)]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_dataframe(subscribers: list[SubscriberFeatures]) -> pd.DataFrame:
    """Convert a list of SubscriberFeatures to a typed DataFrame for the pipeline."""
    rows: list[dict[str, Any]] = [s.model_dump() for s in subscribers]
    df = pd.DataFrame(rows)
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — always returns 200."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(artifacts: ArtifactsDep) -> ReadyResponse:
    """Readiness probe — 503 until the model is loaded."""
    return ReadyResponse(status="ok", model_loaded=True)


@app.get("/info", response_model=InfoResponse, tags=["ops"])
def info(artifacts: ArtifactsDep) -> InfoResponse:
    """Return model name, CV ROC-AUC, and feature count from the training manifest."""
    m = artifacts.manifest
    return InfoResponse(
        model_name=m.get("best_model", "unknown"),
        cv_roc_auc_mean=float(m.get("cv_roc_auc_mean", 0.0)),
        cv_roc_auc_std=float(m.get("cv_roc_auc_std", 0.0)),
        cv_pr_auc_mean=float(m.get("cv_pr_auc_mean", 0.0)),
        n_features=int(m.get("n_features", 0)),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest, artifacts: ArtifactsDep) -> PredictionResponse:
    """Score a batch of subscribers and return churn probabilities.

    Accepts 1–10 000 subscribers per request. Applies the full feature
    engineering pipeline before scoring, so raw Expresso columns are expected
    (not pre-engineered features).
    """
    df = _to_dataframe(request.subscribers)
    X_fe = artifacts.pipeline.transform(df)
    X = X_fe[artifacts.feature_cols].to_numpy(dtype=np.float32)
    probs: np.ndarray = artifacts.model.predict_proba(X)[:, 1]

    predictions = [
        SubscriberPrediction(
            churn_probability=float(p),
            churn_prediction=bool(p >= request.threshold),
        )
        for p in probs
    ]

    logger.debug(
        f"Scored {len(predictions)} subscribers  "
        f"threshold={request.threshold}  "
        f"flagged={sum(s.churn_prediction for s in predictions)}"
    )
    return PredictionResponse(
        predictions=predictions,
        model_name=artifacts.manifest.get("best_model", "unknown"),
        threshold=request.threshold,
        n_subscribers=len(predictions),
    )
