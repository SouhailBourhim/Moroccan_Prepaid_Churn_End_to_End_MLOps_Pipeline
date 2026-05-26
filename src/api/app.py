"""FastAPI application for real-time Expresso churn prediction.

Startup loads the feature pipeline and trained model into app.state once;
every /predict request reuses those in-memory artifacts.

Run locally:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Or via config values in configs/base.yaml:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

Auth:
    Set CHURN_API_KEYS="key1,key2" to enable API-key auth on all protected
    endpoints. Omitting the variable leaves the API open (local dev mode).
    /health is always exempt.

Rate limiting:
    CHURN_RATE_LIMIT_GENERAL (default 200 req/60s) — /ready /info /logs /drift
    CHURN_RATE_LIMIT_PREDICT (default 30 req/60s)  — /predict

Logging:
    Set LOG_FORMAT=json for structured JSON output (production log aggregators).
"""
from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.auth import require_api_key
from src.api.dependencies import ModelArtifacts, load_artifacts
from src.api.logger import PredictionLogger
from src.api.rate_limit import general_limiter, predict_limiter
from src.api.schemas import (
    DriftResponse,
    FeatureDriftResult,
    HealthResponse,
    InfoResponse,
    LogsResponse,
    LogsSummary,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    RecentPrediction,
    SubscriberFeatures,
    SubscriberPrediction,
)
from src.monitoring.drift import DriftDetector, DriftReport
from src.utils.logging import setup_logger

ROOT = Path(__file__).parents[2]
PREDICTIONS_DB = ROOT / "data" / "predictions.db"
FEATURES_DIR = ROOT / "data" / "features"

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

    app.state.pred_logger = PredictionLogger(PREDICTIONS_DB)
    logger.info(f"Prediction logger ready → {PREDICTIONS_DB}")

    if app.state.artifacts is not None:
        art = app.state.artifacts
        app.state.drift_detector = DriftDetector(
            pipeline=art.pipeline,
            feature_cols=art.feature_cols,
            features_dir=FEATURES_DIR,
            pred_db=PREDICTIONS_DB,
        )
        logger.info("Drift detector ready")
    else:
        app.state.drift_detector = None

    yield

    app.state.artifacts = None
    app.state.drift_detector = None


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Expresso Churn Prediction API",
    description=(
        "Real-time prepaid subscriber churn scoring.\n\n"
        "**POST /predict** — score up to 10 000 subscribers per request.\n"
        "**GET  /logs**    — prediction log summary and recent scored rows.\n"
        "**GET  /info**    — model metadata and CV metrics.\n"
        "**GET  /ready**   — readiness probe (503 until model is loaded).\n"
        "**GET  /health**  — liveness probe (always 200)."
    ),
    version="1.2.0",
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
    allow_headers=["Content-Type", "X-API-Key"],
)


# ── Core dependencies ─────────────────────────────────────────────────────────


def get_artifacts() -> ModelArtifacts:
    artifacts: ModelArtifacts | None = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return artifacts


def get_pred_logger() -> PredictionLogger:
    return app.state.pred_logger  # type: ignore[no-any-return]


def get_drift_detector() -> DriftDetector:
    detector: DriftDetector | None = getattr(app.state, "drift_detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialised — model artifacts required.",
        )
    return detector


# ── Rate-limit dependency callables (exported for test overriding) ─────────────


def check_general_rate(request: Request) -> None:
    """Dependency — apply the general rate limit (default 200 req/60s)."""
    general_limiter.check(request)


def check_predict_rate(request: Request) -> None:
    """Dependency — apply the predict rate limit (default 30 req/60s)."""
    predict_limiter.check(request)


# ── Annotated dependency aliases ──────────────────────────────────────────────

ArtifactsDep = Annotated[ModelArtifacts, Depends(get_artifacts)]
LoggerDep = Annotated[PredictionLogger, Depends(get_pred_logger)]
DriftDep = Annotated[DriftDetector, Depends(get_drift_detector)]
KeyDep = Annotated[str, Depends(require_api_key)]
GeneralRateDep = Annotated[None, Depends(check_general_rate)]
PredictRateDep = Annotated[None, Depends(check_predict_rate)]


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
    """Liveness probe — always returns 200. No auth required."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(
    artifacts: ArtifactsDep,
    _key: KeyDep,
    _rate: GeneralRateDep,
) -> ReadyResponse:
    """Readiness probe — 503 until the model is loaded."""
    return ReadyResponse(status="ok", model_loaded=True)


@app.get("/info", response_model=InfoResponse, tags=["ops"])
def info(
    artifacts: ArtifactsDep,
    _key: KeyDep,
    _rate: GeneralRateDep,
) -> InfoResponse:
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
def predict(
    request: PredictionRequest,
    artifacts: ArtifactsDep,
    pred_logger: LoggerDep,
    _key: KeyDep,
    _rate: PredictRateDep,
) -> PredictionResponse:
    """Score a batch of subscribers and return churn probabilities.

    Accepts 1–10 000 subscribers per request. Applies the full feature
    engineering pipeline before scoring, so raw Expresso columns are expected
    (not pre-engineered features). Every scored batch is persisted to the
    prediction log (data/predictions.db).
    """
    t0 = time.perf_counter()

    df = _to_dataframe(request.subscribers)
    X_fe = artifacts.pipeline.transform(df)
    X = X_fe[artifacts.feature_cols].to_numpy(dtype=np.float32)
    probs: np.ndarray = artifacts.model.predict_proba(X)[:, 1]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    predictions = [
        SubscriberPrediction(
            churn_probability=float(p),
            churn_prediction=bool(p >= request.threshold),
        )
        for p in probs
    ]

    model_name: str = artifacts.manifest.get("best_model", "unknown")
    request_id = str(uuid.uuid4())
    raw_inputs: list[dict[str, Any]] = [s.model_dump() for s in request.subscribers]

    pred_logger.log_request(
        request_id=request_id,
        model_name=model_name,
        n_subscribers=len(predictions),
        threshold=request.threshold,
        latency_ms=latency_ms,
        probs=[float(p) for p in probs],
        predictions=[bool(p >= request.threshold) for p in probs],
        raw_inputs=raw_inputs,
    )

    logger.debug(
        f"Scored {len(predictions)} subscribers  "
        f"threshold={request.threshold}  "
        f"flagged={sum(s.churn_prediction for s in predictions)}  "
        f"latency={latency_ms:.1f}ms  "
        f"request_id={request_id}"
    )
    return PredictionResponse(
        predictions=predictions,
        model_name=model_name,
        threshold=request.threshold,
        n_subscribers=len(predictions),
    )


@app.get("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift(
    detector: DriftDep,
    _key: KeyDep,
    _rate: GeneralRateDep,
    hours: int = Query(default=24, ge=1, le=720, description="Look-back window in hours"),
    min_samples: int = Query(default=50, ge=1, description="Minimum live rows required"),
) -> DriftResponse:
    """Return per-feature drift status comparing recent /predict traffic to training baseline.

    Uses PSI (Population Stability Index) and KS 2-sample test.
    Returns overall_status='INSUFFICIENT_DATA' when fewer than min_samples rows
    have been logged in the requested window.
    """
    report: DriftReport = detector.detect(hours=hours, min_samples=min_samples)
    return DriftResponse(
        report_time=report.report_time,
        window_hours=report.window_hours,
        n_live_predictions=report.n_live_predictions,
        n_features_checked=report.n_features_checked,
        n_drifted=report.n_drifted,
        n_warned=report.n_warned,
        overall_status=report.overall_status,
        features=[
            FeatureDriftResult(
                feature=f.feature,
                psi=f.psi,
                ks_statistic=f.ks_statistic,
                ks_pvalue=f.ks_pvalue,
                status=f.status,
                n_live=f.n_live,
            )
            for f in report.features
        ],
    )


@app.get("/logs", response_model=LogsResponse, tags=["ops"])
def logs(
    pred_logger: LoggerDep,
    _key: KeyDep,
    _rate: GeneralRateDep,
    hours: int = Query(default=24, ge=1, le=720, description="Summary window in hours"),
    limit: int = Query(default=50, ge=1, le=500, description="Max recent rows to return"),
) -> LogsResponse:
    """Return prediction log summary stats and the most recent scored rows.

    - `hours`: time window for the 'predictions_last_Nh' counter (default 24)
    - `limit`: number of recent rows to include (default 50, max 500)
    """
    raw_summary = pred_logger.summary(since_hours=hours)
    raw_recent = pred_logger.recent(limit=limit)

    summary = LogsSummary(
        total_predictions=raw_summary["total_predictions"],
        total_requests=raw_summary["total_requests"],
        mean_churn_probability=raw_summary["mean_churn_probability"],
        churn_flag_rate=raw_summary["churn_flag_rate"],
        mean_latency_ms=raw_summary["mean_latency_ms"],
    )
    recent_rows = [RecentPrediction(**row) for row in raw_recent]

    return LogsResponse(summary=summary, recent=recent_rows)
