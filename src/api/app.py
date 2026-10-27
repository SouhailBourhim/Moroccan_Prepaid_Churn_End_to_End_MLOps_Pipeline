"""FastAPI application for real-time prepaid churn prediction.

Startup loads the feature pipeline and the calibrated model for one dataset
(models/{dataset}/model.pkl) into app.state once; every /predict request reuses
those in-memory artifacts.

Run locally:
    CHURN_AUTH_DISABLED=1 uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Environment:
    CHURN_DATASET            dataset whose model is served (default: api.dataset in
                             configs/base.yaml)
    CHURN_API_KEYS           comma-separated API keys for the X-API-Key header
    CHURN_AUTH_DISABLED=1    run without auth (local development only); with neither
                             set, protected endpoints return 503
    CHURN_PREDICTIONS_DB     SQLite prediction log path (default data/predictions.db)
    CHURN_LOG_RETENTION_DAYS prediction-log retention, pruned at startup (default 30)
    CHURN_RATE_LIMIT_GENERAL / CHURN_RATE_LIMIT_PREDICT / CHURN_TRUSTED_PROXIES
                             see src/api/rate_limit.py
    LOG_FORMAT=json          structured JSON logs for production log aggregators
"""
from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.auth import auth_disabled, require_api_key
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
    SubscriberPrediction,
)
from src.data.ingestion import records_to_frame
from src.monitoring.drift import DriftDetector, DriftReport
from src.monitoring.drift import FeatureDriftResult as DriftResult
from src.utils.config import ROOT, dataset_paths, load_config
from src.utils.logging import setup_logger


def _served_dataset() -> str:
    return os.environ.get("CHURN_DATASET") or str(
        load_config().get("api", {}).get("dataset", "expresso")
    )


PREDICTIONS_DB = Path(os.environ.get("CHURN_PREDICTIONS_DB", ROOT / "data" / "predictions.db"))


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger()
    dataset = _served_dataset()
    if auth_disabled():
        logger.warning("CHURN_AUTH_DISABLED is set — API key auth is OFF (development only)")
    try:
        app.state.artifacts = load_artifacts(dataset)
        logger.info(f"API ready (dataset={dataset})")
    except FileNotFoundError as exc:
        logger.warning(f"Model artifacts not found at startup — /predict will return 503. {exc}")
        app.state.artifacts = None

    pred_logger = PredictionLogger(PREDICTIONS_DB)
    retention_days = int(os.environ.get("CHURN_LOG_RETENTION_DAYS", "30"))
    pruned = pred_logger.prune(retention_days)
    app.state.pred_logger = pred_logger
    logger.info(
        f"Prediction logger ready → {PREDICTIONS_DB}  "
        f"(retention {retention_days}d, pruned {pruned} rows)"
    )

    app.state.drift_detector = None
    art: ModelArtifacts | None = app.state.artifacts
    if art is not None:
        baseline = dataset_paths(dataset).holdout_features
        if baseline.exists():
            app.state.drift_detector = DriftDetector(
                pipeline=art.pipeline,
                model=art.model,
                feature_cols=art.feature_cols,
                baseline_path=baseline,
                live_source=pred_logger,
            )
            logger.info("Drift detector ready")
        else:
            logger.warning(f"Drift baseline {baseline} not found — /drift will return 503")

    yield

    app.state.artifacts = None
    app.state.drift_detector = None


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Prepaid Churn Prediction API",
    description=(
        "Real-time prepaid subscriber churn scoring.\n\n"
        "**POST /predict** — score up to 10 000 subscribers per request.\n"
        "**GET  /logs**    — prediction log summary and recent scored rows.\n"
        "**GET  /drift**   — feature and prediction drift vs. the holdout baseline.\n"
        "**GET  /info**    — model metadata and CV metrics.\n"
        "**GET  /ready**   — readiness probe (503 until model is loaded).\n"
        "**GET  /health**  — liveness probe (always 200)."
    ),
    version="2.0.0",
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
            detail="Drift detector not initialised — needs model artifacts and holdout baseline.",
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


def _drift_result(r: DriftResult) -> FeatureDriftResult:
    return FeatureDriftResult(
        feature=r.feature,
        kind=r.kind,
        psi=r.psi,
        ks_statistic=r.ks_statistic,
        ks_pvalue=r.ks_pvalue,
        status=r.status,
        n_live=r.n_live,
    )


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
    """Return model name, dataset, calibration, default threshold, and CV metrics."""
    m = artifacts.manifest
    return InfoResponse(
        model_name=m.get("best_model", "unknown"),
        dataset=artifacts.dataset,
        calibration_method=artifacts.calibration_method,
        default_threshold=artifacts.default_threshold,
        cv_roc_auc_mean=float(m.get("cv_roc_auc_mean", 0.0)),
        cv_roc_auc_std=float(m.get("cv_roc_auc_std", 0.0)),
        cv_pr_auc_mean=float(m.get("cv_pr_auc_mean", 0.0)),
        n_features=int(m.get("n_features", len(artifacts.feature_cols))),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    request: PredictionRequest,
    background: BackgroundTasks,
    artifacts: ArtifactsDep,
    pred_logger: LoggerDep,
    _key: KeyDep,
    _rate: PredictRateDep,
) -> PredictionResponse:
    """Score a batch of subscribers and return calibrated churn probabilities.

    Accepts 1–10 000 subscribers per request. Applies the full feature
    engineering pipeline before scoring, so raw columns are expected (not
    pre-engineered features). Without an explicit `threshold`, labels use the
    model's F1-optimal threshold chosen during training. Every scored batch is
    written to the prediction log after the response is sent.
    """
    t0 = time.perf_counter()

    raw_inputs: list[dict[str, Any]] = [s.model_inputs() for s in request.subscribers]
    df = records_to_frame(raw_inputs)
    X_fe = artifacts.pipeline.transform(df)
    X = X_fe[artifacts.feature_cols].to_numpy(dtype=np.float32)
    probs: np.ndarray = artifacts.model.predict_proba(X)[:, 1]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    threshold = (
        request.threshold if request.threshold is not None else artifacts.default_threshold
    )
    subscriber_ids = [s.subscriber_id for s in request.subscribers]
    flags = [bool(p >= threshold) for p in probs]

    predictions = [
        SubscriberPrediction(
            subscriber_id=sid,
            churn_probability=float(p),
            churn_prediction=flag,
        )
        for sid, p, flag in zip(subscriber_ids, probs, flags, strict=True)
    ]

    model_name: str = artifacts.manifest.get("best_model", "unknown")
    request_id = str(uuid.uuid4())

    background.add_task(
        pred_logger.log_request,
        request_id=request_id,
        model_name=model_name,
        n_subscribers=len(predictions),
        threshold=threshold,
        latency_ms=latency_ms,
        probs=[float(p) for p in probs],
        predictions=flags,
        raw_inputs=raw_inputs,
        subscriber_ids=subscriber_ids,
    )

    logger.debug(
        f"Scored {len(predictions)} subscribers  "
        f"threshold={threshold:.3f}  "
        f"flagged={sum(flags)}  "
        f"latency={latency_ms:.1f}ms  "
        f"request_id={request_id}"
    )
    return PredictionResponse(
        request_id=request_id,
        predictions=predictions,
        model_name=model_name,
        threshold=threshold,
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
    """Per-feature and prediction drift of recent /predict traffic vs. the holdout baseline.

    Status uses PSI (all features) and the KS statistic (continuous features).
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
        features=[_drift_result(f) for f in report.features],
        prediction_drift=(
            _drift_result(report.prediction_drift) if report.prediction_drift else None
        ),
    )


@app.get("/logs", response_model=LogsResponse, tags=["ops"])
def logs(
    pred_logger: LoggerDep,
    _key: KeyDep,
    _rate: GeneralRateDep,
    hours: int = Query(default=24, ge=1, le=720, description="Summary window in hours"),
    limit: int = Query(default=50, ge=1, le=500, description="Max recent rows to return"),
) -> LogsResponse:
    """Return prediction-log statistics and the most recent scored rows.

    - `hours`: window for the traffic statistics (counts, mean probability,
      flag rate, latency). total_predictions / total_requests are all-time.
    - `limit`: number of recent rows to include (default 50, max 500)
    """
    summary = LogsSummary(**pred_logger.summary(since_hours=hours))
    recent_rows = [RecentPrediction(**row) for row in pred_logger.recent(limit=limit)]
    return LogsResponse(summary=summary, recent=recent_rows)
