"""Tests for the FastAPI churn prediction endpoints.

All tests use a lightweight mock of ModelArtifacts so no real model files
are required — the test suite stays runnable in CI without ML artifacts.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, get_artifacts, get_pred_logger
from src.api.dependencies import ModelArtifacts
from src.api.logger import PredictionLogger

# ── Mock artifacts ────────────────────────────────────────────────────────────


class _MockPipeline:
    """Minimal pipeline stub: returns a DataFrame with the expected feature columns."""

    def __init__(self, feature_cols: list[str]) -> None:
        self._cols = feature_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self._cols:
            if col not in out.columns:
                out[col] = 0.0
        return out


class _MockModel:
    """Always predicts churn_probability = 0.3."""

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])


FEATURE_COLS = ["regularity_rate", "total_calls", "revenue_per_freq"]

MOCK_MANIFEST: dict[str, Any] = {
    "best_model": "catboost",
    "cv_roc_auc_mean": 0.9316,
    "cv_roc_auc_std": 0.0005,
    "cv_pr_auc_mean": 0.7039,
    "n_features": len(FEATURE_COLS),
}

MOCK_ARTIFACTS = ModelArtifacts(
    pipeline=_MockPipeline(FEATURE_COLS),  # type: ignore[arg-type]
    model=_MockModel(),
    feature_cols=FEATURE_COLS,
    manifest=MOCK_MANIFEST,
)

VALID_SUBSCRIBER: dict[str, Any] = {
    "REGION": "DAKAR",
    "TENURE": "K > 24 month",
    "MRG": "NO",
    "REGULARITY": 54.0,
    "ON_NET": 388.0,
    "REVENUE": 4251.0,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_logger() -> PredictionLogger:
    """A real PredictionLogger backed by a temporary file-based SQLite DB."""
    with tempfile.TemporaryDirectory() as tmp:
        yield PredictionLogger(Path(tmp) / "test_predictions.db")


@pytest.fixture
def client(tmp_logger: PredictionLogger) -> TestClient:
    """TestClient with mock artifacts and a temp prediction logger."""
    app.dependency_overrides[get_artifacts] = lambda: MOCK_ARTIFACTS
    app.dependency_overrides[get_pred_logger] = lambda: tmp_logger
    c = TestClient(app, raise_server_exceptions=True)
    yield c  # type: ignore[misc]
    app.dependency_overrides.clear()


@pytest.fixture
def unready_client(tmp_logger: PredictionLogger) -> TestClient:
    """TestClient with NO artifacts — simulates startup failure."""
    from fastapi import HTTPException

    def _raise() -> ModelArtifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    app.dependency_overrides[get_artifacts] = _raise
    app.dependency_overrides[get_pred_logger] = lambda: tmp_logger
    c = TestClient(app, raise_server_exceptions=False)
    yield c  # type: ignore[misc]
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_always_200(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_200_without_model(unready_client: TestClient) -> None:
    r = unready_client.get("/health")
    assert r.status_code == 200


# ── /ready ────────────────────────────────────────────────────────────────────


def test_ready_200_when_model_loaded(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_ready_503_when_model_missing(unready_client: TestClient) -> None:
    r = unready_client.get("/ready")
    assert r.status_code == 503


# ── /info ─────────────────────────────────────────────────────────────────────


def test_info_returns_manifest_fields(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "catboost"
    assert body["cv_roc_auc_mean"] == pytest.approx(0.9316)
    assert body["n_features"] == len(FEATURE_COLS)


def test_info_503_when_model_missing(unready_client: TestClient) -> None:
    r = unready_client.get("/info")
    assert r.status_code == 503


# ── /predict ──────────────────────────────────────────────────────────────────


def test_predict_single_subscriber(client: TestClient) -> None:
    r = client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_subscribers"] == 1
    assert body["model_name"] == "catboost"
    assert body["threshold"] == pytest.approx(0.5)
    pred = body["predictions"][0]
    assert pred["churn_probability"] == pytest.approx(0.3)
    assert pred["churn_prediction"] is False  # 0.3 < 0.5


def test_predict_batch(client: TestClient) -> None:
    r = client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER] * 10})
    assert r.status_code == 200
    assert r.json()["n_subscribers"] == 10
    assert len(r.json()["predictions"]) == 10


def test_predict_custom_threshold_flips_label(client: TestClient) -> None:
    # Mock always returns 0.3; threshold 0.2 should flip prediction to True
    r = client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER], "threshold": 0.2})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["churn_prediction"] is True


def test_predict_all_null_subscriber(client: TestClient) -> None:
    """A subscriber with all-None fields is valid — the pipeline handles MNAR."""
    r = client.post("/predict", json={"subscribers": [{}]})
    assert r.status_code == 200
    assert 0.0 <= r.json()["predictions"][0]["churn_probability"] <= 1.0


def test_predict_503_when_model_missing(unready_client: TestClient) -> None:
    r = unready_client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER]})
    assert r.status_code == 503


def test_predict_empty_list_rejected(client: TestClient) -> None:
    r = client.post("/predict", json={"subscribers": []})
    assert r.status_code == 422


def test_predict_invalid_threshold_rejected(client: TestClient) -> None:
    r = client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER], "threshold": 1.5})
    assert r.status_code == 422


def test_predict_invalid_regularity_rejected(client: TestClient) -> None:
    bad = {**VALID_SUBSCRIBER, "REGULARITY": 150.0}  # > 90
    r = client.post("/predict", json={"subscribers": [bad]})
    assert r.status_code == 422


# ── /logs ─────────────────────────────────────────────────────────────────────


def test_logs_empty_on_fresh_db(client: TestClient) -> None:
    r = client.get("/logs")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_predictions"] == 0
    assert body["summary"]["total_requests"] == 0
    assert body["recent"] == []


def test_logs_records_predict_calls(client: TestClient) -> None:
    # Make two prediction requests
    client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER]})
    client.post("/predict", json={"subscribers": [VALID_SUBSCRIBER] * 3})

    r = client.get("/logs")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_predictions"] == 4
    assert body["summary"]["total_requests"] == 2
    assert len(body["recent"]) == 4
    # Each recent row has required fields
    row = body["recent"][0]
    assert "churn_probability" in row
    assert "churn_prediction" in row
    assert "latency_ms" in row
