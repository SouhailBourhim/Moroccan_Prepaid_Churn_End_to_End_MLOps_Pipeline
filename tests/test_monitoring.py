"""Drift detection: PSI, effect-size statuses, and DriftDetector on a real prediction log."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.api.logger import PredictionLogger
from src.features.build_features import FeaturePipeline, get_model_features
from src.monitoring.drift import (
    DriftDetector,
    _compare,
    _reference,
    compute_psi,
)
from tests.conftest import make_raw_frame

# ── PSI / reference bins ──────────────────────────────────────────────────────


def test_psi_is_zero_for_identical_distributions() -> None:
    vals = np.random.default_rng(0).normal(size=5000)
    ref = _reference(vals)
    assert compute_psi(vals, vals, ref.edges) == pytest.approx(0.0, abs=1e-9)


def test_binary_features_are_discrete_and_skip_ks() -> None:
    rng = np.random.default_rng(0)
    ref = _reference(rng.integers(0, 2, 5000).astype(float))
    assert ref.kind == "discrete"
    result = _compare("flag", ref, rng.integers(0, 2, 5000).astype(float))
    assert result.ks_statistic is None and result.ks_pvalue is None
    assert result.status == "OK"


def test_large_sample_tiny_shift_is_not_flagged() -> None:
    """A negligible shift is 'significant' by KS p-value at n=50k but must stay OK."""
    rng = np.random.default_rng(0)
    ref = _reference(rng.normal(size=10_000))
    result = _compare("x", ref, rng.normal(loc=0.03, size=50_000))
    assert result.ks_pvalue is not None and result.ks_pvalue < 0.05
    assert result.status == "OK"


def test_real_shift_is_flagged() -> None:
    rng = np.random.default_rng(0)
    ref = _reference(rng.normal(size=10_000))
    result = _compare("x", ref, rng.normal(loc=1.0, size=2_000))
    assert result.status == "DRIFT"


# ── DriftDetector end to end ──────────────────────────────────────────────────


@pytest.fixture
def detector_setup(tmp_path: Path) -> tuple[DriftDetector, PredictionLogger, Any]:
    raw = make_raw_frame(3000, seed=5)
    train_raw, holdout_raw = raw.iloc[:1500], raw.iloc[1500:]
    pipe = FeaturePipeline(top_pack_min_freq=10)
    train_fe = pipe.fit_transform(train_raw.drop(columns=["CHURN"]), train_raw["CHURN"])
    cols = get_model_features(train_fe)
    model = LogisticRegression(max_iter=2000).fit(
        train_fe[cols].to_numpy(dtype=np.float32), train_raw["CHURN"].to_numpy()
    )
    baseline = tmp_path / "holdout_features.parquet"
    pipe.transform(holdout_raw.drop(columns=["CHURN"]))[cols].to_parquet(baseline)

    pred_logger = PredictionLogger(tmp_path / "predictions.db")
    detector = DriftDetector(pipe, model, cols, baseline, pred_logger)
    return detector, pred_logger, (pipe, model, cols)


def _log_rows(
    pred_logger: PredictionLogger, rows: list[dict[str, Any]], probs: list[float]
) -> None:
    pred_logger.log_request(
        request_id=f"r{len(probs)}-{probs[0]:.3f}", model_name="lr",
        n_subscribers=len(rows), threshold=0.5, latency_ms=1.0, probs=probs,
        predictions=[p >= 0.5 for p in probs], raw_inputs=rows,
    )


def _raw_records(n: int, seed: int) -> list[dict[str, Any]]:
    df = make_raw_frame(n, seed=seed).drop(columns=["CHURN", "user_id"])
    records: list[dict[str, Any]] = (
        df.astype(object).where(df.notna(), None).to_dict(orient="records")
    )
    return records


def test_detector_insufficient_data(
    detector_setup: tuple[DriftDetector, PredictionLogger, Any],
) -> None:
    detector, _, _ = detector_setup
    report = detector.detect(hours=24, min_samples=50)
    assert report.overall_status == "INSUFFICIENT_DATA"
    assert report.prediction_drift is None


def test_detector_reports_features_and_prediction_drift(
    detector_setup: tuple[DriftDetector, PredictionLogger, Any],
) -> None:
    detector, pred_logger, (pipe, model, cols) = detector_setup
    records = _raw_records(1500, seed=11)
    X = pipe.transform(make_raw_frame(1500, seed=11).drop(columns=["CHURN"]))[cols]
    probs = model.predict_proba(X.to_numpy(dtype=np.float32))[:, 1].tolist()
    _log_rows(pred_logger, records, probs)

    report = detector.detect(hours=24, min_samples=50)
    assert report.n_live_predictions == 1500
    assert report.n_features_checked > 0
    assert report.prediction_drift is not None
    assert report.prediction_drift.feature == "churn_probability"
    kinds = {f.feature: f.kind for f in report.features}
    assert kinds.get("is_inactive") == "discrete"
    assert kinds.get("regularity_rate") == "continuous"

    # Same generator, same distribution: the scores themselves must not drift.
    assert report.prediction_drift.status == "OK"


def test_detector_flags_shifted_traffic(
    detector_setup: tuple[DriftDetector, PredictionLogger, Any],
) -> None:
    detector, pred_logger, _ = detector_setup
    records = _raw_records(400, seed=12)
    for r in records:
        r["REGULARITY"] = 2.0          # everyone suddenly inactive
    _log_rows(pred_logger, records, [0.9] * len(records))

    report = detector.detect(hours=24, min_samples=50)
    assert report.overall_status == "DRIFT"
    statuses = {f.feature: f.status for f in report.features}
    assert statuses["regularity_rate"] == "DRIFT"
    assert report.prediction_drift is not None
    assert report.prediction_drift.status == "DRIFT"


def test_point_mass_at_different_float_widths_is_not_drift() -> None:
    """An imputed median stored as float64 in the baseline and float32 live is the same value."""
    import pandas as pd

    from src.monitoring.drift import _as_model_precision

    rng = np.random.default_rng(0)
    median = 5036.4512345678
    base = rng.uniform(100, 10_000, 4000)
    base[:1200] = median                                  # 30 % imputed, float64
    live = rng.uniform(100, 10_000, 1500).astype(np.float32)
    live[:450] = np.float32(median)                       # same imputation, float32

    ref = _reference(_as_model_precision(pd.Series(base)))
    result = _compare("MONTANT", ref, _as_model_precision(pd.Series(live)))
    assert result.status == "OK"
