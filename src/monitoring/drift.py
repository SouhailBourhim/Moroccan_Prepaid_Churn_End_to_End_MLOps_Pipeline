"""Feature and prediction drift detection for the churn prediction API.

Compares live traffic (from the prediction log) against the labeled holdout —
data that went through exactly the serving transform, unlike the training
split whose target encodings are cross-fitted.

Status is driven by effect sizes, not p-values. With thousands of live rows a
KS p-value flags shifts far too small to matter, and testing ~40 features at
p < 0.05 would raise a false WARN on almost every report.

  PSI  (Population Stability Index), every feature
       < 0.1 → OK  |  0.1–0.2 → WARN  |  > 0.2 → DRIFT
  KS statistic D (max CDF gap), continuous features only
       < 0.1 → OK  |  0.1–0.2 → WARN  |  > 0.2 → DRIFT

Discrete features (binary flags, counts, ordinals — at most DISCRETE_MAX_VALUES
distinct values in the baseline) are binned on their own values and judged by
PSI alone; a KS test is not meaningful for them. The KS p-value is still
reported for continuous features, for reference only.

Prediction drift — the served churn_probability vs. the model's scores on the
holdout — is reported separately and counts toward the overall status.
Overall status is DRIFT if anything drifts, WARN if anything warns, else OK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.ingestion import records_to_frame
from src.features.build_features import FeaturePipeline

# ── Constants ─────────────────────────────────────────────────────────────────

PSI_BINS = 10
PSI_WARN = 0.10
PSI_DRIFT = 0.20
KS_STAT_WARN = 0.10
KS_STAT_DRIFT = 0.20
DISCRETE_MAX_VALUES = 12
BASELINE_SAMPLE = 10_000   # holdout rows used as the reference distribution
MAX_LIVE_ROWS = 50_000     # most recent live rows considered per report
MIN_LIVE_DEFAULT = 50      # minimum live rows before reporting
PREDICTION_FEATURE = "churn_probability"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    feature: str
    kind: str                    # "continuous" | "discrete"
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    status: str                  # "OK" | "WARN" | "DRIFT"
    n_live: int


@dataclass
class DriftReport:
    report_time: str
    window_hours: int
    n_live_predictions: int
    n_features_checked: int
    n_drifted: int
    n_warned: int
    overall_status: str  # "OK" | "WARN" | "DRIFT" | "INSUFFICIENT_DATA"
    features: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: FeatureDriftResult | None = None


class LiveRowSource(Protocol):
    """Where live traffic comes from (PredictionLogger implements this)."""

    def fetch_window(
        self, since_hours: int, max_rows: int
    ) -> list[tuple[dict[str, Any], float]]: ...


@dataclass
class _Reference:
    values: np.ndarray
    edges: np.ndarray
    kind: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_psi(
    baseline: np.ndarray,
    live: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Population Stability Index using pre-computed baseline bin edges."""
    eps = 1e-6
    exp_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    act_counts = np.histogram(live, bins=bin_edges)[0].astype(float)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _reference(values: np.ndarray) -> _Reference:
    """Bin edges for one baseline column: per-value for discrete, deciles otherwise."""
    uniq = np.unique(values)
    if len(uniq) <= DISCRETE_MAX_VALUES:
        if len(uniq) == 1:
            edges = np.array([-np.inf, np.inf])
        else:
            mids = (uniq[:-1] + uniq[1:]) / 2
            edges = np.concatenate([[-np.inf], mids, [np.inf]])
        return _Reference(values, edges, "discrete")
    edges = np.unique(np.percentile(values, np.linspace(0, 100, PSI_BINS + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    return _Reference(values, edges, "continuous")


def _severity(value: float, warn: float, drift: float) -> int:
    return 2 if value > drift else 1 if value > warn else 0


_STATUS = ("OK", "WARN", "DRIFT")


def _compare(name: str, ref: _Reference, live: np.ndarray) -> FeatureDriftResult:
    psi = compute_psi(ref.values, live, ref.edges)
    severity = _severity(psi, PSI_WARN, PSI_DRIFT)
    ks_stat: float | None = None
    ks_p: float | None = None
    if ref.kind == "continuous":
        ks = ks_2samp(ref.values, live)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        severity = max(severity, _severity(ks_stat, KS_STAT_WARN, KS_STAT_DRIFT))
    return FeatureDriftResult(
        feature=name,
        kind=ref.kind,
        psi=round(psi, 4),
        ks_statistic=round(ks_stat, 4) if ks_stat is not None else None,
        ks_pvalue=round(ks_p, 4) if ks_p is not None else None,
        status=_STATUS[severity],
        n_live=int(len(live)),
    )


# ── DriftDetector ─────────────────────────────────────────────────────────────


class DriftDetector:
    """Per-feature and prediction drift between live traffic and the holdout baseline."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        model: Any,
        feature_cols: list[str],
        baseline_path: Path,
        live_source: LiveRowSource,
    ) -> None:
        self.pipeline = pipeline
        self.feature_cols = feature_cols
        self.live_source = live_source

        df = pd.read_parquet(baseline_path)[feature_cols]
        if len(df) > BASELINE_SAMPLE:
            df = df.sample(n=BASELINE_SAMPLE, random_state=42)

        self._refs: dict[str, _Reference] = {}
        for col in feature_cols:
            vals = df[col].dropna().to_numpy(dtype=np.float64)
            if len(vals):
                self._refs[col] = _reference(vals)

        scores = model.predict_proba(df.to_numpy(dtype=np.float32))[:, 1]
        self._pred_ref = _reference(np.asarray(scores, dtype=np.float64))

    def detect(
        self,
        hours: int = 24,
        min_samples: int = MIN_LIVE_DEFAULT,
    ) -> DriftReport:
        """Run drift detection over the most recent `hours` of prediction traffic."""
        report_time = datetime.now(UTC).isoformat()
        live_rows = self.live_source.fetch_window(hours, MAX_LIVE_ROWS)
        n_live = len(live_rows)

        if n_live < min_samples:
            return DriftReport(
                report_time=report_time,
                window_hours=hours,
                n_live_predictions=n_live,
                n_features_checked=0,
                n_drifted=0,
                n_warned=0,
                overall_status="INSUFFICIENT_DATA",
                features=[],
            )

        raw_df = records_to_frame([raw for raw, _ in live_rows])
        live_df = self.pipeline.transform(raw_df)[self.feature_cols]

        results: list[FeatureDriftResult] = []
        for col, ref in self._refs.items():
            live_vals = live_df[col].dropna().to_numpy(dtype=np.float64)
            if len(live_vals):
                results.append(_compare(col, ref, live_vals))

        order = {"DRIFT": 0, "WARN": 1, "OK": 2}
        results.sort(key=lambda r: (order[r.status], r.feature))

        live_probs = np.array([p for _, p in live_rows], dtype=np.float64)
        prediction = _compare(PREDICTION_FEATURE, self._pred_ref, live_probs)

        statuses = [r.status for r in results] + [prediction.status]
        if "DRIFT" in statuses:
            overall = "DRIFT"
        elif "WARN" in statuses:
            overall = "WARN"
        else:
            overall = "OK"

        return DriftReport(
            report_time=report_time,
            window_hours=hours,
            n_live_predictions=n_live,
            n_features_checked=len(results),
            n_drifted=sum(1 for r in results if r.status == "DRIFT"),
            n_warned=sum(1 for r in results if r.status == "WARN"),
            overall_status=overall,
            features=results,
            prediction_drift=prediction,
        )
