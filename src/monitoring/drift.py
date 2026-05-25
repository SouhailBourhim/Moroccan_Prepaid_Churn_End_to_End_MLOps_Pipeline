"""Feature drift detection for the churn prediction API.

Compares live subscriber feature distributions (from the prediction log) against
the training baseline using two complementary metrics:

  PSI  (Population Stability Index) — measures distributional shift on a binned
       scale; industry standard for model monitoring.
       < 0.1  → OK   |  0.1–0.2 → WARN  |  > 0.2 → DRIFT

  KS   (Kolmogorov–Smirnov 2-sample test) — non-parametric; detects shape shifts
       that PSI can miss when distributions have similar means but different tails.
       p > 0.05 → OK  |  0.01–0.05 → WARN  |  p < 0.01 → DRIFT

Per-feature status is set to the stricter of the two metrics.
Overall report status is DRIFT if any feature drifts, WARN if any warn, else OK.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features.build_features import FeaturePipeline

# ── Constants ─────────────────────────────────────────────────────────────────

PSI_BINS = 10
PSI_WARN = 0.10
PSI_DRIFT = 0.20
KS_WARN = 0.05
KS_DRIFT = 0.01
BASELINE_SAMPLE = 10_000   # rows sampled from train_features.parquet for KS test
MIN_LIVE_DEFAULT = 50      # minimum live rows before reporting

_CAT_COLS = {"REGION", "TENURE", "MRG", "TOP_PACK"}
_FLOAT_COLS = {
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2",
    "REGULARITY", "FREQ_TOP_PACK",
}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: str         # "OK" | "WARN" | "DRIFT"
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


# ── PSI helper ────────────────────────────────────────────────────────────────


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


# ── Feature status helper ──────────────────────────────────────────────────────


def _feature_status(psi: float, ks_pvalue: float) -> str:
    if psi > PSI_DRIFT or ks_pvalue < KS_DRIFT:
        return "DRIFT"
    if psi > PSI_WARN or ks_pvalue < KS_WARN:
        return "WARN"
    return "OK"


# ── Raw-input → DataFrame (mirrors app._to_dataframe) ────────────────────────


def _raw_inputs_to_df(raw_inputs: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw_inputs)
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


# ── DriftDetector ─────────────────────────────────────────────────────────────


class DriftDetector:
    """Computes per-feature drift between live prediction inputs and training baseline."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        feature_cols: list[str],
        features_dir: Path,
        pred_db: Path,
    ) -> None:
        self.pipeline = pipeline
        self.feature_cols = feature_cols
        self.pred_db = pred_db

        # Precompute baseline arrays and bin edges from a sample of train features
        self._baseline, self._bin_edges = self._build_baseline(features_dir)

    def _build_baseline(
        self, features_dir: Path
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        df = pd.read_parquet(features_dir / "train_features.parquet")[self.feature_cols]
        if len(df) > BASELINE_SAMPLE:
            df = df.sample(n=BASELINE_SAMPLE, random_state=42)

        baseline: dict[str, np.ndarray] = {}
        bin_edges: dict[str, np.ndarray] = {}
        for col in self.feature_cols:
            vals = df[col].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            baseline[col] = vals
            edges = np.percentile(vals, np.linspace(0, 100, PSI_BINS + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            bin_edges[col] = edges

        return baseline, bin_edges

    @contextmanager
    def _db_connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.pred_db, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def _fetch_live_raw(self, since_hours: int) -> list[dict[str, Any]]:
        if not self.pred_db.exists():
            return []
        since = (
            datetime.now(UTC) - timedelta(hours=since_hours)
        ).isoformat()
        with self._db_connect() as conn:
            rows = conn.execute(
                "SELECT r.features_json FROM prediction_rows r "
                "JOIN prediction_requests q ON r.request_id = q.request_id "
                "WHERE q.ts >= ?",
                (since,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def detect(
        self,
        hours: int = 24,
        min_samples: int = MIN_LIVE_DEFAULT,
    ) -> DriftReport:
        """Run drift detection over the last `hours` of prediction traffic."""
        report_time = datetime.now(UTC).isoformat()
        raw_inputs = self._fetch_live_raw(hours)
        n_live = len(raw_inputs)

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

        # Transform raw inputs → engineered features (same pipeline as /predict)
        live_raw_df = _raw_inputs_to_df(raw_inputs)
        live_eng_df = self.pipeline.transform(live_raw_df)[self.feature_cols]

        results: list[FeatureDriftResult] = []
        for col in self.feature_cols:
            if col not in self._baseline:
                continue
            baseline_vals = self._baseline[col]
            live_vals = live_eng_df[col].dropna().to_numpy(dtype=np.float64)
            if len(live_vals) == 0:
                continue

            psi = compute_psi(baseline_vals, live_vals, self._bin_edges[col])
            ks_stat, ks_pval = ks_2samp(baseline_vals, live_vals)
            status = _feature_status(psi, ks_pval)

            results.append(
                FeatureDriftResult(
                    feature=col,
                    psi=round(psi, 4),
                    ks_statistic=round(float(ks_stat), 4),
                    ks_pvalue=round(float(ks_pval), 4),
                    status=status,
                    n_live=int(len(live_vals)),
                )
            )

        # Sort: DRIFT first, then WARN, then OK; alphabetical within each group
        _order = {"DRIFT": 0, "WARN": 1, "OK": 2}
        results.sort(key=lambda r: (_order[r.status], r.feature))

        n_drifted = sum(1 for r in results if r.status == "DRIFT")
        n_warned = sum(1 for r in results if r.status == "WARN")

        if n_drifted > 0:
            overall = "DRIFT"
        elif n_warned > 0:
            overall = "WARN"
        else:
            overall = "OK"

        return DriftReport(
            report_time=report_time,
            window_hours=hours,
            n_live_predictions=n_live,
            n_features_checked=len(results),
            n_drifted=n_drifted,
            n_warned=n_warned,
            overall_status=overall,
            features=results,
        )
