"""Post-hoc probability calibration for the trained churn model.

Fits an isotonic or sigmoid (Platt) calibrator on a held-out calibration
split carved from train_features.parquet, then evaluates on the test holdout.

The calibrated model is wrapped in CalibratedChurnModel which exposes the
same predict_proba(X) interface as the original, making it a drop-in
replacement in best_model.pkl / the FastAPI serving layer.

Usage:
    python -m src.models.calibrate                          # moroccan, both methods
    python -m src.models.calibrate --dataset expresso
    python -m src.models.calibrate --method isotonic
    python -m src.models.calibrate --method sigmoid
    python -m src.models.calibrate --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.models.evaluate import compute_metrics, optimal_threshold
from src.utils.logging import setup_logger

ROOT = Path(__file__).parents[2]
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "configs" / "base.yaml"

CAL_SIZE = 0.20
CAL_SEED = 123
HOLDOUT_SEED = 42


class CalibratedChurnModel:
    """Wraps a pre-fitted base model with a 1-D probability calibrator.

    Exposes predict_proba(X) returning an (n, 2) array so it is a drop-in
    replacement for any sklearn-compatible classifier in the serving layer.
    """

    def __init__(
        self,
        base_model: Any,
        calibrator: IsotonicRegression | LogisticRegression,
        method: str,
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw: np.ndarray = self.base_model.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            cal: np.ndarray = self.calibrator.predict_proba(  # type: ignore[union-attr]
                raw.reshape(-1, 1)
            )[:, 1]
        else:
            cal = self.calibrator.predict(raw)  # type: ignore[union-attr]
        return np.column_stack([1.0 - cal, cal])

    def __repr__(self) -> str:
        return (
            f"CalibratedChurnModel(base={self.base_model.__class__.__name__}, "
            f"method={self.method})"
        )


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _load_splits(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X_cal, y_cal, X_test, y_test, feature_cols)."""
    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")

    if dataset == "moroccan":
        test_df = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
        if "CHURN" not in test_df.columns:
            raise RuntimeError(
                "test_features.parquet has no CHURN column. "
                "Re-run: python -m src.features.run_pipeline --dataset moroccan"
            )
        feature_cols = [c for c in train_df.columns if c not in ("CHURN", "user_id")]
        X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train_df["CHURN"].to_numpy()
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df["CHURN"].to_numpy()
    else:
        feature_cols = [c for c in train_df.columns if c != "CHURN"]
        X_all = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_all = train_df["CHURN"].to_numpy()
        X_tr, X_test, y_tr, y_test = train_test_split(
            X_all, y_all, test_size=0.20, stratify=y_all, random_state=HOLDOUT_SEED
        )

    # 20% of training rows → calibration set (disjoint from test holdout by seed)
    _, X_cal, _, y_cal = train_test_split(
        X_tr, y_tr, test_size=CAL_SIZE, stratify=y_tr, random_state=CAL_SEED
    )
    logger.info(
        f"Cal set:  {len(y_cal):,} rows  churn={y_cal.mean():.3%}  "
        f"Test set: {len(y_test):,} rows  churn={y_test.mean():.3%}"
    )
    return X_cal, y_cal, X_test, y_test, feature_cols


def _align(
    X: np.ndarray, feature_cols: list[str], saved_cols: list[str]
) -> np.ndarray:
    missing = [c for c in saved_cols if c not in feature_cols]
    if missing:
        raise RuntimeError(
            f"Model feature columns missing from features parquet: {missing}. "
            "Regenerate features and retrain."
        )
    return X[:, [feature_cols.index(c) for c in saved_cols]]


def _fit_isotonic(
    raw_cal: np.ndarray, y_cal: np.ndarray
) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)
    return iso


def _fit_sigmoid(
    raw_cal: np.ndarray, y_cal: np.ndarray
) -> LogisticRegression:
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(raw_cal.reshape(-1, 1), y_cal)
    return lr


def run(
    dataset: str = "moroccan",
    method: str = "both",
    use_mlflow: bool = True,
) -> dict[str, Any]:
    setup_logger()
    cfg = _load_config(CONFIG_PATH)

    # ── Load model ─────────────────────────────────────────────────────────────
    artifact: dict[str, Any] = joblib.load(MODELS_DIR / "best_model.pkl")
    base_model: Any = artifact["model"]
    saved_cols: list[str] = artifact["feature_cols"]
    model_class: str = base_model.__class__.__name__

    # ── Load splits ────────────────────────────────────────────────────────────
    X_cal, y_cal, X_test, y_test, feature_cols = _load_splits(dataset)
    X_cal = _align(X_cal, feature_cols, saved_cols)
    X_test = _align(X_test, feature_cols, saved_cols)

    # ── Baseline (uncalibrated) ────────────────────────────────────────────────
    raw_cal: np.ndarray = base_model.predict_proba(X_cal)[:, 1]
    raw_test: np.ndarray = base_model.predict_proba(X_test)[:, 1]
    m_base = compute_metrics(y_test, raw_test)
    churn_rate = float(y_test.mean())
    baseline_brier = churn_rate * (1 - churn_rate) ** 2 + (1 - churn_rate) * churn_rate ** 2
    logger.info(
        f"[uncalibrated {model_class}]  "
        f"ROC-AUC={m_base['roc_auc']:.4f}  "
        f"PR-AUC={m_base['pr_auc']:.4f}  "
        f"Brier={m_base['brier']:.4f}  "
        f"(baseline={baseline_brier:.4f})"
    )

    # ── Fit and evaluate calibrators ───────────────────────────────────────────
    methods: list[str] = ["isotonic", "sigmoid"] if method == "both" else [method]
    brier_by_method: dict[str, float] = {}

    for m in methods:
        logger.info(f"Fitting {m} calibrator on {len(y_cal):,} samples …")
        if m == "isotonic":
            cal_obj: IsotonicRegression | LogisticRegression = _fit_isotonic(raw_cal, y_cal)
            cal_probs: np.ndarray = cal_obj.predict(raw_test)  # type: ignore[union-attr]
        else:
            cal_obj = _fit_sigmoid(raw_cal, y_cal)
            cal_probs = cal_obj.predict_proba(raw_test.reshape(-1, 1))[:, 1]  # type: ignore[union-attr]
        metrics_m = compute_metrics(y_test, cal_probs)
        brier_by_method[m] = metrics_m["brier"]
        logger.info(
            f"  [{m}]  "
            f"ROC-AUC={metrics_m['roc_auc']:.4f}  "
            f"PR-AUC={metrics_m['pr_auc']:.4f}  "
            f"Brier={metrics_m['brier']:.4f}"
        )

    # ── Pick winner, refit, wrap ───────────────────────────────────────────────
    best_method = min(brier_by_method, key=lambda k: brier_by_method[k])
    logger.info(f"Best: {best_method}  (Brier={brier_by_method[best_method]:.4f})")

    if best_method == "isotonic":
        best_cal_obj: IsotonicRegression | LogisticRegression = _fit_isotonic(raw_cal, y_cal)
    else:
        best_cal_obj = _fit_sigmoid(raw_cal, y_cal)

    wrapped = CalibratedChurnModel(base_model, best_cal_obj, best_method)
    y_prob_best = wrapped.predict_proba(X_test)[:, 1]

    t_youden = optimal_threshold(y_test, y_prob_best, strategy="youden")
    t_f1 = optimal_threshold(y_test, y_prob_best, strategy="f1")
    m_cal = compute_metrics(y_test, y_prob_best, threshold=0.5)
    m_youden = compute_metrics(y_test, y_prob_best, threshold=t_youden)
    m_f1_opt = compute_metrics(y_test, y_prob_best, threshold=t_f1)

    brier_delta = m_base["brier"] - m_cal["brier"]
    logger.info(
        f"Brier: {m_base['brier']:.4f} → {m_cal['brier']:.4f}  "
        f"Δ={brier_delta:+.4f}  baseline={baseline_brier:.4f}"
    )
    logger.info(
        f"Youden threshold={t_youden:.3f}  "
        f"F1={m_youden['f1']:.3f}  P={m_youden['precision']:.3f}  R={m_youden['recall']:.3f}"
    )

    # ── Write metrics ──────────────────────────────────────────────────────────
    out_metrics: dict[str, Any] = {
        "dataset": dataset,
        "calibration_method": best_method,
        "base_model": model_class,
        "cal_set_size": int(len(y_cal)),
        "holdout_size": int(len(y_test)),
        "churn_rate": churn_rate,
        "baseline_brier": baseline_brier,
        "uncalibrated_roc_auc": m_base["roc_auc"],
        "uncalibrated_pr_auc": m_base["pr_auc"],
        "uncalibrated_brier": m_base["brier"],
        "roc_auc": m_cal["roc_auc"],
        "pr_auc": m_cal["pr_auc"],
        "brier": m_cal["brier"],
        "brier_improvement": brier_delta,
        "threshold_youden": t_youden,
        "f1_youden": m_youden["f1"],
        "precision_youden": m_youden["precision"],
        "recall_youden": m_youden["recall"],
        "threshold_f1": t_f1,
        "f1_f1opt": m_f1_opt["f1"],
        "precision_f1opt": m_f1_opt["precision"],
        "recall_f1opt": m_f1_opt["recall"],
    }
    suffix = f"_{dataset}" if dataset != "expresso" else ""
    out_path = MODELS_DIR / f"eval_metrics{suffix}_calibrated.json"
    with open(out_path, "w") as fh:
        json.dump(out_metrics, fh, indent=2)
    logger.info(f"Metrics saved → {out_path}")

    # ── Save calibrated model ──────────────────────────────────────────────────
    cal_artifact: dict[str, Any] = {
        "model": wrapped,
        "feature_cols": saved_cols,
        "calibration": {"method": best_method, "dataset": dataset},
    }
    model_out = MODELS_DIR / "best_model_calibrated.pkl"
    joblib.dump(cal_artifact, model_out)
    logger.info(f"Calibrated model saved → {model_out}")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    if use_mlflow:
        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = str(mlflow_cfg.get("tracking_uri", "mlruns"))
        resolved_uri = (
            tracking_uri
            if "://" in tracking_uri or os.path.isabs(tracking_uri)
            else str(ROOT / tracking_uri)
        )
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "moroccan_prepaid_churn"))

        with mlflow.start_run(run_name=f"calibrate__{dataset}__{best_method}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("calibration_method", best_method)
            mlflow.log_param("base_model", model_class)
            mlflow.log_param("cal_set_size", int(len(y_cal)))
            mlflow.log_metric("uncalibrated_brier", m_base["brier"])
            mlflow.log_metric("calibrated_brier", m_cal["brier"])
            mlflow.log_metric("brier_improvement", brier_delta)
            mlflow.log_metric("roc_auc", m_cal["roc_auc"])
            mlflow.log_metric("pr_auc", m_cal["pr_auc"])
            mlflow.log_artifact(str(out_path), "calibration")
            mlflow.log_artifact(str(model_out), "calibration")

            active = mlflow.active_run()
            if active is not None:
                logger.info(f"MLflow run: {active.info.run_id}")

    return out_metrics  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc probability calibration")
    p.add_argument(
        "--dataset", choices=["expresso", "moroccan"], default="moroccan"
    )
    p.add_argument(
        "--method",
        choices=["isotonic", "sigmoid", "both"],
        default="both",
        help="Calibration method (default: both, picks winner by Brier)",
    )
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(dataset=args.dataset, method=args.method, use_mlflow=args.use_mlflow)
