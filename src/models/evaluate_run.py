"""Standalone holdout evaluation script for the DVC pipeline.

Loads the trained model and a stratified holdout split of the training
features, computes key metrics, and writes them to models/eval_metrics.json
so DVC can track and diff them across pipeline runs.

Usage:
    python -m src.models.evaluate_run
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.models.evaluate import compute_metrics, optimal_threshold
from src.utils.logging import setup_logger

ROOT = Path(__file__).parents[2]
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"

HOLDOUT_SIZE = 0.20
RANDOM_STATE = 42


def run() -> dict[str, float]:
    setup_logger()

    # ── Load features ─────────────────────────────────────────────────────────
    df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    y = df["CHURN"].to_numpy()
    feature_cols = [c for c in df.columns if c != "CHURN"]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    artifact: dict[str, Any] = joblib.load(MODELS_DIR / "best_model.pkl")
    model: Any = artifact["model"]
    saved_cols: list[str] = artifact["feature_cols"]

    # Align columns in case pipeline produced a different order
    missing_cols = [c for c in saved_cols if c not in feature_cols]
    if missing_cols:
        raise RuntimeError(
            "Saved model feature columns are missing from train_features.parquet: "
            f"{missing_cols}. Regenerate features and retrain the model."
        )
    col_idx = [feature_cols.index(c) for c in saved_cols]
    X = X[:, col_idx]

    # ── Holdout split (deterministic) ─────────────────────────────────────────
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Holdout: {len(y_val):,} rows  churn={y_val.mean():.3%}")

    # ── Score ─────────────────────────────────────────────────────────────────
    y_prob: np.ndarray = model.predict_proba(X_val)[:, 1]

    t_youden = optimal_threshold(y_val, y_prob, strategy="youden")
    t_f1 = optimal_threshold(y_val, y_prob, strategy="f1")

    m_default = compute_metrics(y_val, y_prob, threshold=0.5)
    m_youden = compute_metrics(y_val, y_prob, threshold=t_youden)
    m_f1 = compute_metrics(y_val, y_prob, threshold=t_f1)

    # ── Flat metrics dict (DVC-friendly) ──────────────────────────────────────
    metrics: dict[str, float] = {
        "holdout_size": float(len(y_val)),
        "churn_rate": float(y_val.mean()),
        # Ranking metrics (threshold-independent)
        "roc_auc": m_default["roc_auc"],
        "pr_auc": m_default["pr_auc"],
        "brier": m_default["brier"],
        # Threshold-based at default 0.5
        "f1_default": m_default["f1"],
        "precision_default": m_default["precision"],
        "recall_default": m_default["recall"],
        # Threshold-based at Youden-J optimum
        "threshold_youden": t_youden,
        "f1_youden": m_youden["f1"],
        "precision_youden": m_youden["precision"],
        "recall_youden": m_youden["recall"],
        # Threshold-based at F1 optimum
        "threshold_f1": t_f1,
        "f1_f1opt": m_f1["f1"],
        "precision_f1opt": m_f1["precision"],
        "recall_f1opt": m_f1["recall"],
    }

    out = MODELS_DIR / "eval_metrics.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(
        f"Holdout ROC-AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier']:.4f}"
    )
    logger.info(
        f"Youden threshold={t_youden:.3f}  "
        f"F1={metrics['f1_youden']:.3f}  "
        f"P={metrics['precision_youden']:.3f}  "
        f"R={metrics['recall_youden']:.3f}"
    )
    logger.info(f"Metrics saved → {out}")
    return metrics


if __name__ == "__main__":
    run()
