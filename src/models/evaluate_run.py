"""Holdout evaluation of the trained model (DVC `evaluate_*` stages).

Scores data/features/{dataset}/holdout_features.parquet — the labeled split that
run_pipeline.py carves off *before* the feature pipeline is fit, and that
training, calibration, and threshold selection never see.

Nothing is tuned here: decision thresholds are the ones train.py chose on
out-of-fold predictions and stored in the model artifact. The holdout only
measures them.

Writes models/{dataset}/eval_metrics.json.

Usage:
    python -m src.models.evaluate_run --dataset expresso
    python -m src.models.evaluate_run --dataset moroccan --no-mlflow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.models.evaluate import compute_metrics
from src.utils.config import BASE_CONFIG, DATASETS, dataset_paths, load_config, setup_mlflow
from src.utils.logging import setup_logger


def _load_holdout(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.features.run_pipeline` first."
        )
    df = pd.read_parquet(path)
    if "CHURN" not in df.columns:
        raise RuntimeError(f"{path} has no CHURN column; regenerate features.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Model feature columns missing from {path.name}: {missing}. "
            "Regenerate features and retrain the model."
        )
    return df[feature_cols].to_numpy(dtype=np.float32), df["CHURN"].to_numpy()


def run(
    dataset: str = "expresso",
    use_mlflow: bool = True,
    config_path: Path = BASE_CONFIG,
) -> dict[str, Any]:
    setup_logger()
    cfg = load_config(dataset, config_path)
    paths = dataset_paths(dataset)

    artifact: dict[str, Any] = joblib.load(paths.model)
    model: Any = artifact["model"]
    feature_cols: list[str] = artifact["feature_cols"]
    thresholds: dict[str, float] = artifact.get("thresholds", {})

    X, y = _load_holdout(paths.holdout_features, feature_cols)
    logger.info(f"[{dataset}] Holdout: {len(y):,} rows  churn={y.mean():.3%}")

    y_prob: np.ndarray = model.predict_proba(X)[:, 1]
    base = compute_metrics(y, y_prob)

    churn_rate = float(y.mean())
    metrics: dict[str, Any] = {
        "dataset": dataset,
        "model_name": artifact.get("model_name", "unknown"),
        "calibration_method": artifact.get("calibration_method"),
        "holdout_size": int(len(y)),
        "churn_rate": churn_rate,
        # Ranking metrics (threshold-independent)
        "roc_auc": base["roc_auc"],
        "pr_auc": base["pr_auc"],
        # Probability quality
        "brier": base["brier"],
        "baseline_brier": churn_rate * (1 - churn_rate),
    }
    if hasattr(model, "predict_raw"):
        metrics["uncalibrated_brier"] = compute_metrics(y, model.predict_raw(X))["brier"]

    # Threshold metrics at the thresholds frozen during training
    for name, t in thresholds.items():
        m = compute_metrics(y, y_prob, threshold=t)
        metrics[f"threshold_{name}"] = t
        metrics[f"f1_{name}"] = m["f1"]
        metrics[f"precision_{name}"] = m["precision"]
        metrics[f"recall_{name}"] = m["recall"]

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.eval_metrics, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(
        f"Holdout ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier']:.4f} (baseline {metrics['baseline_brier']:.4f})"
    )
    for name in thresholds:
        logger.info(
            f"  @{name} t={metrics[f'threshold_{name}']:.3f}  "
            f"F1={metrics[f'f1_{name}']:.3f}  P={metrics[f'precision_{name}']:.3f}  "
            f"R={metrics[f'recall_{name}']:.3f}"
        )
    logger.info(f"Metrics saved → {paths.eval_metrics}")

    if use_mlflow:
        import mlflow

        setup_mlflow(cfg)
        with mlflow.start_run(run_name=f"evaluate__{dataset}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("holdout_size", len(y))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("model_name", metrics["model_name"])
            for key, val in metrics.items():
                if isinstance(val, float):
                    mlflow.log_metric(key, val)
            mlflow.log_artifact(str(paths.eval_metrics), "evaluation")

            active = mlflow.active_run()
            if active is not None:
                logger.info(f"MLflow run: {active.info.run_id}")

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the trained model on the labeled holdout")
    p.add_argument("--dataset", choices=DATASETS, default="expresso")
    p.add_argument("--config", type=Path, default=BASE_CONFIG)
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(dataset=args.dataset, use_mlflow=args.use_mlflow, config_path=args.config)
