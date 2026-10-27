"""Batch inference with the trained churn model.

Loads models/{dataset}/model.pkl (the calibrated model the API serves) and the
feature pipeline from data/features/{dataset}/, applies both to raw subscriber
data, and writes churn probabilities (and hard predictions) to predictions/.

Prerequisites:
    python -m src.features.run_pipeline --dataset expresso
    python -m src.models.train --dataset expresso

Usage:
    python -m src.models.predict                          # Expresso Test.csv
    python -m src.models.predict --input path/to/raw.csv
    python -m src.models.predict --dataset moroccan --input path/to/raw.csv
    python -m src.models.predict --threshold 0.4          # override the trained default
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.ingestion import load_test
from src.features.build_features import FeaturePipeline
from src.models.artifacts import load_artifacts
from src.utils.config import DATASETS, ROOT
from src.utils.logging import setup_logger

PREDICTIONS_DIR = ROOT / "predictions"


def predict_proba(
    df: pd.DataFrame,
    pipeline: FeaturePipeline,
    model: Any,
    feature_cols: list[str],
) -> pd.Series:
    """Return churn probability for each row of raw (untransformed) input data."""
    X_fe = pipeline.transform(df)
    X = X_fe[feature_cols].to_numpy(dtype=np.float32)
    probs: np.ndarray = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index, name="churn_prob")


def run(
    dataset: str = "expresso",
    input_path: Path | None = None,
    threshold: float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Load model, run inference, save results. Returns the predictions DataFrame."""
    setup_logger()
    art = load_artifacts(dataset)

    if input_path is None and dataset != "expresso":
        raise ValueError(f"--input is required for dataset={dataset!r}")
    df = load_test(path=input_path) if input_path else load_test()
    df = df.drop(columns=["CHURN", "OPERATOR"], errors="ignore")
    logger.info(f"Loaded {len(df):,} rows for inference")

    probs = predict_proba(df, art.pipeline, art.model, art.feature_cols)
    t = art.default_threshold if threshold is None else threshold
    out = pd.DataFrame({
        "user_id": df["user_id"],
        "churn_prob": probs,
        "churn_pred": (probs >= t).astype("int8"),
    })
    logger.info(f"Threshold={t:.3f}: {out['churn_pred'].mean():.1%} predicted to churn")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = output_path or PREDICTIONS_DIR / f"{dataset}_predictions.csv"
    out.to_csv(dest, index=False)
    logger.info(f"Predictions saved → {dest}  ({len(out):,} rows)")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run churn model batch inference")
    p.add_argument("--dataset", choices=DATASETS, default="expresso")
    p.add_argument("--input", type=Path, default=None, help="Path to raw CSV input")
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default: the F1-optimal threshold chosen in training)",
    )
    p.add_argument("--output", type=Path, default=None, help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        dataset=args.dataset,
        input_path=args.input,
        threshold=args.threshold,
        output_path=args.output,
    )
