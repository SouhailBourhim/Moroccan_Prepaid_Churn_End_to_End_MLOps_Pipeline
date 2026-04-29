"""Inference script for the Expresso churn model.

Loads the trained model artifact from models/ and the feature pipeline from
data/features/, applies both to raw subscriber data, and outputs churn
probabilities (and optionally hard predictions) to predictions/.

Prerequisites:
    python -m src.features.run_pipeline   # saves feature_pipeline.pkl
    python -m src.models.train            # saves best_model.pkl

Usage:
    python -m src.models.predict
    python -m src.models.predict --input path/to/custom.csv
    python -m src.models.predict --threshold 0.4
    python -m src.models.predict --output path/to/out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.data.ingestion import load_test
from src.features.build_features import FeaturePipeline
from src.utils.logging import setup_logger

ROOT = Path(__file__).parents[2]
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"


# ── Artifact loading ───────────────────────────────────────────────────────────


def load_artifacts(
    model_dir: Path = MODELS_DIR,
    features_dir: Path = FEATURES_DIR,
) -> tuple[FeaturePipeline, Any, list[str]]:
    """Load the saved feature pipeline, best model, and feature column list."""
    pipeline_path = features_dir / "feature_pipeline.pkl"
    model_path = model_dir / "best_model.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"{pipeline_path} not found — run `python -m src.features.run_pipeline`"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python -m src.models.train`"
        )

    pipeline: FeaturePipeline = joblib.load(pipeline_path)
    artifact: dict[str, Any] = joblib.load(model_path)
    model: Any = artifact["model"]
    feature_cols: list[str] = artifact["feature_cols"]
    return pipeline, model, feature_cols


# ── Inference ─────────────────────────────────────────────────────────────────


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


# ── Entry point ───────────────────────────────────────────────────────────────


def run(
    input_path: Path | None = None,
    threshold: float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Load model, run inference, save results. Returns the predictions DataFrame."""
    setup_logger()
    pipeline, model, feature_cols = load_artifacts()

    df = load_test(path=input_path) if input_path else load_test()
    logger.info(f"Loaded {len(df):,} rows for inference")

    probs = predict_proba(df, pipeline, model, feature_cols)
    out = pd.DataFrame({"user_id": df["user_id"], "churn_prob": probs})

    if threshold is not None:
        out["churn_pred"] = (probs >= threshold).astype("int8")
        positive_rate = float(out["churn_pred"].mean())
        logger.info(
            f"Threshold={threshold:.3f}: {positive_rate:.1%} of subscribers predicted to churn"
        )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = output_path or PREDICTIONS_DIR / "test_predictions.csv"
    out.to_csv(dest, index=False)
    logger.info(f"Predictions saved → {dest}  ({len(out):,} rows)")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Expresso churn model inference")
    p.add_argument("--input", type=Path, default=None, help="Path to raw CSV input")
    p.add_argument("--threshold", type=float, default=None, help="Decision threshold (optional)")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(input_path=args.input, threshold=args.threshold, output_path=args.output)
