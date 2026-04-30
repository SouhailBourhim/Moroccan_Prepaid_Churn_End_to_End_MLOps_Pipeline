"""Model artifact loading for the churn prediction API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import joblib
from loguru import logger

from src.features.build_features import FeaturePipeline

ROOT = Path(__file__).parents[2]
MODELS_DIR = ROOT / "models"
FEATURES_DIR = ROOT / "data" / "features"


class ModelArtifacts(NamedTuple):
    pipeline: FeaturePipeline
    model: Any
    feature_cols: list[str]
    manifest: dict[str, Any]


def load_artifacts(
    model_dir: Path = MODELS_DIR,
    features_dir: Path = FEATURES_DIR,
) -> ModelArtifacts:
    """Load feature pipeline, trained model, and training manifest from disk."""
    pipeline_path = features_dir / "feature_pipeline.pkl"
    model_path = model_dir / "best_model.pkl"
    manifest_path = model_dir / "training_manifest.json"

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

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    logger.info(
        f"Artifacts loaded: model={manifest.get('best_model', 'unknown')}  "
        f"features={len(feature_cols)}"
    )
    return ModelArtifacts(
        pipeline=pipeline,
        model=model,
        feature_cols=feature_cols,
        manifest=manifest,
    )
