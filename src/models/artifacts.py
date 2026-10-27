"""Loading the served model: feature pipeline + calibrated model + metadata.

Used by both the FastAPI service and the batch predict CLI, so they always
serve the same artifact: models/{dataset}/model.pkl, which train.py writes as
a CalibratedChurnModel together with the decision thresholds it chose.

Only scikit-learn and the library of the winning model are needed to unpickle
it; this module itself imports no boosting library.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import joblib
from loguru import logger

from src.features.build_features import FeaturePipeline
from src.utils.config import ROOT, dataset_paths


class ModelArtifacts(NamedTuple):
    pipeline: FeaturePipeline
    model: Any
    feature_cols: list[str]
    manifest: dict[str, Any]
    default_threshold: float = 0.5
    dataset: str = "expresso"
    calibration_method: str | None = None


def load_artifacts(dataset: str = "expresso", root: Path = ROOT) -> ModelArtifacts:
    """Load feature pipeline, trained model, and training manifest for one dataset."""
    paths = dataset_paths(dataset, root)
    if not paths.feature_pipeline.exists():
        raise FileNotFoundError(
            f"{paths.feature_pipeline} not found — run "
            f"`python -m src.features.run_pipeline --dataset {dataset}`"
        )
    if not paths.model.exists():
        raise FileNotFoundError(
            f"{paths.model} not found — run `python -m src.models.train --dataset {dataset}`"
        )

    pipeline: FeaturePipeline = joblib.load(paths.feature_pipeline)
    artifact: dict[str, Any] = joblib.load(paths.model)

    manifest: dict[str, Any] = {}
    if paths.training_manifest.exists():
        with open(paths.training_manifest) as fh:
            manifest = json.load(fh)

    artifacts = ModelArtifacts(
        pipeline=pipeline,
        model=artifact["model"],
        feature_cols=artifact["feature_cols"],
        manifest=manifest,
        default_threshold=float(artifact.get("default_threshold", 0.5)),
        dataset=dataset,
        calibration_method=artifact.get("calibration_method"),
    )
    logger.info(
        f"Artifacts loaded: dataset={dataset}  model={artifact.get('model_name', 'unknown')}  "
        f"calibration={artifacts.calibration_method}  features={len(artifacts.feature_cols)}  "
        f"default_threshold={artifacts.default_threshold:.3f}"
    )
    return artifacts
