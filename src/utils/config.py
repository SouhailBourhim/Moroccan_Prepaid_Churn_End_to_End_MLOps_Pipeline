"""Shared configuration, artifact-path, and MLflow helpers.

Every entry point (feature pipeline, training, tuning, evaluation, API) goes
through this module so that config loading, per-dataset artifact locations,
and MLflow tracking-URI resolution are defined exactly once.

Config layering
---------------
configs/base.yaml holds defaults shared by every dataset. An optional overlay,
configs/{dataset}.yaml, is deep-merged on top — this is where tuned
hyperparameters live, so tuning one dataset never changes another.

Artifact layout
---------------
    data/features/{dataset}/   train/holdout/score parquet, feature_pipeline.pkl, …
    models/{dataset}/          model.pkl, training_manifest.json, eval_metrics.json, …
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"
DATASETS: tuple[str, ...] = ("expresso", "moroccan")


def check_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return dataset


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def overlay_path(dataset: str, base_path: Path = BASE_CONFIG) -> Path:
    """Path of the per-dataset overlay that sits next to the base config."""
    return base_path.parent / f"{check_dataset(dataset)}.yaml"


def load_config(dataset: str | None = None, path: Path = BASE_CONFIG) -> dict[str, Any]:
    """Load the base config and, when a dataset is given, deep-merge its overlay."""
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    if dataset is not None:
        overlay = overlay_path(dataset, path)
        if overlay.exists():
            with open(overlay) as fh:
                cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


@dataclass(frozen=True)
class DatasetPaths:
    """Every on-disk artifact location for one dataset."""

    dataset: str
    features_dir: Path
    models_dir: Path

    @property
    def train_features(self) -> Path:
        return self.features_dir / "train_features.parquet"

    @property
    def holdout_features(self) -> Path:
        return self.features_dir / "holdout_features.parquet"

    @property
    def score_features(self) -> Path:
        """Unlabeled rows to score (Expresso Test.csv); absent for Moroccan."""
        return self.features_dir / "score_features.parquet"

    @property
    def feature_pipeline(self) -> Path:
        return self.features_dir / "feature_pipeline.pkl"

    @property
    def feature_stats(self) -> Path:
        return self.features_dir / "feature_stats.csv"

    @property
    def feature_manifest(self) -> Path:
        return self.features_dir / "feature_manifest.json"

    @property
    def model(self) -> Path:
        return self.models_dir / "model.pkl"

    @property
    def training_manifest(self) -> Path:
        return self.models_dir / "training_manifest.json"

    @property
    def eval_metrics(self) -> Path:
        return self.models_dir / "eval_metrics.json"

    def tuning_results(self, model_name: str) -> Path:
        return self.models_dir / f"tuning_results_{model_name}.json"

    def ensure_dirs(self) -> None:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def dataset_paths(dataset: str, root: Path = ROOT) -> DatasetPaths:
    check_dataset(dataset)
    return DatasetPaths(
        dataset=dataset,
        features_dir=root / "data" / "features" / dataset,
        models_dir=root / "models" / dataset,
    )


def resolve_tracking_uri(cfg: dict[str, Any]) -> str:
    """Return the MLflow tracking URI; relative file paths are anchored at ROOT."""
    uri = str(cfg.get("mlflow", {}).get("tracking_uri", "mlruns"))
    if "://" in uri or os.path.isabs(uri):
        return uri
    return str(ROOT / uri)


def setup_mlflow(cfg: dict[str, Any]) -> None:
    """Point MLflow at the configured tracking URI and experiment."""
    import mlflow  # local import: the API image does not ship mlflow

    mlflow.set_tracking_uri(resolve_tracking_uri(cfg))
    mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "moroccan_prepaid_churn"))
