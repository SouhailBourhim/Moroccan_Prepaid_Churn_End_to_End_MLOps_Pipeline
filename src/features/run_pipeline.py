"""Run the feature engineering pipeline end-to-end for one dataset.

Load labeled data → validate → split off a stratified labeled holdout →
fit FeaturePipeline on the training split only → transform holdout (and, for
Expresso, the unlabeled Test.csv) → save to data/features/{dataset}/ → MLflow.

The holdout is carved off *before* anything is fit, and is never used for
training, early stopping, calibration, or threshold selection. It exists only
for src.models.evaluate_run.

Datasets (--dataset):
  expresso  (default) — Expresso Telecom Senegal. Train.csv is split into
                        train/holdout; Test.csv (unlabeled) becomes score_features.
  moroccan            — Synthetic Moroccan prepaid dataset
                        (generate first: python generate_moroccan_dataset.py)

Outputs (data/features/{dataset}/):
    train_features.parquet    model features + CHURN (target encodings cross-fitted)
    holdout_features.parquet  user_id + model features + CHURN
    score_features.parquet    user_id + model features (Expresso only)
    feature_pipeline.pkl, feature_stats.csv, feature_manifest.json

Usage
-----
    python -m src.features.run_pipeline --dataset expresso
    python -m src.features.run_pipeline --dataset moroccan --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.ingestion import load_moroccan, load_test, load_train
from src.data.validation import ValidationReport, validate_raw
from src.features.build_features import FeaturePipeline, get_model_features
from src.utils.config import (
    BASE_CONFIG,
    DATASETS,
    ROOT,
    dataset_paths,
    load_config,
    setup_mlflow,
)
from src.utils.logging import setup_logger

# Metadata columns that exist in a raw file but are not pipeline inputs.
_METADATA_COLS = ["OPERATOR"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute per-feature diagnostics on the engineered matrix."""
    X = df[feature_cols]
    stats = pd.DataFrame({
        "dtype":       X.dtypes.astype(str),
        "null_rate":   X.isna().mean().round(6),
        "pct_zero":    (X == 0).mean().round(6),
        "mean":        X.mean().round(6),
        "std":         X.std().round(6),
        "min":         X.min(),
        "max":         X.max(),
    })
    if "CHURN" in df.columns:
        stats["spearman_with_churn"] = pd.Series(
            {
                col: df[col].astype(float).corr(df["CHURN"].astype(float), method="spearman")
                for col in feature_cols
            }
        ).round(4)
        stats = stats.sort_values("spearman_with_churn", key=abs, ascending=False)
    return stats


def _check(report: ValidationReport, split: str) -> None:
    if not report.passed:
        raise RuntimeError(f"Validation failed on {split}: {report.errors}")
    for w in report.warnings:
        logger.warning(f"[{split}] {w}")


# ── Main ───────────────────────────────────────────────────────────────────────


def run(
    use_mlflow: bool = True,
    config_path: Path = BASE_CONFIG,
    dataset: str = "expresso",
) -> None:
    setup_logger()
    cfg = load_config(dataset, config_path)
    paths = dataset_paths(dataset)
    paths.ensure_dirs()
    training_cfg = cfg.get("training", {})
    random_state = int(training_cfg.get("random_state", 42))

    # ── 1. Load & validate ────────────────────────────────────────────────────
    logger.info(f"Loading raw data  (dataset={dataset}) …")
    t0 = time.perf_counter()

    score_raw: pd.DataFrame | None = None
    if dataset == "moroccan":
        moroccan_file = cfg.get("data", {}).get(
            "moroccan_raw_file", "data/raw/moroccan_telecom_churn.csv"
        )
        labeled = load_moroccan(path=ROOT / moroccan_file)
    else:
        labeled = load_train()
        score_raw = load_test()

    labeled = labeled.drop(columns=_METADATA_COLS, errors="ignore")
    _check(validate_raw(labeled, is_train=True), "labeled")
    if score_raw is not None:
        _check(validate_raw(score_raw, is_train=False), "score")

    n_dupes = int(labeled.drop(columns=["user_id"]).duplicated().sum())
    if n_dupes:
        logger.warning(
            f"{n_dupes:,} labeled rows are exact duplicates of another row (ignoring user_id). "
            "Duplicates can land on both sides of the holdout split; if they are "
            "generator artifacts rather than distinct subscribers, deduplicate upstream."
        )

    # ── 2. Split off the labeled holdout before fitting anything ──────────────
    train_raw, holdout_raw = train_test_split(
        labeled,
        test_size=float(training_cfg.get("test_size", 0.2)),
        stratify=labeled["CHURN"],
        random_state=random_state,
    )
    train_raw = train_raw.reset_index(drop=True)
    holdout_raw = holdout_raw.reset_index(drop=True)
    logger.info(
        f"Loaded  train={len(train_raw):,}  holdout={len(holdout_raw):,}"
        + (f"  score={len(score_raw):,}" if score_raw is not None else "")
        + f"  ({time.perf_counter() - t0:.1f}s)"
    )

    # ── 3. Fit pipeline on the training split only ────────────────────────────
    logger.info("Fitting FeaturePipeline on train…")
    t1 = time.perf_counter()

    X_train = train_raw.drop(columns=["CHURN"])
    y_train = train_raw["CHURN"]

    features_cfg = cfg.get("features", {})
    pipeline = FeaturePipeline(
        top_pack_min_freq=int(features_cfg.get("top_pack_min_freq", 200)),
        target_enc_smoothing=float(features_cfg.get("target_enc_smoothing", 20.0)),
        regularity_inactive_threshold=float(
            features_cfg.get("regularity_inactive_threshold", 5)
        ),
        target_enc_cv=int(features_cfg.get("target_enc_cv", 5)),
        random_state=random_state,
    )
    # fit_transform returns cross-fitted target encodings for the training rows.
    X_train_fe = pipeline.fit_transform(X_train, y_train)
    fit_seconds = time.perf_counter() - t1
    logger.info(f"Pipeline fit complete ({fit_seconds:.1f}s)")

    feature_cols = get_model_features(X_train_fe)
    logger.info(f"Final feature count: {len(feature_cols)}")

    # ── 4. Transform holdout / score sets with train-fit parameters ───────────
    X_holdout_fe = pipeline.transform(holdout_raw.drop(columns=["CHURN"]))
    X_score_fe = pipeline.transform(score_raw) if score_raw is not None else None

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    train_fe = X_train_fe.copy()
    train_fe["CHURN"] = y_train.values
    stats = _feature_stats(train_fe, feature_cols)
    logger.info("\n" + stats[["null_rate", "pct_zero", "spearman_with_churn"]].head(15).to_string())

    null_features = stats[stats["null_rate"] > 0]["null_rate"]
    if len(null_features):
        logger.warning(
            f"Features with remaining nulls after pipeline:\n{null_features.to_string()}"
        )

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    logger.info(f"Saving features to {paths.features_dir} …")

    train_out = X_train_fe[feature_cols].copy()
    train_out["CHURN"] = y_train.values
    train_out.to_parquet(paths.train_features, index=False)

    holdout_out = X_holdout_fe[feature_cols].copy()
    holdout_out.insert(0, "user_id", holdout_raw["user_id"].values)
    holdout_out["CHURN"] = holdout_raw["CHURN"].values
    holdout_out.to_parquet(paths.holdout_features, index=False)

    if X_score_fe is not None and score_raw is not None:
        score_out = X_score_fe[feature_cols].copy()
        score_out.insert(0, "user_id", score_raw["user_id"].values)
        score_out.to_parquet(paths.score_features, index=False)
    elif paths.score_features.exists():
        paths.score_features.unlink()

    joblib.dump(pipeline, paths.feature_pipeline)
    stats.to_csv(paths.feature_stats)

    feature_manifest = {
        "dataset": dataset,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_holdout": float(holdout_raw["CHURN"].mean()),
        "n_train": len(train_out),
        "n_holdout": len(holdout_out),
        "n_score": len(X_score_fe) if X_score_fe is not None else 0,
        "split_random_state": random_state,
    }
    with open(paths.feature_manifest, "w") as f:
        json.dump(feature_manifest, f, indent=2)

    logger.info(
        f"Saved: train ({len(train_out):,} rows × {len(feature_cols)} cols), "
        f"holdout ({len(holdout_out):,} rows)"
    )

    # ── 7. MLflow logging ─────────────────────────────────────────────────────
    if use_mlflow:
        import mlflow

        setup_mlflow(cfg)
        with mlflow.start_run(run_name=f"feature_engineering__{dataset}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("top_pack_min_freq", features_cfg.get("top_pack_min_freq", 200))
            mlflow.log_param("target_enc_smoothing", features_cfg.get("target_enc_smoothing", 20.0))
            mlflow.log_param("target_enc_cv", features_cfg.get("target_enc_cv", 5))

            mlflow.log_metric("n_train_rows", len(train_out))
            mlflow.log_metric("n_holdout_rows", len(holdout_out))
            mlflow.log_metric("churn_rate_train", float(y_train.mean()))
            mlflow.log_metric("pipeline_fit_seconds", round(fit_seconds, 2))
            mlflow.log_metric("pct_null_features", float((stats["null_rate"] > 0).mean()))

            if "spearman_with_churn" in stats.columns:
                for feat, rho in stats["spearman_with_churn"].abs().head(10).items():
                    mlflow.log_metric(f"rho_{feat}", float(rho))

            mlflow.log_artifact(str(paths.feature_stats), "feature_engineering")
            mlflow.log_artifact(str(paths.feature_manifest), "feature_engineering")
            mlflow.log_artifact(str(paths.feature_pipeline), "feature_engineering")

            active_run = mlflow.active_run()
            if active_run is not None:
                logger.info(f"MLflow run logged: {active_run.info.run_id}")

    total = time.perf_counter() - t0
    logger.info(f"Feature engineering complete in {total:.1f}s")


# ── Entry point ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run churn feature engineering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=BASE_CONFIG)
    p.add_argument("--dataset", choices=DATASETS, default="expresso", help="Source dataset")
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(use_mlflow=args.use_mlflow, config_path=args.config, dataset=args.dataset)
