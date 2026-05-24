"""Run the feature engineering pipeline end-to-end.

Loads raw data → validates → fits FeaturePipeline on train →
transforms train + test → saves to data/features/ → logs a summary run to MLflow.

Supports two dataset sources via --dataset:
  expresso  (default) — Expresso Telecom Senegal dataset
  moroccan            — Synthetic Moroccan prepaid dataset
                        (generate first: python generate_moroccan_dataset.py)

Usage
-----
    python -m src.features.run_pipeline
    python -m src.features.run_pipeline --dataset moroccan
    python -m src.features.run_pipeline --no-mlflow
    python -m src.features.run_pipeline --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from loguru import logger

from src.data.ingestion import load_moroccan, load_test, load_train
from src.data.validation import validate_raw
from src.features.build_features import FeaturePipeline, get_model_features
from src.utils.logging import setup_logger

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parents[2]
FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "configs" / "base.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


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


# ── Main ───────────────────────────────────────────────────────────────────────


def run(
    use_mlflow: bool = True,
    config_path: Path = CONFIG_PATH,
    dataset: str = "expresso",
) -> None:
    setup_logger()
    cfg = _load_config(config_path)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if dataset not in ("expresso", "moroccan"):
        raise ValueError(f"--dataset must be 'expresso' or 'moroccan', got {dataset!r}")

    # ── 1. Load & validate ────────────────────────────────────────────────────
    logger.info(f"Loading raw data  (dataset={dataset}) …")
    t0 = time.perf_counter()

    if dataset == "moroccan":
        data_cfg = cfg.get("data", {})
        moroccan_path = ROOT / data_cfg.get(
            "moroccan_raw_file", "data/raw/moroccan_telecom_churn.csv"
        )
        train_raw = load_moroccan(path=moroccan_path)
        # Moroccan CSV has no separate test split — use an 80/20 stratified split
        from sklearn.model_selection import train_test_split
        train_raw, test_raw = train_test_split(
            train_raw,
            test_size=float(cfg.get("training", {}).get("test_size", 0.2)),
            stratify=train_raw["CHURN"],
            random_state=int(cfg.get("training", {}).get("random_state", 42)),
        )
        train_raw = train_raw.reset_index(drop=True)
        test_raw  = test_raw.reset_index(drop=True)
        # OPERATOR column is metadata — drop before the feature pipeline so it
        # doesn't land in the model feature matrix as an unencoded categorical.
        for df in (train_raw, test_raw):
            df.drop(columns=["OPERATOR"], errors="ignore", inplace=True)
        logger.info(
            f"Moroccan split  train={len(train_raw):,}  test={len(test_raw):,}  "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        test_has_churn = True
    else:
        train_raw = load_train()
        test_raw  = load_test()
        test_has_churn = False

        for split, df, is_train in [("train", train_raw, True), ("test", test_raw, False)]:
            report = validate_raw(df, is_train=is_train)
            if not report.passed:
                raise RuntimeError(f"Validation failed on {split}: {report.errors}")
            for w in report.warnings:
                logger.warning(f"[{split}] {w}")

        logger.info(
            f"Loaded  train={len(train_raw):,} rows  test={len(test_raw):,} rows  "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    # ── 2. Fit pipeline on train ──────────────────────────────────────────────
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
    )
    X_train_fe = pipeline.fit_transform(X_train, y_train)
    fit_seconds = time.perf_counter() - t1
    logger.info(f"Pipeline fit complete ({fit_seconds:.1f}s)")

    # ── 3. Transform test (with train-fit parameters) ─────────────────────────
    logger.info("Transforming test set…")
    # For Moroccan split the test set carries CHURN — strip it before transform.
    test_for_transform = test_raw.drop(columns=["CHURN"], errors="ignore")
    X_test_fe = pipeline.transform(test_for_transform)

    feature_cols = get_model_features(X_train_fe)
    logger.info(f"Final feature count: {len(feature_cols)}")

    # ── 4. Diagnostics ────────────────────────────────────────────────────────
    train_fe = X_train_fe.copy()
    train_fe["CHURN"] = y_train.values

    stats = _feature_stats(train_fe, feature_cols)

    logger.info("\n" + stats[["null_rate", "pct_zero", "spearman_with_churn"]].head(15).to_string())

    null_features = stats[stats["null_rate"] > 0]["null_rate"]
    if len(null_features):
        logger.warning(
            f"Features with remaining nulls after pipeline:\n{null_features.to_string()}"
        )

    # ── 5. Save artifacts ─────────────────────────────────────────────────────
    logger.info("Saving features to parquet…")

    train_out = X_train_fe[feature_cols].copy()
    train_out["CHURN"] = y_train.values
    train_out.to_parquet(FEATURES_DIR / "train_features.parquet", index=False)

    test_out = X_test_fe[feature_cols].copy()
    if "user_id" in test_raw.columns:
        test_out.insert(0, "user_id", test_raw["user_id"].values)
    if test_has_churn and "CHURN" in test_raw.columns:
        test_out["CHURN"] = test_raw["CHURN"].values
    test_out.to_parquet(FEATURES_DIR / "test_features.parquet", index=False)

    joblib.dump(pipeline, FEATURES_DIR / "feature_pipeline.pkl")
    stats.to_csv(FEATURES_DIR / "feature_stats.csv")

    feature_manifest = {
        "dataset": dataset,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "churn_rate_train": float(y_train.mean()),
        "n_train": len(train_out),
        "n_test": len(test_out),
    }
    with open(FEATURES_DIR / "feature_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=2)

    logger.info(
        f"Saved: train_features.parquet ({len(train_out):,} rows × {len(feature_cols)} cols), "
        f"test_features.parquet ({len(test_out):,} rows)"
    )

    # ── 6. MLflow logging ─────────────────────────────────────────────────────
    if use_mlflow:
        mlflow_cfg = cfg.get("mlflow", {})
        mlflow.set_tracking_uri(str(ROOT / mlflow_cfg.get("tracking_uri", "mlruns")))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "moroccan_prepaid_churn"))

        with mlflow.start_run(run_name=f"feature_engineering__{dataset}"):
            # Parameters
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("n_features", len(feature_cols))
            features_cfg = cfg.get("features", {})
            mlflow.log_param("top_pack_min_freq", features_cfg.get("top_pack_min_freq", 200))
            mlflow.log_param("target_enc_smoothing", features_cfg.get("target_enc_smoothing", 20.0))

            # Metrics
            mlflow.log_metric("n_train_rows", len(train_out))
            mlflow.log_metric("n_test_rows", len(test_out))
            mlflow.log_metric("churn_rate_train", float(y_train.mean()))
            mlflow.log_metric("pipeline_fit_seconds", round(fit_seconds, 2))
            mlflow.log_metric("pct_null_features", float((stats["null_rate"] > 0).mean()))

            # Top-10 features by |Spearman ρ| with churn
            if "spearman_with_churn" in stats.columns:
                for feat, rho in stats["spearman_with_churn"].abs().head(10).items():
                    mlflow.log_metric(f"rho_{feat}", float(rho))

            # Artifacts
            mlflow.log_artifact(str(FEATURES_DIR / "feature_stats.csv"), "feature_engineering")
            mlflow.log_artifact(str(FEATURES_DIR / "feature_manifest.json"), "feature_engineering")
            mlflow.log_artifact(str(FEATURES_DIR / "feature_pipeline.pkl"), "feature_engineering")

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
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument(
        "--dataset",
        choices=["expresso", "moroccan"],
        default="expresso",
        help="Source dataset to use",
    )
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(use_mlflow=args.use_mlflow, config_path=args.config, dataset=args.dataset)
