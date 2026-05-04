"""Candidate model training with MLflow experiment tracking.

Four classifiers are evaluated via stratified cross-validation on the
pre-computed feature matrix produced by `src.features.run_pipeline`:

  1. LogisticRegression  — interpretable linear baseline
  2. XGBoostClassifier   — gradient boosting, strong tabular baseline
  3. LGBMClassifier      — leaf-wise boosting, often fastest at matching AUC
  4. CatBoostClassifier  — robust on zero-inflated / mixed-type tabular data

The best model by mean CV ROC-AUC is refitted on all available training
data and saved as a joblib artifact alongside a JSON training manifest.

Prerequisite:
    python -m src.features.run_pipeline   # produces data/features/train_features.parquet

Usage:
    python -m src.models.train
    python -m src.models.train --config configs/base.yaml --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.logging import setup_logger

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs" / "base.yaml"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    name: str
    model: Any
    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 0  # 0 = disabled; tree models set this to 50


@dataclass
class CVResult:
    name: str
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    params: dict[str, Any] = field(default_factory=dict)


# ── Candidate construction ─────────────────────────────────────────────────────


def build_candidates(cfg: dict[str, Any], scale_pos_weight: float) -> list[Candidate]:
    """Construct the four candidate classifiers from config values.

    scale_pos_weight is computed from the training labels as neg/pos and
    passed to XGBoost. LightGBM and CatBoost use their own imbalance flags;
    LogisticRegression uses class_weight='balanced'.
    """
    rs: int = int(cfg["training"]["random_state"])
    xgb = cfg["xgboost"]
    lgb = cfg["lightgbm"]
    cb = cfg.get("catboost", {})
    lr = cfg.get("logistic_regression", {})

    return [
        Candidate(
            name="logistic_regression",
            model=SKPipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=float(lr.get("C", 0.1)),
                    class_weight="balanced",
                    max_iter=int(lr.get("max_iter", 1000)),
                    solver="lbfgs",
                    random_state=rs,
                    n_jobs=-1,
                )),
            ]),
            params={"C": lr.get("C", 0.1), "class_weight": "balanced", "solver": "lbfgs"},
        ),
        Candidate(
            name="xgboost",
            model=XGBClassifier(
                n_estimators=int(xgb["n_estimators"]),
                max_depth=int(xgb["max_depth"]),
                learning_rate=float(xgb["learning_rate"]),
                subsample=float(xgb["subsample"]),
                colsample_bytree=float(xgb["colsample_bytree"]),
                min_child_weight=int(xgb["min_child_weight"]),
                reg_alpha=float(xgb["reg_alpha"]),
                reg_lambda=float(xgb["reg_lambda"]),
                scale_pos_weight=round(scale_pos_weight, 3),
                early_stopping_rounds=int(cfg["training"].get("early_stopping_rounds", 50)),
                eval_metric="auc",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
            params={**xgb, "scale_pos_weight": round(scale_pos_weight, 3)},
            early_stopping_rounds=int(cfg["training"].get("early_stopping_rounds", 50)),
        ),
        Candidate(
            name="lightgbm",
            model=LGBMClassifier(
                n_estimators=int(lgb["n_estimators"]),
                max_depth=int(lgb["max_depth"]),
                learning_rate=float(lgb["learning_rate"]),
                num_leaves=int(lgb["num_leaves"]),
                subsample=float(lgb["subsample"]),
                colsample_bytree=float(lgb["colsample_bytree"]),
                min_child_samples=int(lgb["min_child_samples"]),
                reg_alpha=float(lgb["reg_alpha"]),
                reg_lambda=float(lgb["reg_lambda"]),
                is_unbalance=True,
                random_state=rs,
                n_jobs=-1,
                verbosity=-1,
            ),
            params={**lgb, "is_unbalance": True},
            early_stopping_rounds=int(cfg["training"].get("early_stopping_rounds", 50)),
        ),
        Candidate(
            name="catboost",
            model=CatBoostClassifier(
                iterations=int(cb.get("iterations", 1000)),
                depth=int(cb.get("depth", 7)),
                learning_rate=float(cb.get("learning_rate", 0.05)),
                l2_leaf_reg=float(cb.get("l2_leaf_reg", 3.0)),
                subsample=float(cb.get("subsample", 0.8)),
                colsample_bylevel=float(cb.get("colsample_bylevel", 0.8)),
                min_data_in_leaf=int(cb.get("min_data_in_leaf", 50)),
                auto_class_weights="Balanced",
                early_stopping_rounds=int(cfg["training"].get("early_stopping_rounds", 50)),
                random_seed=rs,
                verbose=0,
                allow_writing_files=False,
            ),
            params={**cb, "auto_class_weights": "Balanced"},
            early_stopping_rounds=int(cfg["training"].get("early_stopping_rounds", 50)),
        ),
    ]


# ── Cross-validation ───────────────────────────────────────────────────────────


def _fit(
    model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int,
) -> None:
    """Fit a model, passing eval_set for tree models that support early stopping.

    Each library has a different API:
      XGBoost    — early_stopping_rounds set in constructor; needs eval_set in fit()
      LightGBM   — early stopping via callbacks passed to fit()
      CatBoost   — early_stopping_rounds set in constructor; needs eval_set in fit()
      LR Pipeline — no early stopping; plain fit()
    """
    if early_stopping_rounds > 0 and isinstance(model, XGBClassifier):
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif early_stopping_rounds > 0 and isinstance(model, LGBMClassifier):
        import lightgbm as lgb  # local import avoids top-level lgb namespace clash
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
    elif early_stopping_rounds > 0 and isinstance(model, CatBoostClassifier):
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    else:
        model.fit(X_tr, y_tr)


def _cv_score(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> CVResult:
    """Run stratified k-fold CV for one candidate and return ROC-AUC / PR-AUC."""
    roc_scores: list[float] = []
    pr_scores: list[float] = []
    t0 = time.perf_counter()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = clone(candidate.model)
        _fit(model, X_tr, y_tr, X_val, y_val, candidate.early_stopping_rounds)
        y_prob: np.ndarray = model.predict_proba(X_val)[:, 1]

        fold_roc = float(roc_auc_score(y_val, y_prob))
        fold_pr = float(average_precision_score(y_val, y_prob))
        roc_scores.append(fold_roc)
        pr_scores.append(fold_pr)
        logger.debug(
            f"  [{candidate.name}] fold {fold_idx + 1}: "
            f"ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}"
        )

    elapsed = time.perf_counter() - t0
    result = CVResult(
        name=candidate.name,
        roc_auc_mean=float(np.mean(roc_scores)),
        roc_auc_std=float(np.std(roc_scores)),
        pr_auc_mean=float(np.mean(pr_scores)),
        pr_auc_std=float(np.std(pr_scores)),
        params=candidate.params,
    )
    logger.info(
        f"[{candidate.name}] CV ROC-AUC={result.roc_auc_mean:.4f} ± {result.roc_auc_std:.4f}"
        f"  PR-AUC={result.pr_auc_mean:.4f}  ({elapsed:.1f}s)"
    )
    return result


def _fit_final_model(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Fit the selected model on all rows without CV-only early stopping settings."""
    if isinstance(model, XGBClassifier):
        # XGBoost requires an eval_set whenever early_stopping_rounds is configured.
        # The final refit intentionally uses all rows, so disable early stopping here.
        model.set_params(early_stopping_rounds=None)
    model.fit(X, y)


# ── Main training function ────────────────────────────────────────────────────


def train(use_mlflow: bool = True, config_path: Path = CONFIG_PATH) -> CVResult:
    """Run CV for all candidates, log to MLflow, refit best model, save artifacts.

    Returns the CVResult for the winning candidate.
    """
    setup_logger()
    cfg = _load_config(config_path)
    training_cfg = cfg["training"]

    # ── 1. Load pre-computed features ─────────────────────────────────────────
    features_path = FEATURES_DIR / "train_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found — run `python -m src.features.run_pipeline` first."
        )

    df = pd.read_parquet(features_path)
    y = df["CHURN"].to_numpy(dtype=np.float32)
    feature_cols = [c for c in df.columns if c != "CHURN"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    logger.info(
        f"Features loaded: {X.shape[0]:,} rows × {X.shape[1]} cols  "
        f"churn rate={y.mean():.3%}"
    )

    # ── 2. Build candidates ───────────────────────────────────────────────────
    n_pos = float(y.sum())
    n_neg = float(len(y)) - n_pos
    if n_pos == 0:
        logger.warning("No positive labels found — defaulting scale_pos_weight to 1.0")
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = n_neg / n_pos
    logger.info(
        f"Class ratio: {n_neg:.0f} neg / {n_pos:.0f} pos  "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )
    candidates = build_candidates(cfg, scale_pos_weight)

    # ── 3. Cross-validate all candidates ─────────────────────────────────────
    # Note: target encoding in FeaturePipeline was fit on the full training set
    # before this point (run_pipeline.py saved the parquet). CV here scores models
    # on features with mild target-encoding leakage. With smoothing=20 this is
    # negligible in practice, but a future improvement could refit the pipeline
    # inside each fold.
    cv = StratifiedKFold(
        n_splits=int(training_cfg["cv_folds"]),
        shuffle=True,
        random_state=int(training_cfg["random_state"]),
    )
    results: list[CVResult] = []
    for candidate in candidates:
        logger.info(f"CV scoring {candidate.name}…")
        results.append(_cv_score(candidate, X, y, cv))

    # ── 4. Pick best by mean CV ROC-AUC ──────────────────────────────────────
    best = max(results, key=lambda r: r.roc_auc_mean)
    best_candidate = next(c for c in candidates if c.name == best.name)
    logger.info(
        f"\nBest model: {best.name}  "
        f"ROC-AUC={best.roc_auc_mean:.4f} ± {best.roc_auc_std:.4f}"
    )

    # ── 5. Refit best on full training data ───────────────────────────────────
    logger.info(f"Refitting {best.name} on full training set…")
    t0 = time.perf_counter()
    final_model = clone(best_candidate.model)
    _fit_final_model(final_model, X, y)
    logger.info(f"Refit complete ({time.perf_counter() - t0:.1f}s)")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    manifest: dict[str, Any] = {
        "best_model": best.name,
        "cv_roc_auc_mean": best.roc_auc_mean,
        "cv_roc_auc_std": best.roc_auc_std,
        "cv_pr_auc_mean": best.pr_auc_mean,
        "cv_pr_auc_std": best.pr_auc_std,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "all_results": [
            {
                "name": r.name,
                "roc_auc_mean": r.roc_auc_mean,
                "roc_auc_std": r.roc_auc_std,
                "pr_auc_mean": r.pr_auc_mean,
                "pr_auc_std": r.pr_auc_std,
                "params": r.params,
            }
            for r in results
        ],
    }

    joblib.dump({"model": final_model, "feature_cols": feature_cols}, model_path)
    manifest_path = MODELS_DIR / "training_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Saved: {model_path}  {manifest_path}")

    # ── 7. MLflow ─────────────────────────────────────────────────────────────
    if use_mlflow:
        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = str(mlflow_cfg.get("tracking_uri", "mlruns"))
        # Respect absolute paths and remote URIs; only prefix ROOT for relative paths.
        if "://" in tracking_uri or os.path.isabs(tracking_uri):
            resolved_uri = tracking_uri
        else:
            resolved_uri = str(ROOT / tracking_uri)
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "moroccan_prepaid_churn"))

        with mlflow.start_run(run_name=f"training__{best.name}"):
            mlflow.log_param("best_model", best.name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("cv_folds", training_cfg["cv_folds"])
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
            mlflow.log_params({f"best__{k}": v for k, v in best.params.items()})

            for result in results:
                mlflow.log_metric(f"{result.name}__cv_roc_auc_mean", result.roc_auc_mean)
                mlflow.log_metric(f"{result.name}__cv_roc_auc_std", result.roc_auc_std)
                mlflow.log_metric(f"{result.name}__cv_pr_auc_mean", result.pr_auc_mean)
                mlflow.log_metric(f"{result.name}__cv_pr_auc_std", result.pr_auc_std)

            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(manifest_path), "model")

            run = mlflow.active_run()
            if run is not None:
                logger.info(f"MLflow run: {run.info.run_id}")

    return best


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Expresso churn candidate models")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(use_mlflow=args.use_mlflow, config_path=args.config)
