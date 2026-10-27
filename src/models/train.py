"""Candidate model training with MLflow experiment tracking.

Four classifiers are evaluated via stratified cross-validation on the
pre-computed feature matrix produced by `src.features.run_pipeline`:

  1. LogisticRegression  — interpretable linear baseline
  2. XGBoostClassifier   — gradient boosting, strong tabular baseline
  3. LGBMClassifier      — leaf-wise boosting, often fastest at matching AUC
  4. CatBoostClassifier  — robust on zero-inflated / mixed-type tabular data

Protocol (all on the training split; the labeled holdout is never touched here):
  - Stratified k-fold CV. Boosters early-stop on an inner split of each training
    fold (src.models.fitting), never on the fold being scored.
  - The best model by mean CV ROC-AUC is refit on all training rows with the
    mean early-stopped round count from CV.
  - A calibrator is fit on the winner's out-of-fold predictions
    (src.models.calibrate), and decision thresholds (Youden-J, F1) are chosen on
    cross-fitted calibrated OOF probabilities. The holdout evaluation
    (src.models.evaluate_run) then applies those frozen thresholds.

Artifacts (models/{dataset}/):
    model.pkl               {"model": CalibratedChurnModel, "feature_cols", "thresholds", …}
    training_manifest.json  CV results, calibration choice, thresholds, OOF metrics

Prerequisite:
    python -m src.features.run_pipeline --dataset expresso

Usage:
    python -m src.models.train --dataset expresso
    python -m src.models.train --dataset moroccan --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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

from src.models.calibrate import CalibratedChurnModel, fit_calibration
from src.models.evaluate import compute_metrics, optimal_threshold
from src.models.fitting import fit_with_early_stopping, refit_with_iterations
from src.utils.config import BASE_CONFIG, DATASETS, dataset_paths, load_config, setup_mlflow
from src.utils.logging import setup_logger

# The operating point the API uses when a request does not pass a threshold.
DEFAULT_THRESHOLD_STRATEGY = "f1"


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    name: str
    model: Any
    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 0  # 0 = disabled; boosters use training.early_stopping_rounds


@dataclass
class CVResult:
    name: str
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    params: dict[str, Any] = field(default_factory=dict)
    # Mean early-stopped boosting rounds across folds (None for non-boosters).
    mean_best_iteration: int | None = None


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
    es_rounds = int(cfg["training"].get("early_stopping_rounds", 50))

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
                gamma=float(xgb.get("gamma", 0.0)),
                scale_pos_weight=round(scale_pos_weight, 3),
                eval_metric="auc",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
            params={
                **xgb,
                "gamma": float(xgb.get("gamma", 0.0)),
                "scale_pos_weight": round(scale_pos_weight, 3),
            },
            early_stopping_rounds=es_rounds,
        ),
        Candidate(
            name="lightgbm",
            model=LGBMClassifier(
                n_estimators=int(lgb["n_estimators"]),
                max_depth=int(lgb["max_depth"]),
                learning_rate=float(lgb["learning_rate"]),
                num_leaves=int(lgb["num_leaves"]),
                subsample=float(lgb["subsample"]),
                # LightGBM ignores `subsample` unless bagging is enabled via subsample_freq.
                subsample_freq=1,
                colsample_bytree=float(lgb["colsample_bytree"]),
                min_child_samples=int(lgb["min_child_samples"]),
                reg_alpha=float(lgb["reg_alpha"]),
                reg_lambda=float(lgb["reg_lambda"]),
                is_unbalance=True,
                metric="auc",
                random_state=rs,
                n_jobs=-1,
                verbosity=-1,
            ),
            params={**lgb, "is_unbalance": True, "subsample_freq": 1},
            early_stopping_rounds=es_rounds,
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
                eval_metric="AUC",
                random_seed=rs,
                verbose=0,
                allow_writing_files=False,
            ),
            params={**cb, "auto_class_weights": "Balanced"},
            early_stopping_rounds=es_rounds,
        ),
    ]




# ── Cross-validation ───────────────────────────────────────────────────────────


def _cv_score(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    random_state: int,
) -> tuple[CVResult, np.ndarray]:
    """Run stratified k-fold CV for one candidate.

    Returns the CV summary and the out-of-fold predicted probabilities (each row
    scored by the fold model that did not train on it).
    """
    roc_scores: list[float] = []
    pr_scores: list[float] = []
    best_iters: list[int] = []
    oof = np.zeros(len(y), dtype=np.float64)
    t0 = time.perf_counter()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        model = clone(candidate.model)
        n_iter = fit_with_early_stopping(
            model, X[train_idx], y[train_idx],
            rounds=candidate.early_stopping_rounds,
            random_state=random_state + fold_idx,
        )
        if n_iter is not None:
            best_iters.append(n_iter)
        y_prob: np.ndarray = model.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = y_prob

        fold_roc = float(roc_auc_score(y[val_idx], y_prob))
        fold_pr = float(average_precision_score(y[val_idx], y_prob))
        roc_scores.append(fold_roc)
        pr_scores.append(fold_pr)
        logger.debug(
            f"  [{candidate.name}] fold {fold_idx + 1}: "
            f"ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}  iters={n_iter}"
        )

    elapsed = time.perf_counter() - t0
    result = CVResult(
        name=candidate.name,
        roc_auc_mean=float(np.mean(roc_scores)),
        roc_auc_std=float(np.std(roc_scores)),
        pr_auc_mean=float(np.mean(pr_scores)),
        pr_auc_std=float(np.std(pr_scores)),
        params=candidate.params,
        mean_best_iteration=int(np.mean(best_iters)) if best_iters else None,
    )
    logger.info(
        f"[{candidate.name}] CV ROC-AUC={result.roc_auc_mean:.4f} ± {result.roc_auc_std:.4f}"
        f"  PR-AUC={result.pr_auc_mean:.4f}  iters={result.mean_best_iteration}"
        f"  ({elapsed:.1f}s)"
    )
    return result, oof


def _threshold_metrics(
    y: np.ndarray, probs: np.ndarray, thresholds: dict[str, float]
) -> dict[str, float]:
    """F1 / precision / recall at each named threshold, keyed '{name}_{metric}'."""
    out: dict[str, float] = {}
    for name, t in thresholds.items():
        m = compute_metrics(y, probs, threshold=t)
        for key in ("f1", "precision", "recall"):
            out[f"{name}_{key}"] = m[key]
    return out


# ── Main training function ────────────────────────────────────────────────────


def train(
    dataset: str = "expresso",
    use_mlflow: bool = True,
    config_path: Path = BASE_CONFIG,
) -> CVResult:
    """Run CV for all candidates, refit + calibrate the best, save artifacts.

    Returns the CVResult for the winning candidate.
    """
    setup_logger()
    cfg = load_config(dataset, config_path)
    paths = dataset_paths(dataset)
    paths.ensure_dirs()
    training_cfg = cfg["training"]
    rs = int(training_cfg["random_state"])
    cv_folds = int(training_cfg["cv_folds"])

    # ── 1. Load pre-computed training features (holdout is a separate file) ───
    if not paths.train_features.exists():
        raise FileNotFoundError(
            f"{paths.train_features} not found — run "
            f"`python -m src.features.run_pipeline --dataset {dataset}` first."
        )

    df = pd.read_parquet(paths.train_features)
    y = df["CHURN"].to_numpy(dtype=np.float32)
    feature_cols = [c for c in df.columns if c != "CHURN"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    logger.info(
        f"[{dataset}] Training features: {X.shape[0]:,} rows × {X.shape[1]} cols  "
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
    # Target encodings in train_features.parquet are cross-fitted (see
    # FeaturePipeline), so no row's own label is encoded into its features.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    results: list[CVResult] = []
    oof_by_model: dict[str, np.ndarray] = {}
    for candidate in candidates:
        logger.info(f"CV scoring {candidate.name}…")
        result, oof = _cv_score(candidate, X, y, cv, rs)
        results.append(result)
        oof_by_model[candidate.name] = oof

    # ── 4. Pick best by mean CV ROC-AUC ──────────────────────────────────────
    best = max(results, key=lambda r: r.roc_auc_mean)
    best_candidate = next(c for c in candidates if c.name == best.name)
    logger.info(
        f"Best model: {best.name}  "
        f"ROC-AUC={best.roc_auc_mean:.4f} ± {best.roc_auc_std:.4f}"
    )

    # ── 5. Refit best on all training rows with the CV round count ────────────
    logger.info(f"Refitting {best.name} on all training rows (iters={best.mean_best_iteration})…")
    t0 = time.perf_counter()
    final_model = refit_with_iterations(
        clone(best_candidate.model), X, y, best.mean_best_iteration
    )
    logger.info(f"Refit complete ({time.perf_counter() - t0:.1f}s)")

    # ── 6. Calibrate on OOF scores; choose thresholds on cross-fitted OOF ─────
    oof_raw = oof_by_model[best.name]
    calibration = fit_calibration(oof_raw, y, cv_folds=cv_folds, random_state=rs)
    oof_cal = calibration.cross_fitted
    thresholds = {
        "youden": optimal_threshold(y, oof_cal, strategy="youden"),
        "f1": optimal_threshold(y, oof_cal, strategy="f1"),
    }
    default_threshold = thresholds[DEFAULT_THRESHOLD_STRATEGY]
    oof_metrics: dict[str, float] = {
        "raw_brier": compute_metrics(y, oof_raw)["brier"],
        "calibrated_brier": compute_metrics(y, oof_cal)["brier"],
        **_threshold_metrics(y, oof_cal, thresholds),
    }
    rounded = {k: round(v, 4) for k, v in thresholds.items()}
    logger.info(
        f"Calibration: {calibration.method}  CV Brier by method={calibration.brier_by_method}"
        f"  thresholds={rounded}"
    )
    wrapped = CalibratedChurnModel(final_model, calibration.calibrator, calibration.method)

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "best_model": best.name,
        "cv_roc_auc_mean": best.roc_auc_mean,
        "cv_roc_auc_std": best.roc_auc_std,
        "cv_pr_auc_mean": best.pr_auc_mean,
        "cv_pr_auc_std": best.pr_auc_std,
        "n_iterations": best.mean_best_iteration,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "calibration": {
            "method": calibration.method,
            "cv_brier_by_method": calibration.brier_by_method,
        },
        "thresholds": thresholds,
        "default_threshold_strategy": DEFAULT_THRESHOLD_STRATEGY,
        "default_threshold": default_threshold,
        "oof_metrics": oof_metrics,
        "all_results": [
            {
                "name": r.name,
                "roc_auc_mean": r.roc_auc_mean,
                "roc_auc_std": r.roc_auc_std,
                "pr_auc_mean": r.pr_auc_mean,
                "pr_auc_std": r.pr_auc_std,
                "mean_best_iteration": r.mean_best_iteration,
                "params": r.params,
            }
            for r in results
        ],
    }
    artifact: dict[str, Any] = {
        "model": wrapped,
        "model_name": best.name,
        "dataset": dataset,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "default_threshold": default_threshold,
        "calibration_method": calibration.method,
    }
    joblib.dump(artifact, paths.model)
    with open(paths.training_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Saved: {paths.model}  {paths.training_manifest}")

    # ── 8. MLflow ─────────────────────────────────────────────────────────────
    if use_mlflow:
        import mlflow

        setup_mlflow(cfg)
        with mlflow.start_run(run_name=f"training__{dataset}__{best.name}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("best_model", best.name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
            mlflow.log_param("calibration_method", calibration.method)
            mlflow.log_param("n_iterations", best.mean_best_iteration)
            mlflow.log_params({f"best__{k}": v for k, v in best.params.items()})

            for result in results:
                mlflow.log_metric(f"{result.name}__cv_roc_auc_mean", result.roc_auc_mean)
                mlflow.log_metric(f"{result.name}__cv_roc_auc_std", result.roc_auc_std)
                mlflow.log_metric(f"{result.name}__cv_pr_auc_mean", result.pr_auc_mean)
                mlflow.log_metric(f"{result.name}__cv_pr_auc_std", result.pr_auc_std)
            for key, value in oof_metrics.items():
                mlflow.log_metric(f"oof__{key}", value)
            for name, t in thresholds.items():
                mlflow.log_metric(f"threshold__{name}", t)

            mlflow.log_artifact(str(paths.model), "model")
            mlflow.log_artifact(str(paths.training_manifest), "model")

            run = mlflow.active_run()
            if run is not None:
                logger.info(f"MLflow run: {run.info.run_id}")

    return best


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train churn candidate models")
    p.add_argument("--dataset", choices=DATASETS, default="expresso")
    p.add_argument("--config", type=Path, default=BASE_CONFIG)
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(dataset=args.dataset, use_mlflow=args.use_mlflow, config_path=args.config)
