"""Optuna hyperparameter search for CatBoost or XGBoost on churn features.

Strategy
--------
Tuning on 1.6M rows with 5-fold CV × 50 trials would take many hours.
Instead we:
  1. Draw a stratified subsample (default 400k rows) for the Optuna search.
     This is large enough to capture the signal but fast enough to finish in
     a reasonable time (~30–90 min on CPU depending on model).
  2. Use 3-fold CV during search (not 5) to halve the per-trial cost.
  3. Use early stopping so `iterations`/`n_estimators` is a ceiling,
     not a fixed cost — bad configs terminate early.
  4. After Optuna picks the best params, re-validate them on the full
     training set with the standard 5-fold CV to confirm the gain holds.
  5. Save the best params to models/tuning_results_{model}.json and patch
     them into configs/base.yaml so subsequent train.py runs pick them up.

Usage
-----
    python -m src.models.tune                            # CatBoost, 50 trials
    python -m src.models.tune --model xgboost            # XGBoost, 50 trials
    python -m src.models.tune --model xgboost --trials 30 --sample 200000
    python -m src.models.tune --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.utils.logging import setup_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs" / "base.yaml"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


# ── Search spaces ─────────────────────────────────────────────────────────────


def _suggest_catboost(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    return {
        "iterations": 2000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 15.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300),
        "auto_class_weights": "Balanced",
        "early_stopping_rounds": 50,
        "verbose": 0,
        "allow_writing_files": False,
        "random_seed": random_state,
    }


def _suggest_xgboost(
    trial: optuna.Trial, random_state: int, scale_pos_weight: float
) -> dict[str, Any]:
    return {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 300),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


# ── Objective ─────────────────────────────────────────────────────────────────


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    model_name: str,
    random_state: int,
    scale_pos_weight: float,
) -> float:
    """Optuna objective: mean CV ROC-AUC with early stopping and pruning."""
    if model_name == "catboost":
        params = _suggest_catboost(trial, random_state)
    else:
        params = _suggest_xgboost(trial, random_state, scale_pos_weight)

    scores: list[float] = []
    best_iters: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if model_name == "catboost":
            model: Any = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            best_iters.append(int(model.get_best_iteration() or params["iterations"]))
        else:
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best_iters.append(int(model.best_iteration + 1))

        y_prob: np.ndarray = model.predict_proba(X_val)[:, 1]
        fold_auc = float(roc_auc_score(y_val, y_prob))
        scores.append(fold_auc)

        trial.report(float(np.mean(scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
    trial.set_user_attr("roc_auc_std", float(np.std(scores)))
    return float(np.mean(scores))


# ── Full-data validation ───────────────────────────────────────────────────────


def _full_cv_score(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    model_name: str,
) -> tuple[float, float]:
    """Re-evaluate the tuned params on the full dataset with 5-fold CV."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        if model_name == "catboost":
            model: Any = CatBoostClassifier(**params)
            model.fit(X[train_idx], y[train_idx], eval_set=(X[val_idx], y[val_idx]))
        else:
            model = XGBClassifier(**params)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
        y_prob: np.ndarray = model.predict_proba(X[val_idx])[:, 1]
        scores.append(float(roc_auc_score(y[val_idx], y_prob)))
    return float(np.mean(scores)), float(np.std(scores))


# ── Config patch ──────────────────────────────────────────────────────────────


def _patch_config(
    config_path: Path, best_params: dict[str, Any], model_name: str
) -> None:
    """Write the tuned params back into configs/base.yaml."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    n_est = best_params.get("mean_best_iteration", 1000)

    if model_name == "catboost":
        cfg["catboost"] = {
            "iterations": n_est,
            "depth": best_params["depth"],
            "learning_rate": round(best_params["learning_rate"], 6),
            "l2_leaf_reg": round(best_params["l2_leaf_reg"], 4),
            "subsample": round(best_params["subsample"], 4),
            "colsample_bylevel": round(best_params["colsample_bylevel"], 4),
            "min_data_in_leaf": best_params["min_data_in_leaf"],
        }
    else:
        cfg["xgboost"] = {
            "n_estimators": n_est,
            "max_depth": best_params["max_depth"],
            "learning_rate": round(best_params["learning_rate"], 6),
            "subsample": round(best_params["subsample"], 4),
            "colsample_bytree": round(best_params["colsample_bytree"], 4),
            "min_child_weight": best_params["min_child_weight"],
            "reg_alpha": round(best_params["reg_alpha"], 6),
            "reg_lambda": round(best_params["reg_lambda"], 4),
            "gamma": round(best_params["gamma"], 4),
        }

    with open(config_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    logger.info(f"configs/base.yaml {model_name} section updated → {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def tune(
    n_trials: int = 50,
    sample_size: int = 400_000,
    use_mlflow: bool = True,
    config_path: Path = CONFIG_PATH,
    model_name: str = "catboost",
) -> dict[str, Any]:
    """Run Optuna search, validate winner on full data, persist results.

    Returns the best params dict (with meta-keys mean_best_iteration, etc.).
    """
    if model_name not in ("catboost", "xgboost"):
        raise ValueError(f"--model must be catboost or xgboost, got {model_name!r}")
    setup_logger()
    cfg = _load_config(config_path)
    training_cfg = cfg["training"]
    random_state = int(training_cfg["random_state"])

    # ── 1. Load features ──────────────────────────────────────────────────────
    features_path = FEATURES_DIR / "train_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} — run `python -m src.features.run_pipeline` first."
        )

    df = pd.read_parquet(features_path)
    y_full = df["CHURN"].to_numpy(dtype=np.float32)
    feature_cols = [c for c in df.columns if c != "CHURN"]
    X_full = df[feature_cols].to_numpy(dtype=np.float32)

    n_pos = float(y_full.sum())
    n_neg = float(len(y_full)) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    logger.info(
        f"Full dataset: {X_full.shape[0]:,} rows × {X_full.shape[1]} cols  "
        f"churn={y_full.mean():.3%}  model={model_name}"
    )

    # ── 2. Stratified subsample for Optuna ────────────────────────────────────
    actual_sample = min(sample_size, len(y_full))
    rng = np.random.default_rng(random_state)
    # Stratified: sample proportionally from each class
    pos_idx = np.where(y_full == 1)[0]
    neg_idx = np.where(y_full == 0)[0]
    n_pos = int(actual_sample * y_full.mean())
    n_neg = actual_sample - n_pos
    sample_idx = np.concatenate([
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
    ])
    rng.shuffle(sample_idx)
    X_sample = X_full[sample_idx]
    y_sample = y_full[sample_idx]

    logger.info(
        f"Optuna sample: {len(y_sample):,} rows  "
        f"churn={y_sample.mean():.3%}  (3-fold CV)"
    )

    # ── 3. Optuna study ───────────────────────────────────────────────────────
    search_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    t0 = time.perf_counter()
    logger.info(f"Starting Optuna search: {n_trials} trials…")

    study.optimize(
        lambda trial: _objective(
            trial, X_sample, y_sample, search_cv, model_name, random_state, scale_pos_weight
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    search_elapsed = time.perf_counter() - t0
    best_trial = study.best_trial
    logger.info(
        f"Search complete in {search_elapsed/60:.1f} min  "
        f"best trial #{best_trial.number}  "
        f"sample CV ROC-AUC={best_trial.value:.4f}"
    )

    # ── 4. Full-data validation ───────────────────────────────────────────────
    mean_best_iter = best_trial.user_attrs.get("mean_best_iteration", 1000)
    if model_name == "catboost":
        full_params: dict[str, Any] = {
            **best_trial.params,
            "iterations": mean_best_iter,
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "allow_writing_files": False,
            "random_seed": random_state,
        }
    else:
        full_params = {
            **best_trial.params,
            "n_estimators": mean_best_iter,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 50,
            "eval_metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

    logger.info("Validating best params on full dataset (5-fold CV)…")
    t1 = time.perf_counter()
    full_auc_mean, full_auc_std = _full_cv_score(
        full_params, X_full, y_full,
        cv_folds=int(training_cfg["cv_folds"]),
        random_state=random_state,
        model_name=model_name,
    )
    val_elapsed = time.perf_counter() - t1

    baseline_auc = cfg.get("catboost_tuning_baseline", {}).get("cv_roc_auc_mean", None)
    logger.info(
        f"Full CV ROC-AUC = {full_auc_mean:.4f} ± {full_auc_std:.4f}  "
        f"({val_elapsed/60:.1f} min)"
    )
    if baseline_auc:
        lift = (full_auc_mean - baseline_auc) * 100
        logger.info(f"Δ vs default config: {lift:+.2f} pp")

    # ── 5. Save results ───────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_out: dict[str, Any] = {
        **best_trial.params,
        "mean_best_iteration": mean_best_iter,
        "sample_cv_roc_auc": best_trial.value,
        "full_cv_roc_auc_mean": full_auc_mean,
        "full_cv_roc_auc_std": full_auc_std,
        "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "sample_size": actual_sample,
    }

    results_path = MODELS_DIR / f"tuning_results_{model_name}.json"
    with open(results_path, "w") as fh:
        json.dump(best_params_out, fh, indent=2)

    _patch_config(config_path, best_params_out, model_name)
    logger.info(f"Tuning results saved → {results_path}")

    # ── 6. Refit on full data and replace best_model.pkl ──────────────────────
    logger.info(f"Refitting tuned {model_name} on full training data…")
    t2 = time.perf_counter()
    if model_name == "catboost":
        final_model: Any = CatBoostClassifier(**full_params)
        final_model.fit(X_full, y_full)
    else:
        xgb_refit_params = {**full_params}
        xgb_refit_params.pop("early_stopping_rounds", None)
        final_model = XGBClassifier(**xgb_refit_params)
        final_model.fit(X_full, y_full)
    refit_elapsed = time.perf_counter() - t2
    logger.info(f"Refit complete ({refit_elapsed/60:.1f} min)")

    # Load existing feature_cols from the saved artifact and overwrite model
    existing: dict[str, Any] = {}
    model_path = MODELS_DIR / "best_model.pkl"
    if model_path.exists():
        existing = joblib.load(model_path)
    existing["model"] = final_model
    existing.setdefault("feature_cols", feature_cols)
    joblib.dump(existing, model_path)
    logger.info(f"best_model.pkl updated → {model_path}")

    # ── 7. MLflow ─────────────────────────────────────────────────────────────
    if use_mlflow:
        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = str(mlflow_cfg.get("tracking_uri", "mlruns"))
        import os
        resolved_uri = (
            tracking_uri
            if "://" in tracking_uri or os.path.isabs(tracking_uri)
            else str(ROOT / tracking_uri)
        )
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "moroccan_prepaid_churn"))

        with mlflow.start_run(run_name=f"{model_name}_optuna_tuning"):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("sample_size", actual_sample)
            mlflow.log_param("n_trials_pruned", best_params_out["n_trials_pruned"])
            mlflow.log_params(best_trial.params)
            mlflow.log_param("mean_best_iteration", mean_best_iter)
            mlflow.log_metric("sample_cv_roc_auc", float(best_trial.value or 0))
            mlflow.log_metric("full_cv_roc_auc_mean", full_auc_mean)
            mlflow.log_metric("full_cv_roc_auc_std", full_auc_std)
            mlflow.log_artifact(str(results_path), "tuning")
            run = mlflow.active_run()
            if run is not None:
                logger.info(f"MLflow run: {run.info.run_id}")

    logger.info(
        f"\n{'='*55}\n"
        f"  Tuning summary\n"
        f"{'='*55}\n"
        f"  Trials run      : {best_params_out['n_trials_completed']}"
        f"  ({best_params_out['n_trials_pruned']} pruned)\n"
        f"  Best trial #    : {best_trial.number}\n"
        f"  Sample CV AUC   : {best_trial.value:.4f}\n"
        f"  Full CV AUC     : {full_auc_mean:.4f} ± {full_auc_std:.4f}\n"
        f"{'='*55}"
    )

    return best_params_out


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CatBoost or XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["catboost", "xgboost"], default="catboost",
                   help="Model to tune")
    p.add_argument("--trials", type=int, default=50, dest="n_trials",
                   help="Number of Optuna trials")
    p.add_argument("--sample", type=int, default=400_000, dest="sample_size",
                   help="Stratified subsample size for search")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tune(
        n_trials=args.n_trials,
        sample_size=args.sample_size,
        use_mlflow=args.use_mlflow,
        config_path=args.config,
        model_name=args.model,
    )
