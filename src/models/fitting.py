"""Early-stopping helpers shared by training and tuning.

Early stopping must never watch the rows a model is scored on — otherwise the
number of trees is tuned on the evaluation fold and CV scores are optimistic.
Every early-stopped fit here therefore carves its own inner validation split
out of the training rows it was given.

Libraries are recognised by class name so this module does not import XGBoost,
LightGBM, or CatBoost itself.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

ES_FRACTION = 0.1
_BOOSTERS = frozenset({"XGBClassifier", "LGBMClassifier", "CatBoostClassifier"})


def supports_early_stopping(model: Any) -> bool:
    return type(model).__name__ in _BOOSTERS


def fit_with_early_stopping(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    rounds: int,
    random_state: int,
    es_fraction: float = ES_FRACTION,
) -> int | None:
    """Fit `model` on X/y, early-stopping on an inner stratified split of X/y.

    Returns the number of boosting rounds actually used, or None for models
    without early stopping (they are fit on all rows).
    """
    if rounds <= 0 or not supports_early_stopping(model):
        model.fit(X, y)
        return None

    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=es_fraction, stratify=y, random_state=random_state
    )
    name = type(model).__name__
    if name == "XGBClassifier":
        model.set_params(early_stopping_rounds=rounds)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        return int(model.best_iteration) + 1
    if name == "LGBMClassifier":
        import lightgbm as lgb

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(-1)],
        )
        return int(model.best_iteration_) or None
    # CatBoostClassifier
    model.set_params(early_stopping_rounds=rounds)
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    return int(model.get_best_iteration()) + 1


def refit_with_iterations(model: Any, X: np.ndarray, y: np.ndarray, n_iter: int | None) -> Any:
    """Fit on all rows with a fixed number of boosting rounds and no early stopping.

    `n_iter` is normally the mean early-stopped round count from cross-validation,
    so the final model has the same capacity that CV measured. Returns the fitted
    model (a fresh instance for CatBoost, whose set_params cannot unset a param).
    """
    name = type(model).__name__
    if name == "XGBClassifier":
        model.set_params(early_stopping_rounds=None)
        if n_iter:
            model.set_params(n_estimators=n_iter)
    elif name == "LGBMClassifier":
        if n_iter:
            model.set_params(n_estimators=n_iter)
    elif name == "CatBoostClassifier":
        params = model.get_params()
        params.pop("early_stopping_rounds", None)
        if n_iter:
            params["iterations"] = n_iter
        model = type(model)(**params)
    model.fit(X, y)
    return model
