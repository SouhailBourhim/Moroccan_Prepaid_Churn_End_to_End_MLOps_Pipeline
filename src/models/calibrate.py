"""Post-hoc probability calibration for the trained churn model.

The candidates are trained with class re-weighting (scale_pos_weight, Balanced,
is_unbalance), which ranks well but inflates probabilities. A 1-D calibrator
maps raw scores back to churn probabilities.

The calibrator is fit on **out-of-fold** predictions from training CV: every
score it sees comes from a fold model that never trained on that row. This is
sklearn's CalibratedClassifierCV(ensemble=False) scheme — the final model is
refit on all training rows and wrapped with the calibrator fit on OOF scores.
Fitting a calibrator on in-sample scores, or on the evaluation holdout, would
bias it; neither happens here.

Isotonic vs sigmoid (Platt) is chosen by cross-validated Brier score on the
same OOF scores, so the choice never looks at the holdout either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

CALIBRATION_METHODS: tuple[str, ...] = ("isotonic", "sigmoid")

Calibrator: TypeAlias = IsotonicRegression | LogisticRegression


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, y)
        return iso
    if method == "sigmoid":
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(scores.reshape(-1, 1), y)
        return lr
    raise ValueError(f"Unknown calibration method {method!r}")


def apply_calibrator(method: str, calibrator: Calibrator, scores: np.ndarray) -> np.ndarray:
    if method == "sigmoid":
        proba: np.ndarray = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return proba
    calibrated: np.ndarray = calibrator.predict(scores)
    return calibrated


class CalibratedChurnModel:
    """Wraps a pre-fitted base model with a 1-D probability calibrator.

    Exposes predict_proba(X) returning an (n, 2) array so it is a drop-in
    replacement for any sklearn-compatible classifier in the serving layer.
    """

    def __init__(self, base_model: Any, calibrator: Calibrator, method: str) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        raw: np.ndarray = self.base_model.predict_proba(X)[:, 1]
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cal = apply_calibrator(self.method, self.calibrator, self.predict_raw(X))
        return np.column_stack([1.0 - cal, cal])

    def __repr__(self) -> str:
        return (
            f"CalibratedChurnModel(base={self.base_model.__class__.__name__}, "
            f"method={self.method})"
        )


@dataclass
class CalibrationResult:
    method: str
    calibrator: Calibrator
    brier_by_method: dict[str, float]
    # Cross-fitted calibrated OOF probabilities for the chosen method: each value
    # comes from a calibrator that did not see that row. Use these (not the
    # in-sample calibrated scores) to pick decision thresholds.
    cross_fitted: np.ndarray = field(repr=False)


def fit_calibration(
    oof_scores: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    methods: tuple[str, ...] = CALIBRATION_METHODS,
) -> CalibrationResult:
    """Pick a calibration method by CV Brier on OOF scores, then fit it on all of them."""
    scores = np.asarray(oof_scores, dtype=np.float64)
    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(scores, y_arr))

    brier: dict[str, float] = {}
    cross: dict[str, np.ndarray] = {}
    for method in methods:
        out = np.empty_like(scores)
        for fit_idx, val_idx in splits:
            cal = fit_calibrator(method, scores[fit_idx], y_arr[fit_idx])
            out[val_idx] = apply_calibrator(method, cal, scores[val_idx])
        cross[method] = out
        brier[method] = float(brier_score_loss(y_arr, out))

    best = min(brier, key=lambda m: brier[m])
    return CalibrationResult(
        method=best,
        calibrator=fit_calibrator(best, scores, y_arr),
        brier_by_method=brier,
        cross_fitted=cross[best],
    )
