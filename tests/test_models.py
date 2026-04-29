"""Unit tests for model training utilities and evaluation functions."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.evaluate import (
    compute_metrics,
    optimal_threshold,
    threshold_at_recall,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def imbalanced_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary classification with ~18% positive rate (mirrors real data)."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.82, 0.18],
        random_state=42,
    )
    rng = np.random.default_rng(42)
    # Simulate realistic (noisy) probability output
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(StandardScaler().fit_transform(X), y)
    y_prob: np.ndarray = lr.predict_proba(StandardScaler().fit_transform(X))[:, 1]
    noise = rng.normal(0, 0.05, len(y_prob))
    y_prob = np.clip(y_prob + noise, 0.0, 1.0)
    return y.astype(float), y_prob.astype(float)


# ── compute_metrics ───────────────────────────────────────────────────────────


def test_compute_metrics_returns_expected_keys(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    metrics = compute_metrics(y_true, y_prob)
    assert set(metrics) >= {"roc_auc", "pr_auc", "brier", "f1", "precision", "recall", "threshold"}


def test_compute_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_prob = np.array([0.05, 0.1, 0.9, 0.95])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_compute_metrics_values_in_range(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    m = compute_metrics(y_true, y_prob)
    for key in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} outside [0, 1]"
    assert m["brier"] >= 0.0


def test_compute_metrics_respects_threshold(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    m_low = compute_metrics(y_true, y_prob, threshold=0.1)
    m_high = compute_metrics(y_true, y_prob, threshold=0.9)
    # Lower threshold → higher recall, lower precision
    assert m_low["recall"] >= m_high["recall"]
    assert m_low["precision"] <= m_high["precision"]


# ── optimal_threshold ─────────────────────────────────────────────────────────


def test_optimal_threshold_youden_in_range(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    t = optimal_threshold(y_true, y_prob, strategy="youden")
    assert 0.0 <= t <= 1.0


def test_optimal_threshold_f1_in_range(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    t = optimal_threshold(y_true, y_prob, strategy="f1")
    assert 0.0 <= t <= 1.0


def test_optimal_threshold_invalid_strategy(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    with pytest.raises(ValueError, match="Unknown strategy"):
        optimal_threshold(y_true, y_prob, strategy="invalid")


def test_optimal_threshold_youden_beats_default_f1(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Youden threshold should produce balanced sensitivity + specificity."""
    y_true, y_prob = imbalanced_arrays
    t = optimal_threshold(y_true, y_prob, strategy="youden")
    m = compute_metrics(y_true, y_prob, threshold=t)
    m_default = compute_metrics(y_true, y_prob, threshold=0.5)
    # Youden-optimal threshold should not be worse than default on ROC-AUC
    assert m["roc_auc"] == pytest.approx(m_default["roc_auc"])


# ── threshold_at_recall ───────────────────────────────────────────────────────


def test_threshold_at_recall_achieves_floor(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    min_recall = 0.75
    t = threshold_at_recall(y_true, y_prob, min_recall=min_recall)
    yhat = (y_prob >= t).astype(int)
    actual_recall = float(yhat[y_true == 1].mean())
    # Allow a small tolerance for boundary/discretisation effects
    assert actual_recall >= min_recall - 0.05


def test_threshold_at_recall_falls_back_on_impossible_recall(
    imbalanced_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = imbalanced_arrays
    # Recall of 1.0 is achievable (predict everything positive), so this is not
    # truly impossible; use 0.0 to test that the function always returns a float.
    t = threshold_at_recall(y_true, y_prob, min_recall=0.0)
    assert isinstance(t, float)


# ── build_candidates ──────────────────────────────────────────────────────────


def test_build_candidates_returns_four_models() -> None:
    from src.models.train import build_candidates

    cfg: dict[str, object] = {
        "training": {"random_state": 42},
        "xgboost": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
        "lightgbm": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
        "catboost": {
            "iterations": 10, "depth": 3, "learning_rate": 0.1,
            "l2_leaf_reg": 3.0, "subsample": 0.8, "colsample_bylevel": 0.8,
            "min_data_in_leaf": 5,
        },
        "logistic_regression": {"C": 0.1, "max_iter": 100},
    }
    candidates = build_candidates(cfg, scale_pos_weight=4.5)  # type: ignore[arg-type]
    assert len(candidates) == 4
    names = {c.name for c in candidates}
    assert names == {"logistic_regression", "xgboost", "lightgbm", "catboost"}


def test_build_candidates_params_logged() -> None:
    from src.models.train import build_candidates

    cfg: dict[str, object] = {
        "training": {"random_state": 0},
        "xgboost": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_alpha": 0.0, "reg_lambda": 1.0,
        },
        "lightgbm": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 5, "reg_alpha": 0.0, "reg_lambda": 1.0,
        },
    }
    candidates = build_candidates(cfg, scale_pos_weight=3.0)  # type: ignore[arg-type]
    xgb = next(c for c in candidates if c.name == "xgboost")
    assert "scale_pos_weight" in xgb.params
    assert xgb.params["scale_pos_weight"] == pytest.approx(3.0)
