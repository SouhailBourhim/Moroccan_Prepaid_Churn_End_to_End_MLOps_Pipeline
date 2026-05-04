"""Evaluation utilities for the Expresso churn model.

Provides:
  - compute_metrics     : ROC-AUC, PR-AUC, F1, Brier at a given threshold
  - optimal_threshold   : Youden-J or F1-maximising threshold selection
  - threshold_at_recall : highest threshold that still meets a minimum recall
  - plot_roc_curve      : ROC curve Figure
  - plot_pr_curve       : Precision-Recall curve Figure
  - plot_calibration    : reliability diagram Figure
  - shap_summary        : SHAP beeswarm plot (saved to disk)

All plot functions return matplotlib Figure objects; the caller controls
display and persistence. Figures are closed inside shap_summary only because
SHAP's own plotting API manages the axes differently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Metric computation ─────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return a standard set of binary classification metrics.

    Returns NaN for ranking metrics (ROC-AUC, PR-AUC) when only one class is
    present in y_true, rather than raising. Threshold-based metrics (F1,
    precision, recall) are still computed as normal.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    yhat = (yp >= threshold).astype(int)

    n_classes = len(np.unique(yt))
    if n_classes < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(yt, yp))
        pr_auc = float(average_precision_score(yt, yp))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(yt, yp)),
        "f1": float(f1_score(yt, yhat, zero_division=0)),
        "precision": float(precision_score(yt, yhat, zero_division=0)),
        "recall": float(recall_score(yt, yhat, zero_division=0)),
        "threshold": threshold,
    }


# ── Threshold selection ────────────────────────────────────────────────────────


def optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    strategy: str = "youden",
) -> float:
    """Return the probability threshold that optimises a chosen strategy.

    Strategies
    ----------
    youden  Maximises sensitivity + specificity - 1 (Youden's J statistic).
            Balanced default for unknown cost ratios.
    f1      Maximises F1 score on the positive class.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(yt, yp)
        # sklearn >=1.3 prepends a sentinel thresholds[0] = inf; exclude it so
        # argmax can never select an out-of-range value for probability scoring.
        finite = np.isfinite(thresholds)
        j = (tpr - fpr)[finite]
        return float(thresholds[finite][np.argmax(j)])

    if strategy == "f1":
        prec, rec, thresholds = precision_recall_curve(yt, yp)
        denom = prec + rec
        f1 = np.where(denom == 0, 0.0, 2 * prec * rec / denom)
        # precision_recall_curve returns one more point than thresholds
        return float(thresholds[np.argmax(f1[:-1])])

    raise ValueError(f"Unknown strategy '{strategy}'. Choose 'youden' or 'f1'.")


def threshold_at_recall(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    min_recall: float = 0.80,
) -> float:
    """Return the highest threshold that still achieves at least min_recall.

    Business use-case: the retention team can contact at most N% of subscribers,
    but must catch at least 80% of true churners first — find the decision
    boundary that maximises precision subject to that recall floor.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    _prec, rec, thresholds = precision_recall_curve(yt, yp)
    # precision_recall_curve appends a sentinel point; align with thresholds
    mask = rec[:-1] >= min_recall
    if not mask.any():
        # Recall floor is unattainable at any threshold — return the lowest
        # threshold to maximise recall (predict as many positives as possible).
        return float(thresholds[0])
    return float(thresholds[mask][-1])


# ── Plot utilities ─────────────────────────────────────────────────────────────


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    label: str = "model",
) -> Figure:
    """Return a matplotlib Figure with the ROC curve."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    label: str = "model",
) -> Figure:
    """Return a matplotlib Figure with the Precision-Recall curve."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    prec, rec, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    baseline = float(yt.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(rec, prec, lw=2, where="post", label=f"{label} (AP={ap:.4f})")
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_calibration(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
    label: str = "model",
) -> Figure:
    """Return a matplotlib Figure with a reliability diagram (calibration curve)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(yt, yp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", lw=2, label=f"{label} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    return fig


# ── SHAP ──────────────────────────────────────────────────────────────────────


def shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    output_path: Path | None = None,
) -> None:
    """Generate a SHAP beeswarm summary plot and optionally save it to disk.

    Uses shap.Explainer which auto-selects TreeExplainer for GBDT models and
    LinearExplainer for the LogisticRegression pipeline.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required: pip install shap") from exc

    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(X)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
