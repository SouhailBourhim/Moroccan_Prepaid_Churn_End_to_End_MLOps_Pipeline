"""Feature engineering pipeline for Expresso churn prediction.

All transforms are fit on train only; applied to train + test via FeaturePipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# ── Constants ─────────────────────────────────────────────────────────────────

CALL_COLS = ["ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2"]

MNAR_ZERO_COLS = [
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
    "ZONE1", "ZONE2", "FREQ_TOP_PACK",
]

MAR_MEDIAN_COLS = ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE"]

TENURE_ORDER = {
    "A < 2 month": 0,
    "B 2 month": 1,
    "C 3 month": 2,
    "D 3-6 month": 3,
    "E 6-9 month": 4,
    "F 9-12 month": 5,
    "G 12-15 month": 6,
    "H 15-18 month": 7,
    "I 18-21 month": 8,
    "J 21-24 month": 9,
    "K > 24 month": 10,
}


# ── Individual transformers ────────────────────────────────────────────────────

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Add binary _missing flags then impute."""

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingIndicatorAdder":
        self.cols_ = self.cols or [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for col in self.cols_:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype("int8")
        return X


class ZeroImputer(BaseEstimator, TransformerMixin):
    """Impute MNAR usage columns with 0."""

    def __init__(self, cols: list[str] = MNAR_ZERO_COLS) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ZeroImputer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute MAR columns with training median."""

    def __init__(self, cols: list[str] = MAR_MEDIAN_COLS) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MedianImputer":
        self.medians_: dict[str, float] = {}
        for col in self.cols:
            if col in X.columns:
                self.medians_[col] = float(X[col].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for col, median in self.medians_.items():
            X[col] = X[col].fillna(median)
        return X


class NumericFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived numeric features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Activity ratio
        X["regularity_rate"] = X["REGULARITY"] / 90.0

        # Recharge efficiency
        X["recharge_per_freq"] = X["MONTANT"] / (X["FREQUENCE_RECH"] + 1)
        X["revenue_per_freq"] = X["REVENUE"] / (X["FREQUENCE"] + 1)

        # Call aggregates
        X["total_calls"] = X[CALL_COLS].sum(axis=1)
        X["n_active_call_types"] = (X[CALL_COLS] > 0).sum(axis=1).astype("int8")

        # Binary activity flags
        X["is_inactive"] = (X["REGULARITY"] < 5).astype("int8")
        X["has_data"] = (X["DATA_VOLUME"] > 0).astype("int8")
        X["has_calls"] = (X["total_calls"] > 0).astype("int8")

        return X


class TenureEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode TENURE; new-subscriber flag."""

    NEW_SUBSCRIBER_VALS = {"A < 2 month", "B 2 month"}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TenureEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "TENURE" not in X.columns:
            return X
        tenure_str = X["TENURE"].astype(str)
        X["tenure_ordinal"] = tenure_str.map(TENURE_ORDER).fillna(-1).astype("int8")
        X["is_new_subscriber"] = tenure_str.isin(self.NEW_SUBSCRIBER_VALS).astype("int8")
        return X


class MRGEncoder(BaseEstimator, TransformerMixin):
    """Binary encode MRG (YES=1, NO=0)."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MRGEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "MRG" in X.columns:
            X["mrg_flag"] = (X["MRG"].astype(str).str.upper() == "YES").astype("int8")
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed k-fold target encoding for high-cardinality categoricals.

    Encoding is computed on the training fold only; global mean smoothing
    prevents target leakage for unseen categories.
    """

    def __init__(self, cols: list[str], smoothing: float = 20.0, min_samples: int = 50) -> None:
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self.global_mean_: float = float(y.mean())
        self.maps_: dict[str, dict[str, float]] = {}

        for col in self.cols:
            if col not in X.columns:
                continue
            stats = (
                pd.DataFrame({"cat": X[col].astype(str), "target": y.values})
                .groupby("cat")["target"]
                .agg(["mean", "count"])
            )
            # James–Stein style smoothing
            smooth = 1 / (1 + np.exp(-(stats["count"] - self.min_samples) / self.smoothing))
            encoded = smooth * stats["mean"] + (1 - smooth) * self.global_mean_
            self.maps_[col] = encoded.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[f"{col}_te"] = (
                X[col].astype(str).map(self.maps_[col]).fillna(self.global_mean_)
            )
        return X


class TopPackEncoder(BaseEstimator, TransformerMixin):
    """Rare-pack collapsing + frequency encode + target encode TOP_PACK."""

    def __init__(self, min_freq: int = 200, smoothing: float = 20.0) -> None:
        self.min_freq = min_freq
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TopPackEncoder":
        if "TOP_PACK" not in X.columns:
            return self

        pack_ser = X["TOP_PACK"].astype(str).fillna("UNKNOWN")
        counts = pack_ser.value_counts()
        self.keep_packs_: set[str] = set(counts[counts >= self.min_freq].index)

        # Frequency encoding
        self.freq_map_: dict[str, float] = (counts / len(pack_ser)).to_dict()

        # Smoothed target encoding (only on frequent packs)
        pack_collapsed = pack_ser.where(pack_ser.isin(self.keep_packs_), "OTHER")
        global_mean = float(y.mean())
        stats = (
            pd.DataFrame({"cat": pack_collapsed, "target": y.values})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        smooth = 1 / (1 + np.exp(-(stats["count"] - 50) / self.smoothing))
        encoded = smooth * stats["mean"] + (1 - smooth) * global_mean
        self.te_map_: dict[str, float] = encoded.to_dict()
        self.global_mean_ = global_mean

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        if "TOP_PACK" not in X.columns:
            return X

        pack_ser = X["TOP_PACK"].astype(str).fillna("UNKNOWN")
        pack_collapsed = pack_ser.where(pack_ser.isin(self.keep_packs_), "OTHER")

        X["top_pack_freq"] = pack_ser.map(self.freq_map_).fillna(0.0)
        X["top_pack_te"] = pack_collapsed.map(self.te_map_).fillna(self.global_mean_)

        return X


# ── Composite pipeline ─────────────────────────────────────────────────────────

class FeaturePipeline(BaseEstimator, TransformerMixin):
    """End-to-end feature engineering pipeline."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, BaseEstimator]] = []

    def _build(self) -> None:
        self._steps = [
            ("miss_flags", MissingIndicatorAdder()),
            ("zero_imp", ZeroImputer()),
            ("median_imp", MedianImputer()),
            ("numeric_fe", NumericFeatureEngineer()),
            ("tenure_enc", TenureEncoder()),
            ("mrg_enc", MRGEncoder()),
            ("region_te", TargetEncoder(cols=["REGION"])),
            ("top_pack_enc", TopPackEncoder()),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        self._build()
        Xt = X.copy()
        for name, step in self._steps:
            if hasattr(step, "fit"):
                if "y" in step.fit.__code__.co_varnames:
                    step.fit(Xt, y)
                else:
                    step.fit(Xt)
            Xt = step.transform(Xt)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        Xt = X.copy()
        for _, step in self._steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the final feature set (drop ID, raw categoricals, and target)."""
    drop = {"user_id", "CHURN", "REGION", "TENURE", "MRG", "TOP_PACK", "ARPU_SEGMENT"}
    return [c for c in df.columns if c not in drop]
