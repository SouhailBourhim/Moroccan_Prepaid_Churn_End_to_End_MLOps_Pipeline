"""Feature engineering pipeline for Expresso churn prediction.

Design principles (grounded in cross-dataset EDA — notebooks/01–04):
- Every transformer is fit on train only and applied identically to train and test.
- Target encoders are cross-fitted: in fit_transform, each training row is encoded
  with statistics from the other folds, so no row's own label leaks into its
  features (the same scheme as sklearn.preprocessing.TargetEncoder). transform()
  on unseen data uses statistics from the full training set.
- Missing values are informative (MNAR): _missing flags are added before imputation.
- Each feature decision is traceable to a specific EDA or cross-dataset finding.
- All transformers are sklearn-compatible (BaseEstimator + TransformerMixin).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

# ── Column group constants ─────────────────────────────────────────────────────

CALL_COLS = ["ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2"]
INTL_COLS = ["ZONE1", "ZONE2"]

# MNAR: missingness is not random — absent = subscriber never used that service.
# EDA (notebook 01): churn rate is 2–4× higher when these columns are missing.
MNAR_ZERO_COLS = [
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
    "ZONE1", "ZONE2", "FREQ_TOP_PACK",
]

# MAR: missing but not caused by service absence; impute with training median.
# REGULARITY is never missing in Expresso training data, but the API accepts it as
# optional; imputing it keeps regularity_rate / is_inactive defined at serving time.
MAR_MEDIAN_COLS = [
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "REGULARITY",
]

# _missing indicator names produced by MissingIndicatorAdder for MNAR columns.
# ServiceAbsenceEncoder reads these to count absent service channels.
MNAR_MISSING_FLAGS = [f"{c}_missing" for c in MNAR_ZERO_COLS]

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
    """Add binary {col}_missing flags for every column that has NaN in train.

    EDA (notebook 01): all high-missingness columns show strongly elevated churn
    rates when missing (MNAR pattern). Flags must be created before imputation so
    ServiceAbsenceEncoder and model can use them directly.
    """

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MissingIndicatorAdder:
        self.cols_ = self.cols or [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for col in self.cols_:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype("int8")
        return X


class ServiceAbsenceEncoder(BaseEstimator, TransformerMixin):
    """Customer-friction proxy: count of service channels the subscriber never used.

    Cross-dataset motivation (notebook 04): customer service calls are the dominant
    predictor in Orange Telecom (churn rate jumps from 8% → 50% at 4+ calls) and
    strong in Cell2Cell. Expresso has no equivalent feature. The closest proxy is
    counting how many service channels are entirely absent — subscribers who appear
    in the billing system but consume nothing map to the same disengaged/frustrated
    segment that drives high service-call volumes elsewhere.

    Must run AFTER MissingIndicatorAdder (reads the {col}_missing flags).

    Outputs
    -------
    n_services_absent   : int8  — count of MNAR service flags that were originally
                          missing, range [0, 7].
    is_ghost_subscriber : int8  — 1 if subscriber is absent from ≥ 5 of 7 service
                          channels; these subscribers have ~50%+ churn in Expresso.
    """

    GHOST_THRESHOLD: int = 5

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ServiceAbsenceEncoder:
        self.present_flags_ = [c for c in MNAR_MISSING_FLAGS if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        # Intersect with columns actually present — a test split may not contain
        # _missing flags for columns that had zero NaN in that slice.
        available = [c for c in self.present_flags_ if c in X.columns]
        if available:
            X["n_services_absent"] = X[available].sum(axis=1).astype("int8")
            X["is_ghost_subscriber"] = (
                X["n_services_absent"] >= self.GHOST_THRESHOLD
            ).astype("int8")
        return X


class ZeroImputer(BaseEstimator, TransformerMixin):
    """Impute MNAR usage columns with 0.

    EDA (notebook 01): these columns are missing because the subscriber never used
    that service, not because of a data collection failure. Setting them to 0 is
    semantically correct and preserves the distinction from low-but-nonzero usage.
    The _missing flags (created by MissingIndicatorAdder) ensure the original
    absence signal is not lost after imputation.
    """

    def __init__(self, cols: list[str] = MNAR_ZERO_COLS) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ZeroImputer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute MAR columns with training-set median.

    EDA (notebook 01): MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE
    are missing for ~34–36% of subscribers. The missingness correlates with churn
    (MAR, not MNAR) — captured by _missing flags — so median imputation is
    appropriate without introducing leakage.

    Median is computed on training data only and stored for application to test.
    """

    def __init__(self, cols: list[str] = MAR_MEDIAN_COLS) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MedianImputer:
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
    """Derive behavioural features from raw usage columns.

    Every feature below is tied to a specific EDA or cross-dataset finding:

    regularity_rate    Normalised REGULARITY [0, 1]. Dominant predictor across
                       Expresso (Spearman ρ = 0.53, DT Gini = 0.92). Cross-dataset:
                       engagement/activity is the universal churn signal (notebook 04).

    recharge_per_freq  MONTANT / (FREQUENCE_RECH + 1). Value per recharge event;
    revenue_per_freq   REVENUE / (FREQUENCE + 1). Revenue per transaction. Churners
                       show 3–4× lower ratios — spending less per interaction signals
                       lower commitment (confirmed across all three datasets, notebook 04).

    data_per_freq      DATA_VOLUME / (FREQUENCE + 1). Data intensity per transaction;
                       zero for non-data users. Separates light data browsers from
                       heavy data users within active subscribers.

    total_calls        Sum of all five call columns. Aggregate usage volume; churners
    intl_calls         ZONE1 + ZONE2. Separated from domestic calls because EDA showed
                       international usage has a distinct signal pattern (ratio ≈ 1.0
                       in total_calls but meaningful in isolation; notebook 01).

    n_active_call_types Count of non-zero call channels. Network breadth — subscribers
                        who use multiple call types are more embedded in the network.

    is_inactive        REGULARITY < 5 binary. Captures the extreme low-regularity tail
                       that drives the highest churn rates (>60% in lowest bucket,
                       notebook 01). Explicit flag lets tree models split cleanly here.

    has_data           DATA_VOLUME > 0 binary. Data service adoption; non-data users
    has_calls          total_calls > 0 binary. Any call activity.
    has_intl_usage     intl_calls > 0 binary. International connectivity indicator.

    All +1 denominators prevent division-by-zero on inactive subscribers.
    Transformer is stateless — no train information stored, no leakage risk.
    """

    def __init__(self, inactive_threshold: float = 5.0) -> None:
        self.inactive_threshold = inactive_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NumericFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Activity ratio — normalised; dominant predictor
        X["regularity_rate"] = X["REGULARITY"] / 90.0

        # Spend efficiency per interaction
        X["recharge_per_freq"] = X["MONTANT"] / (X["FREQUENCE_RECH"] + 1)
        X["revenue_per_freq"] = X["REVENUE"] / (X["FREQUENCE"] + 1)

        # Data intensity per transaction
        X["data_per_freq"] = X["DATA_VOLUME"] / (X["FREQUENCE"] + 1)

        # Call aggregates: domestic + international separated for interpretability
        X["total_calls"] = X[CALL_COLS].sum(axis=1)
        X["intl_calls"] = X[INTL_COLS].sum(axis=1)
        X["n_active_call_types"] = (X[CALL_COLS] > 0).sum(axis=1).astype("int8")

        # Binary activity flags
        inactive_threshold = getattr(self, "inactive_threshold", 5.0)
        X["is_inactive"] = (X["REGULARITY"] < inactive_threshold).astype("int8")
        X["has_data"] = (X["DATA_VOLUME"] > 0).astype("int8")
        X["has_calls"] = (X["total_calls"] > 0).astype("int8")
        X["has_intl_usage"] = (X["intl_calls"] > 0).astype("int8")

        return X


class TenureEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode TENURE; add new-subscriber flag.

    EDA (notebook 01): new subscribers (< 3 months) churn at 5× the rate of
    loyal subscribers (> 24 months). Cross-dataset: tenure is a top-3 predictor
    in Cell2Cell's decision tree (Gini = 0.49). Ordinal encoding preserves the
    monotonic duration relationship while keeping a binary flag for the high-risk
    early-tenure tail.
    """

    NEW_SUBSCRIBER_VALS: frozenset[str] = frozenset({"A < 2 month", "B 2 month"})

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TenureEncoder:
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
    """Binary encode MRG VAS subscription flag (YES → 1, NO → 0).

    EDA (notebook 01): MRG=YES is a value-added service indicator. Cross-dataset:
    plan/service enrollment correlates with lower churn across all three datasets
    (plan engagement = loyalty signal, notebook 04).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MRGEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "MRG" in X.columns:
            X["mrg_flag"] = (X["MRG"].astype(str).str.upper() == "YES").astype("int8")
        return X


def _smoothed_encoding(
    cats: pd.Series, y: np.ndarray, smoothing: float, min_samples: int
) -> tuple[dict[str, float], float]:
    """Return ({category: smoothed churn rate}, global churn rate) for one column."""
    global_mean = float(np.mean(y))
    stats = (
        pd.DataFrame({"cat": cats.to_numpy(), "target": y})
        .groupby("cat")["target"]
        .agg(["mean", "count"])
    )
    lam = 1 / (1 + np.exp(-(stats["count"] - min_samples) / smoothing))
    mapping: dict[str, float] = (lam * stats["mean"] + (1 - lam) * global_mean).to_dict()
    return mapping, global_mean


def _cross_fit_encoding(
    cats: pd.Series,
    y: np.ndarray,
    smoothing: float,
    min_samples: int,
    cv: int,
    random_state: int,
) -> np.ndarray:
    """Out-of-fold target encoding: each row is encoded without its own label."""
    encoded = np.empty(len(cats), dtype=np.float64)
    folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for fit_idx, enc_idx in folds.split(cats):
        mapping, global_mean = _smoothed_encoding(
            cats.iloc[fit_idx], y[fit_idx], smoothing, min_samples
        )
        encoded[enc_idx] = (
            cats.iloc[enc_idx].map(mapping).astype(float).fillna(global_mean).to_numpy()
        )
    return encoded


class TargetEncoder(BaseEstimator, TransformerMixin):
    """James–Stein smoothed target encoding for high-cardinality categoricals.

    EDA (notebook 01): REGION has 14 unique values with up to 2× churn rate
    variation across regions. One-hot encoding would add 14 sparse columns; ordinal
    encoding loses magnitude information. Target encoding compresses the signal into
    a single continuous column while handling unseen categories via the global mean.

    Smoothing formula: encoded = λ * group_mean + (1 - λ) * global_mean
    where λ = sigmoid((n - min_samples) / smoothing) — groups with fewer than
    min_samples observations are pulled toward the global mean, preventing overfit
    on small regions.

    fit_transform() cross-fits over `cv` folds so the encoded training rows never
    see their own label; transform() uses maps fit on the full training set.
    Unseen categories at inference map to global_mean_.
    """

    def __init__(
        self,
        cols: list[str],
        smoothing: float = 20.0,
        min_samples: int = 50,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetEncoder:
        y_arr = np.asarray(y, dtype=np.float64)
        self.global_mean_: float = float(y_arr.mean())
        self.maps_: dict[str, dict[str, float]] = {}
        for col in self.cols:
            if col in X.columns:
                self.maps_[col], _ = _smoothed_encoding(
                    X[col].astype(str), y_arr, self.smoothing, self.min_samples
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[f"{col}_te"] = (
                X[col].astype(str).map(self.maps_[col]).astype(float).fillna(self.global_mean_)
            )
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: object
    ) -> pd.DataFrame:
        if y is None:
            raise ValueError("TargetEncoder.fit_transform requires y")
        self.fit(X, y)
        if len(X) < self.cv:
            return self.transform(X)
        y_arr = np.asarray(y, dtype=np.float64)
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[f"{col}_te"] = _cross_fit_encoding(
                    X[col].astype(str), y_arr, self.smoothing, self.min_samples,
                    self.cv, self.random_state,
                )
        return X


class TopPackEncoder(BaseEstimator, TransformerMixin):
    """Rare-pack collapsing + frequency encoding + target encoding for TOP_PACK.

    EDA (notebook 01): TOP_PACK has 140 unique values; packs with < 200 subscribers
    are too sparse for reliable target encoding and are collapsed to 'OTHER'.
    Two output signals:
      top_pack_freq  — subscriber fraction using this pack (popularity signal).
      top_pack_te    — smoothed churn rate of this pack (engagement/value signal).

    Cross-dataset: plan engagement is a robust predictor across all three datasets
    (Orange voicemail plan, Cell2Cell retention offers, Expresso TOP_PACK).

    Subscribers with no pack are encoded as their own 'UNKNOWN' category. Like
    TargetEncoder, fit_transform() cross-fits top_pack_te; top_pack_freq uses no
    labels and is always computed from the full training set.
    """

    MIN_SAMPLES: int = 50

    def __init__(
        self,
        min_freq: int = 200,
        smoothing: float = 20.0,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.min_freq = min_freq
        self.smoothing = smoothing
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _pack_names(X: pd.DataFrame) -> pd.Series:
        packs = X["TOP_PACK"].astype("object")
        return packs.where(packs.notna(), "UNKNOWN").astype(str)

    def _collapse(self, packs: pd.Series) -> pd.Series:
        return packs.where(packs.isin(self.keep_packs_), "OTHER")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TopPackEncoder:
        if "TOP_PACK" not in X.columns:
            return self

        packs = self._pack_names(X)
        counts = packs.value_counts()
        self.keep_packs_: set[str] = set(counts[counts >= self.min_freq].index)
        self.freq_map_: dict[str, float] = (counts / len(packs)).to_dict()
        self.te_map_, self.global_mean_ = _smoothed_encoding(
            self._collapse(packs), np.asarray(y, dtype=np.float64),
            self.smoothing, self.MIN_SAMPLES,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "TOP_PACK" not in X.columns:
            return X.copy()
        check_is_fitted(self)
        X = X.copy()

        packs = self._pack_names(X)
        X["top_pack_freq"] = packs.map(self.freq_map_).astype(float).fillna(0.0)
        X["top_pack_te"] = (
            self._collapse(packs).map(self.te_map_).astype(float).fillna(self.global_mean_)
        )
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: object
    ) -> pd.DataFrame:
        if y is None:
            raise ValueError("TopPackEncoder.fit_transform requires y")
        self.fit(X, y)
        X_out = self.transform(X)
        if "TOP_PACK" in X.columns and len(X) >= self.cv:
            X_out["top_pack_te"] = _cross_fit_encoding(
                self._collapse(self._pack_names(X)), np.asarray(y, dtype=np.float64),
                self.smoothing, self.MIN_SAMPLES, self.cv, self.random_state,
            )
        return X_out


# ── Composite pipeline ─────────────────────────────────────────────────────────


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """End-to-end feature engineering pipeline for Expresso churn prediction.

    Pipeline order (each step is fit on train only):
        1. MissingIndicatorAdder  — {col}_missing flags before any imputation
        2. ServiceAbsenceEncoder  — n_services_absent, is_ghost_subscriber
        3. ZeroImputer            — MNAR cols → 0
        4. MedianImputer          — MAR cols → training median
        5. NumericFeatureEngineer — derived features (regularity_rate, calls, etc.)
        6. TenureEncoder          — ordinal + is_new_subscriber
        7. MRGEncoder             — YES/NO → mrg_flag
        8. TargetEncoder(REGION)  — James–Stein smoothed target encoding
        9. TopPackEncoder         — rare collapsing + freq + target encode

    Use fit_transform() on the training set: it returns cross-fitted target
    encodings for those rows. transform() is for data the pipeline was not fit on
    (holdout, scoring, live API traffic).
    """

    def __init__(
        self,
        top_pack_min_freq: int = 200,
        target_enc_smoothing: float = 20.0,
        regularity_inactive_threshold: float = 5.0,
        target_enc_cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.top_pack_min_freq = top_pack_min_freq
        self.target_enc_smoothing = target_enc_smoothing
        self.regularity_inactive_threshold = regularity_inactive_threshold
        self.target_enc_cv = target_enc_cv
        self.random_state = random_state
        self._steps: list[tuple[str, BaseEstimator]] = []

    def _build(self) -> None:
        self._steps = [
            ("miss_flags", MissingIndicatorAdder()),
            ("svc_absence", ServiceAbsenceEncoder()),
            ("zero_imp", ZeroImputer()),
            ("median_imp", MedianImputer()),
            ("numeric_fe", NumericFeatureEngineer(
                inactive_threshold=self.regularity_inactive_threshold
            )),
            ("tenure_enc", TenureEncoder()),
            ("mrg_enc", MRGEncoder()),
            ("region_te", TargetEncoder(
                cols=["REGION"],
                smoothing=self.target_enc_smoothing,
                cv=self.target_enc_cv,
                random_state=self.random_state,
            )),
            ("top_pack_enc", TopPackEncoder(
                min_freq=self.top_pack_min_freq,
                smoothing=self.target_enc_smoothing,
                cv=self.target_enc_cv,
                random_state=self.random_state,
            )),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeaturePipeline:
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        Xt = X.copy()
        for _name, step in self._steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: object
    ) -> pd.DataFrame:
        if y is None:
            raise ValueError("FeaturePipeline requires the target y to fit target encoders")
        self._build()
        Xt = X.copy()
        for _name, step in self._steps:
            # Every step accepts y; the target encoders use it to cross-fit.
            Xt = step.fit_transform(Xt, y)
        self.fitted_ = True
        return Xt


# Columns that are never model inputs: identifiers, the target, raw categoricals
# that have encoded replacements, and exact re-scalings of a kept feature.
STRUCTURAL_DROP: frozenset[str] = frozenset({
    "user_id", "CHURN",
    "REGION",        # → REGION_te
    "TENURE",        # → tenure_ordinal + is_new_subscriber
    "MRG",           # → mrg_flag
    "TOP_PACK",      # → top_pack_freq + top_pack_te
    "ARPU_SEGMENT",  # ρ = 1.00 with REVENUE (a rescaling of it)
    "ARPU_SEGMENT_missing",
    "REGULARITY",    # regularity_rate = REGULARITY / 90 is kept instead
})


def _column_digest(s: pd.Series) -> str:
    values = pd.util.hash_pandas_object(s.astype("float64"), index=False).to_numpy()
    return hashlib.sha1(values.tobytes()).hexdigest()


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the model feature list for a fitted training frame.

    Beyond STRUCTURAL_DROP, columns are pruned from the data itself rather than
    from a hard-coded list, so the same code is right for every dataset:
      - constant columns (e.g. mrg_flag on Expresso, where MRG is always NO);
      - exact duplicates of an earlier column (e.g. FREQUENCE_RECH_missing is
        always equal to MONTANT_missing on Expresso) — the first one is kept.

    Call this on the training frame only, and reuse the list for every other split.
    """
    keep: list[str] = []
    seen: dict[str, str] = {}
    for col in df.columns:
        if col in STRUCTURAL_DROP:
            continue
        s = df[col]
        if s.nunique(dropna=False) <= 1:
            continue
        digest = _column_digest(s)
        twin = seen.get(digest)
        if twin is not None and df[twin].astype("float64").equals(s.astype("float64")):
            continue
        seen.setdefault(digest, col)
        keep.append(col)
    return keep
