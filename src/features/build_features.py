"""Feature engineering pipeline for Expresso churn prediction.

Design principles (grounded in cross-dataset EDA — notebooks/01–04):
- Every transformer is fit on train only and applied identically to train and test.
- Missing values are informative (MNAR): _missing flags are added before imputation.
- Each feature decision is traceable to a specific EDA or cross-dataset finding.
- All transformers are sklearn-compatible (BaseEstimator + TransformerMixin).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
MAR_MEDIAN_COLS = ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE"]

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
        X["is_inactive"] = (X["REGULARITY"] < self.inactive_threshold).astype("int8")
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

    Fit on training data only; unseen categories at inference map to global_mean_.
    """

    def __init__(
        self, cols: list[str], smoothing: float = 20.0, min_samples: int = 50
    ) -> None:
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetEncoder:
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
            lam = 1 / (1 + np.exp(-(stats["count"] - self.min_samples) / self.smoothing))
            self.maps_[col] = (lam * stats["mean"] + (1 - lam) * self.global_mean_).to_dict()

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
    """Rare-pack collapsing + frequency encoding + target encoding for TOP_PACK.

    EDA (notebook 01): TOP_PACK has 140 unique values; packs with < 200 subscribers
    are too sparse for reliable target encoding and are collapsed to 'OTHER'.
    Two output signals:
      top_pack_freq  — subscriber fraction using this pack (popularity signal).
      top_pack_te    — smoothed churn rate of this pack (engagement/value signal).

    Cross-dataset: plan engagement is a robust predictor across all three datasets
    (Orange voicemail plan, Cell2Cell retention offers, Expresso TOP_PACK).
    """

    def __init__(self, min_freq: int = 200, smoothing: float = 20.0) -> None:
        self.min_freq = min_freq
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TopPackEncoder:
        if "TOP_PACK" not in X.columns:
            return self

        pack_ser = X["TOP_PACK"].astype(str).fillna("UNKNOWN")
        counts = pack_ser.value_counts()
        self.keep_packs_: set[str] = set(counts[counts >= self.min_freq].index)
        self.freq_map_: dict[str, float] = (counts / len(pack_ser)).to_dict()

        pack_collapsed = pack_ser.where(pack_ser.isin(self.keep_packs_), "OTHER")
        global_mean = float(y.mean())
        stats = (
            pd.DataFrame({"cat": pack_collapsed, "target": y.values})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        lam = 1 / (1 + np.exp(-(stats["count"] - 50) / self.smoothing))
        self.te_map_: dict[str, float] = (lam * stats["mean"] + (1 - lam) * global_mean).to_dict()
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
    """

    def __init__(
        self,
        top_pack_min_freq: int = 200,
        target_enc_smoothing: float = 20.0,
        regularity_inactive_threshold: float = 5.0,
    ) -> None:
        self.top_pack_min_freq = top_pack_min_freq
        self.target_enc_smoothing = target_enc_smoothing
        self.regularity_inactive_threshold = regularity_inactive_threshold
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
            )),
            ("top_pack_enc", TopPackEncoder(
                min_freq=self.top_pack_min_freq,
                smoothing=self.target_enc_smoothing,
            )),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeaturePipeline:
        self._build()
        Xt = X.copy()
        for _name, step in self._steps:
            # Inspect fit signature: pass y only if the parameter exists.
            # This avoids fragile co_varnames checks and works correctly when
            # a step's fit only accepts X (e.g. stateless transformers with y=None default).
            sig_params = inspect.signature(step.fit).parameters
            if "y" in sig_params:
                step.fit(Xt, y)
            else:
                step.fit(Xt)
            Xt = step.transform(Xt)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_")
        Xt = X.copy()
        for _name, step in self._steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the final feature list, dropping IDs, raw categoricals, and target.

    Dropped columns:
      user_id               — identifier, not a feature
      CHURN                 — target variable
      REGION                — replaced by REGION_te (target encoded)
      TENURE                — replaced by tenure_ordinal + is_new_subscriber
      MRG                   — replaced by mrg_flag
      TOP_PACK              — replaced by top_pack_freq + top_pack_te
      ARPU_SEGMENT          — perfectly collinear with REVENUE (ρ = 1.00);
                              ARPU_SEGMENT_missing is also dropped (≡ REVENUE_missing)

    Redundant features removed after post-engineering correlation audit:
      REGULARITY            — ρ = 1.00 with regularity_rate (= REGULARITY / 90);
                              regularity_rate is kept as the normalised [0,1] form
      is_new_subscriber     — all-zero in dataset; TENURE never has < 2-month bands
      mrg_flag              — all-zero in dataset; MRG is always NO
      FREQUENCE_RECH_missing — ρ = 1.00 with MONTANT_missing (always co-missing);
                              MONTANT_missing is kept
      FREQUENCE_missing      — ρ = 1.00 with REVENUE_missing; REVENUE_missing kept
      FREQ_TOP_PACK_missing  — ρ = 1.00 with TOP_PACK_missing; TOP_PACK_missing kept
    """
    drop = {
        "user_id", "CHURN",
        "REGION", "TENURE", "MRG", "TOP_PACK",
        "ARPU_SEGMENT", "ARPU_SEGMENT_missing",
        # Perfectly collinear with regularity_rate
        "REGULARITY",
        # All-zero in Expresso dataset — zero information content
        "is_new_subscriber",
        "mrg_flag",
        # Perfectly correlated missing-flag duplicates
        "FREQUENCE_RECH_missing",
        "FREQUENCE_missing",
        "FREQ_TOP_PACK_missing",
    }
    return [c for c in df.columns if c not in drop]
