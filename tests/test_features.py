"""Unit tests for feature engineering transforms."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    MNAR_MISSING_FLAGS,
    TENURE_ORDER,
    FeaturePipeline,
    MedianImputer,
    MissingIndicatorAdder,
    MRGEncoder,
    NumericFeatureEngineer,
    ServiceAbsenceEncoder,
    TargetEncoder,
    TenureEncoder,
    ZeroImputer,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "REGION": rng.choice(["DAKAR", "THIES", "SAINT-LOUIS", None], n),
            "TENURE": rng.choice(list(TENURE_ORDER.keys()) + [None], n),
            "MONTANT": rng.choice([np.nan, *rng.uniform(100, 10000, n - 50)], n),
            "FREQUENCE_RECH": rng.uniform(0, 30, n),
            "REVENUE": rng.uniform(0, 20000, n),
            "ARPU_SEGMENT": rng.uniform(0, 7000, n),
            "FREQUENCE": rng.uniform(0, 50, n),
            "DATA_VOLUME": rng.choice([np.nan, *rng.uniform(0, 5000, n - 100)], n),
            "ON_NET": rng.choice([np.nan, *rng.uniform(0, 500, n - 80)], n),
            "ORANGE": rng.choice([np.nan, *rng.uniform(0, 200, n - 80)], n),
            "TIGO": rng.choice([np.nan, *rng.uniform(0, 100, n - 80)], n),
            "ZONE1": rng.choice([np.nan, *rng.uniform(0, 50, n - 80)], n),
            "ZONE2": rng.choice([np.nan, *rng.uniform(0, 20, n - 80)], n),
            "MRG": rng.choice(["YES", "NO", None], n),
            "REGULARITY": rng.choice([np.nan, *rng.integers(0, 91, n - 30)], n),
            "TOP_PACK": rng.choice(["PackA", "PackB", "PackC", None], n),
            "FREQ_TOP_PACK": rng.choice([np.nan, *rng.uniform(0, 20, n - 50)], n),
        }
    )


@pytest.fixture
def target(sample_df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice([0, 1], len(sample_df), p=[0.82, 0.18]), name="CHURN")


# ── MissingIndicatorAdder ─────────────────────────────────────────────────────

def test_missing_flags_created(sample_df: pd.DataFrame) -> None:
    t = MissingIndicatorAdder()
    out = t.fit_transform(sample_df)
    missing_cols_in_input = [c for c in sample_df.columns if sample_df[c].isna().any()]
    for col in missing_cols_in_input:
        assert f"{col}_missing" in out.columns
        assert out[f"{col}_missing"].isin([0, 1]).all()


# ── ServiceAbsenceEncoder ─────────────────────────────────────────────────────


def test_service_absence_requires_missing_flags(sample_df: pd.DataFrame) -> None:
    # Must run after MissingIndicatorAdder so flags exist
    with_flags = MissingIndicatorAdder().fit_transform(sample_df)
    enc = ServiceAbsenceEncoder()
    out = enc.fit_transform(with_flags)
    assert "n_services_absent" in out.columns
    assert "is_ghost_subscriber" in out.columns


def test_service_absence_range(sample_df: pd.DataFrame) -> None:
    with_flags = MissingIndicatorAdder().fit_transform(sample_df)
    enc = ServiceAbsenceEncoder()
    out = enc.fit_transform(with_flags)
    n_flags = len([c for c in MNAR_MISSING_FLAGS if c in with_flags.columns])
    assert out["n_services_absent"].between(0, n_flags).all()
    assert out["is_ghost_subscriber"].isin([0, 1]).all()


def test_service_absence_ghost_threshold(sample_df: pd.DataFrame) -> None:
    # is_ghost_subscriber must be 1 whenever n_services_absent >= GHOST_THRESHOLD
    with_flags = MissingIndicatorAdder().fit_transform(sample_df)
    enc = ServiceAbsenceEncoder()
    out = enc.fit_transform(with_flags)
    mask_ghost = out["n_services_absent"] >= enc.GHOST_THRESHOLD
    assert out.loc[mask_ghost, "is_ghost_subscriber"].eq(1).all()
    assert out.loc[~mask_ghost, "is_ghost_subscriber"].eq(0).all()


def test_service_absence_no_leakage(sample_df: pd.DataFrame) -> None:
    # In real usage one MissingIndicatorAdder is fit on train and applied to both splits.
    # Verify ServiceAbsenceEncoder produces consistent columns in that scenario.
    miss_adder = MissingIndicatorAdder().fit(sample_df.iloc[:400])
    train_flagged = miss_adder.transform(sample_df.iloc[:400])
    test_flagged = miss_adder.transform(sample_df.iloc[400:])
    enc = ServiceAbsenceEncoder().fit(train_flagged)
    out_train = enc.transform(train_flagged)
    out_test = enc.transform(test_flagged)
    assert out_train.columns.tolist() == out_test.columns.tolist()


# ── ZeroImputer ───────────────────────────────────────────────────────────────

def test_zero_imputer_fills_na(sample_df: pd.DataFrame) -> None:
    t = ZeroImputer()
    out = t.fit_transform(sample_df)
    assert out["DATA_VOLUME"].isna().sum() == 0
    assert out["ON_NET"].isna().sum() == 0


# ── MedianImputer ─────────────────────────────────────────────────────────────

def test_median_imputer_no_leakage(sample_df: pd.DataFrame) -> None:
    train = sample_df.iloc[:400]
    test  = sample_df.iloc[400:]
    t = MedianImputer()
    t.fit(train)
    out = t.transform(test)
    assert out["MONTANT"].isna().sum() == 0
    # Verify median was computed from train only
    expected_median = float(train["MONTANT"].median())
    assert abs(t.medians_["MONTANT"] - expected_median) < 1e-6


# ── NumericFeatureEngineer ────────────────────────────────────────────────────

def test_numeric_fe_creates_features(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy().fillna(0)
    out = NumericFeatureEngineer().fit_transform(df)
    expected = [
        "regularity_rate", "recharge_per_freq", "revenue_per_freq", "data_per_freq",
        "total_calls", "intl_calls", "n_active_call_types",
        "is_inactive", "has_data", "has_calls", "has_intl_usage",
    ]
    for col in expected:
        assert col in out.columns, f"Missing engineered feature: {col}"


def test_regularity_rate_bounds(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df["REGULARITY"] = df["REGULARITY"].fillna(45)
    out = NumericFeatureEngineer().fit_transform(df)
    assert (out["regularity_rate"] >= 0).all()
    assert (out["regularity_rate"] <= 1.0).all()


def test_intl_calls_equals_zone_sum(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy().fillna(0)
    out = NumericFeatureEngineer().fit_transform(df)
    expected = df["ZONE1"] + df["ZONE2"]
    pd.testing.assert_series_equal(
        out["intl_calls"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_has_intl_usage_consistent_with_intl_calls(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy().fillna(0)
    out = NumericFeatureEngineer().fit_transform(df)
    assert (out.loc[out["intl_calls"] > 0, "has_intl_usage"] == 1).all()
    assert (out.loc[out["intl_calls"] == 0, "has_intl_usage"] == 0).all()


def test_data_per_freq_zero_for_no_data(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy().fillna(0)
    df["DATA_VOLUME"] = 0.0
    out = NumericFeatureEngineer().fit_transform(df)
    assert (out["data_per_freq"] == 0).all()


# ── TenureEncoder ─────────────────────────────────────────────────────────────

def test_tenure_ordinal_order(sample_df: pd.DataFrame) -> None:
    out = TenureEncoder().fit_transform(sample_df)
    assert "tenure_ordinal" in out.columns
    assert "is_new_subscriber" in out.columns
    # A < 2 month should be 0, K > 24 month should be 10
    row_a = sample_df[sample_df["TENURE"] == "A < 2 month"]
    if len(row_a):
        assert out.loc[row_a.index, "tenure_ordinal"].eq(0).all()


# ── MRGEncoder ────────────────────────────────────────────────────────────────

def test_mrg_encoding(sample_df: pd.DataFrame) -> None:
    out = MRGEncoder().fit_transform(sample_df)
    yes_mask = sample_df["MRG"].astype(str).str.upper() == "YES"
    assert out.loc[yes_mask, "mrg_flag"].eq(1).all()
    assert out.loc[~yes_mask, "mrg_flag"].eq(0).all()


# ── TargetEncoder ─────────────────────────────────────────────────────────────

def test_target_encoder_no_leakage(sample_df: pd.DataFrame, target: pd.Series) -> None:
    train = sample_df.iloc[:400]
    test  = sample_df.iloc[400:]
    y_train = target.iloc[:400]
    enc = TargetEncoder(cols=["REGION"])
    enc.fit(train, y_train)
    out = enc.transform(test)
    assert "REGION_te" in out.columns
    # All values should be in [0, 1] (they are probabilities)
    assert (out["REGION_te"] >= 0).all() and (out["REGION_te"] <= 1).all()


# ── FeaturePipeline ───────────────────────────────────────────────────────────

def test_pipeline_fit_transform(sample_df: pd.DataFrame, target: pd.Series) -> None:
    pipe = FeaturePipeline()
    out = pipe.fit_transform(sample_df, target)
    assert len(out) == len(sample_df)
    for col in [
        "regularity_rate", "REGION_te", "top_pack_te",
        "n_services_absent", "is_ghost_subscriber",
        "intl_calls", "has_intl_usage", "data_per_freq",
    ]:
        assert col in out.columns, f"Pipeline output missing: {col}"


def test_pipeline_train_test_consistency(sample_df: pd.DataFrame, target: pd.Series) -> None:
    train = sample_df.iloc[:400].reset_index(drop=True)
    test  = sample_df.iloc[400:].reset_index(drop=True)
    y_train = target.iloc[:400].reset_index(drop=True)

    pipe = FeaturePipeline()
    pipe.fit(train, y_train)
    out_train = pipe.transform(train)
    out_test  = pipe.transform(test)

    assert out_train.columns.tolist() == out_test.columns.tolist()
    assert not out_train.isnull().all(axis=None)


def test_pipeline_uses_configurable_feature_params(
    sample_df: pd.DataFrame,
    target: pd.Series,
) -> None:
    pipe = FeaturePipeline(
        top_pack_min_freq=10,
        target_enc_smoothing=7.0,
        regularity_inactive_threshold=20.0,
    )
    pipe.fit(sample_df, target)

    steps = dict(pipe._steps)
    numeric = steps["numeric_fe"]
    region = steps["region_te"]
    top_pack = steps["top_pack_enc"]

    assert numeric.inactive_threshold == pytest.approx(20.0)
    assert region.smoothing == pytest.approx(7.0)
    assert top_pack.min_freq == 10
    assert top_pack.smoothing == pytest.approx(7.0)
