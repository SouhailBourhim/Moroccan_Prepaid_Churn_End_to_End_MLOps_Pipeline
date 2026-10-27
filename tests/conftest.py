"""Shared test fixtures and synthetic raw-data helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TENURE_ORDER


@pytest.fixture(autouse=True)
def _dev_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run API tests in explicit dev mode; auth tests opt back in."""
    monkeypatch.setenv("CHURN_AUTH_DISABLED", "1")
    monkeypatch.delenv("CHURN_API_KEYS", raising=False)


def make_raw_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Raw Expresso-schema rows with a learnable churn signal.

    Churn probability falls with REGULARITY and rises for REGION 'THIES' and for
    subscribers with no recorded calls, so models and encoders have something
    real to pick up.
    """
    rng = np.random.default_rng(seed)
    regularity = rng.integers(1, 91, n).astype(float)
    region = rng.choice(["DAKAR", "THIES", "SAINT-LOUIS", None], n)
    on_net = np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0, 500, n))
    logit = 1.5 - regularity / 20 + (region == "THIES") * 1.0 + np.isnan(on_net) * 0.8
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")

    def maybe_nan(values: np.ndarray, rate: float) -> np.ndarray:
        return np.where(rng.random(n) < rate, np.nan, values)

    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "REGION": region,
        "TENURE": rng.choice(list(TENURE_ORDER), n),
        "MONTANT": maybe_nan(rng.uniform(100, 10_000, n), 0.3),
        "FREQUENCE_RECH": maybe_nan(rng.uniform(1, 30, n), 0.3),
        "REVENUE": maybe_nan(rng.uniform(100, 20_000, n), 0.3),
        "ARPU_SEGMENT": maybe_nan(rng.uniform(30, 7_000, n), 0.3),
        "FREQUENCE": maybe_nan(rng.uniform(1, 50, n), 0.3),
        "DATA_VOLUME": maybe_nan(rng.uniform(0, 5_000, n), 0.4),
        "ON_NET": on_net,
        "ORANGE": maybe_nan(rng.uniform(0, 200, n), 0.4),
        "TIGO": maybe_nan(rng.uniform(0, 100, n), 0.5),
        "ZONE1": maybe_nan(rng.uniform(0, 50, n), 0.8),
        "ZONE2": maybe_nan(rng.uniform(0, 20, n), 0.8),
        "MRG": "NO",
        "REGULARITY": regularity,
        "TOP_PACK": rng.choice(["PackA", "PackB", "PackC", None], n),
        "FREQ_TOP_PACK": maybe_nan(rng.uniform(1, 20, n), 0.4),
        "CHURN": churn,
    })
    for col in ("REGION", "TENURE", "MRG", "TOP_PACK"):
        df[col] = df[col].astype("category")
    return df
