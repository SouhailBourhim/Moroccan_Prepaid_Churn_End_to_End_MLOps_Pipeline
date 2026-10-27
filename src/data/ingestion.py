"""Raw data loading with dtypes and basic validation."""
from pathlib import Path
from typing import Any

import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "expresso"
MOROCCAN_FILE = Path(__file__).parents[2] / "data" / "raw" / "moroccan_telecom_churn.csv"

DTYPES: dict[str, str] = {
    "user_id": "string",
    "REGION": "category",
    "TENURE": "category",
    "MRG": "category",
    "TOP_PACK": "category",
    "MONTANT": "float32",
    "FREQUENCE_RECH": "float32",
    "REVENUE": "float32",
    "ARPU_SEGMENT": "float32",
    "FREQUENCE": "float32",
    "DATA_VOLUME": "float32",
    "ON_NET": "float32",
    "ORANGE": "float32",
    "TIGO": "float32",
    "ZONE1": "float32",
    "ZONE2": "float32",
    "REGULARITY": "float32",
    "FREQ_TOP_PACK": "float32",
}

CAT_COLS: frozenset[str] = frozenset(c for c, t in DTYPES.items() if t == "category")
FLOAT_COLS: frozenset[str] = frozenset(c for c, t in DTYPES.items() if t == "float32")

# Moroccan dataset has the same schema + OPERATOR column
_MOROCCAN_EXTRA: dict[str, str] = {"OPERATOR": "category"}


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a raw-schema DataFrame from JSON-like records (API requests, prediction log).

    Applies the same dtypes as the CSV loaders so the fitted FeaturePipeline sees
    identical input types at serving time and at training time.
    """
    df = pd.DataFrame(records)
    for col in CAT_COLS & set(df.columns):
        df[col] = df[col].astype("category")
    for col in FLOAT_COLS & set(df.columns):
        df[col] = df[col].astype("float32")
    return df


def load_train(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    p = path or RAW_DIR / "Train.csv"
    df = pd.read_csv(p, dtype=DTYPES, nrows=nrows, low_memory=False)
    df["CHURN"] = df["CHURN"].astype("int8")
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    p = path or RAW_DIR / "Test.csv"
    return pd.read_csv(p, dtype=DTYPES, low_memory=False)


def load_moroccan(
    path: Path | str | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Load the synthetic Moroccan prepaid churn dataset.

    Same schema as Expresso Train.csv plus an OPERATOR column
    (Inwi / Maroc_Telecom / Orange_Maroc).  CHURN is always present.
    """
    p = Path(path) if path is not None else MOROCCAN_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"Moroccan dataset not found at {p}.\n"
            "Generate it first:  python generate_moroccan_dataset.py"
        )
    dtype_full = {**DTYPES, **_MOROCCAN_EXTRA}
    df = pd.read_csv(p, dtype=dtype_full, nrows=nrows, low_memory=False)
    df["CHURN"] = df["CHURN"].astype("int8")
    return df


def load_variable_definitions() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "VariableDefinitions.csv", header=1)
