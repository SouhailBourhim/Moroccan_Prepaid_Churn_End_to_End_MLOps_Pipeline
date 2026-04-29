"""Raw data loading with dtypes and basic validation."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "expresso"

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


def load_train(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    p = path or RAW_DIR / "Train.csv"
    df = pd.read_csv(p, dtype=DTYPES, nrows=nrows, low_memory=False)
    df["CHURN"] = df["CHURN"].astype("int8")
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    p = path or RAW_DIR / "Test.csv"
    return pd.read_csv(p, dtype=DTYPES, low_memory=False)


def load_variable_definitions() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "VariableDefinitions.csv", header=1)
