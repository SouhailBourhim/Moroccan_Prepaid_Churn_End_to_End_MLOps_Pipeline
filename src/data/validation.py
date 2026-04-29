"""Schema and domain-rule validation for raw Expresso data."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


REQUIRED_COLS = [
    "user_id", "REGION", "TENURE", "MONTANT", "FREQUENCE_RECH",
    "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME",
    "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2", "MRG",
    "REGULARITY", "TOP_PACK", "FREQ_TOP_PACK",
]

NON_NEGATIVE_COLS = [
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT",
    "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
    "ZONE1", "ZONE2", "REGULARITY", "FREQ_TOP_PACK",
]


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> dict[str, Any]:
        return {"passed": self.passed, "errors": self.errors, "warnings": self.warnings}


def validate_raw(df: pd.DataFrame, is_train: bool = True) -> ValidationReport:
    report = ValidationReport()

    # Column presence
    cols = set(df.columns)
    required = set(REQUIRED_COLS + (["CHURN"] if is_train else []))
    missing_cols = required - cols
    if missing_cols:
        report.fail(f"Missing required columns: {missing_cols}")

    # No duplicate user_ids
    dupes = int(df["user_id"].duplicated().sum())
    if dupes:
        report.warn(f"{dupes:,} duplicate user_ids found")

    # Non-negative numeric checks
    for col in NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            report.warn(f"{col}: {n_neg:,} negative values")

    # REGULARITY domain bounds [0, 90]
    if "REGULARITY" in df.columns:
        out = int(((df["REGULARITY"] < 0) | (df["REGULARITY"] > 90)).sum())
        if out:
            report.warn(f"REGULARITY: {out:,} values outside [0, 90]")

    # CHURN binary
    if is_train and "CHURN" in df.columns:
        unexpected = set(df["CHURN"].unique()) - {0, 1}
        if unexpected:
            report.fail(f"CHURN contains unexpected values: {unexpected}")

    return report
