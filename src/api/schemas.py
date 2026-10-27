"""Pydantic request/response schemas for the churn prediction API."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Usage, revenue, and frequency columns are counts or amounts — never negative.
NonNegFloat = Annotated[float, Field(ge=0)]

TenureBand = Literal[
    "A < 2 month",
    "B 2 month",
    "C 3 month",
    "D 3-6 month",
    "E 6-9 month",
    "F 9-12 month",
    "G 12-15 month",
    "H 15-18 month",
    "I 18-21 month",
    "J 21-24 month",
    "K > 24 month",
]


class SubscriberFeatures(BaseModel):
    """Raw subscriber features matching the Expresso dataset schema.

    All usage columns are optional — pass null when a subscriber has no
    recorded activity for that channel (treated as MNAR by the pipeline).
    Unknown field names are rejected (422) rather than silently ignored, so a
    misspelled column cannot turn into a missing value.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "subscriber_id": "sub-000123",
                "REGION": "DAKAR",
                "TENURE": "K > 24 month",
                "MRG": "NO",
                "TOP_PACK": "On net 200F=Unlimited _call24H",
                "MONTANT": 4250.0,
                "FREQUENCE_RECH": 15.0,
                "REVENUE": 4251.0,
                "ARPU_SEGMENT": 1417.0,
                "FREQUENCE": 17.0,
                "DATA_VOLUME": 4.0,
                "ON_NET": 388.0,
                "ORANGE": 46.0,
                "TIGO": 1.0,
                "ZONE1": 1.0,
                "ZONE2": 2.0,
                "REGULARITY": 54.0,
                "FREQ_TOP_PACK": 8.0,
            }
        },
    )

    subscriber_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Caller's subscriber identifier. Not a model input; stored with the "
            "prediction so churn labels can later be joined back to it."
        ),
    )
    REGION: str | None = Field(default=None, max_length=64)
    TENURE: TenureBand | None = None
    MRG: Literal["YES", "NO"] | None = None
    TOP_PACK: str | None = Field(default=None, max_length=128)
    MONTANT: NonNegFloat | None = None
    FREQUENCE_RECH: NonNegFloat | None = None
    REVENUE: NonNegFloat | None = None
    ARPU_SEGMENT: NonNegFloat | None = None
    FREQUENCE: NonNegFloat | None = None
    DATA_VOLUME: NonNegFloat | None = None
    ON_NET: NonNegFloat | None = None
    ORANGE: NonNegFloat | None = None
    TIGO: NonNegFloat | None = None
    ZONE1: NonNegFloat | None = None
    ZONE2: NonNegFloat | None = None
    REGULARITY: float | None = Field(default=None, ge=0, le=90)
    FREQ_TOP_PACK: NonNegFloat | None = None

    def model_inputs(self) -> dict[str, object]:
        """The raw feature columns only (no identifiers)."""
        return self.model_dump(exclude={"subscriber_id"})


class PredictionRequest(BaseModel):
    """Batch prediction request — up to 10 000 subscribers per call."""

    model_config = ConfigDict(extra="forbid")

    subscribers: list[SubscriberFeatures] = Field(..., min_length=1, max_length=10_000)
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Probability threshold for the binary churn_prediction label. Defaults "
            "to the model's F1-optimal threshold, chosen on out-of-fold training "
            "predictions."
        ),
    )


class SubscriberPrediction(BaseModel):
    subscriber_id: str | None = None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool


class PredictionResponse(BaseModel):
    request_id: str
    predictions: list[SubscriberPrediction]
    model_name: str
    threshold: float
    n_subscribers: int


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_name: str
    dataset: str
    calibration_method: str | None
    default_threshold: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_pr_auc_mean: float
    n_features: int


class LogsSummary(BaseModel):
    # All-time counters
    total_predictions: int
    total_requests: int
    # Everything below is restricted to the last `window_hours`
    window_hours: int
    predictions_in_window: int
    requests_in_window: int
    mean_churn_probability: float | None
    churn_flag_rate: float | None
    mean_latency_ms: float | None


class RecentPrediction(BaseModel):
    request_id: str
    timestamp: str
    model_name: str | None
    threshold: float
    latency_ms: float | None
    subscriber_idx: int
    subscriber_id: str | None
    churn_probability: float
    churn_prediction: bool


class LogsResponse(BaseModel):
    summary: LogsSummary
    recent: list[RecentPrediction]


class FeatureDriftResult(BaseModel):
    feature: str
    kind: str     # "continuous" | "discrete"
    psi: float
    ks_statistic: float | None   # continuous features only
    ks_pvalue: float | None      # reported for reference; not used for status
    status: str   # "OK" | "WARN" | "DRIFT"
    n_live: int


class DriftResponse(BaseModel):
    report_time: str
    window_hours: int
    n_live_predictions: int
    n_features_checked: int
    n_drifted: int
    n_warned: int
    overall_status: str   # "OK" | "WARN" | "DRIFT" | "INSUFFICIENT_DATA"
    features: list[FeatureDriftResult]
    # Drift of the served churn_probability itself vs. its holdout distribution.
    prediction_drift: FeatureDriftResult | None = None
