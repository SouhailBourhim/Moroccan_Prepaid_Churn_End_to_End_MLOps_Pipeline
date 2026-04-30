"""Pydantic request/response schemas for the churn prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SubscriberFeatures(BaseModel):
    """Raw subscriber features matching the Expresso dataset schema.

    All usage columns are optional — pass None when a subscriber has no
    recorded activity for that channel (treated as MNAR by the pipeline).
    REGULARITY is the only field with 0% missingness in training data;
    it is strongly recommended but also accepts None for robustness.
    """

    REGION: str | None = None
    TENURE: str | None = None
    MRG: str | None = None
    TOP_PACK: str | None = None
    MONTANT: float | None = None
    FREQUENCE_RECH: float | None = None
    REVENUE: float | None = None
    ARPU_SEGMENT: float | None = None
    FREQUENCE: float | None = None
    DATA_VOLUME: float | None = None
    ON_NET: float | None = None
    ORANGE: float | None = None
    TIGO: float | None = None
    ZONE1: float | None = None
    ZONE2: float | None = None
    REGULARITY: float | None = Field(default=None, ge=0, le=90)
    FREQ_TOP_PACK: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionRequest(BaseModel):
    """Batch prediction request — up to 10 000 subscribers per call."""

    subscribers: list[SubscriberFeatures] = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for the binary churn_prediction label.",
    )


class SubscriberPrediction(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool


class PredictionResponse(BaseModel):
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
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_pr_auc_mean: float
    n_features: int
