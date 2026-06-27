"""
schemas.py
----------
Pydantic request / response models for the DairyVision AI API.

These schemas define:
    - The JSON body accepted by each POST endpoint
    - The JSON body returned by each endpoint (response models)

Using Pydantic v2-compatible syntax (FastAPI >= 0.100).
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Shared literals ───────────────────────────────────────────────────────────

BreedType = Literal[
    "Holstein", "Jersey", "Brown Swiss", "Guernsey", "Ayrshire"
]

HealthType = Literal["Healthy", "Unhealthy"]


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST BODIES
# ═════════════════════════════════════════════════════════════════════════════

class CowFeatures(BaseModel):
    """
    Core cow features required for both milk prediction and health prediction.
    All numeric values are validated to be within realistic farm ranges.
    """

    Breed:              BreedType = Field(..., description="Cow breed")
    Age:                float     = Field(..., ge=0.5,  le=20.0,  description="Age in years")
    Weight:             float     = Field(..., ge=200,  le=1200,  description="Body weight in kg")
    Feed_Intake:        float     = Field(..., ge=1.0,  le=60.0,  description="Daily feed intake in kg")
    Water_Intake:       float     = Field(..., ge=10.0, le=200.0, description="Daily water intake in litres")
    Temperature:        float     = Field(..., ge=0.0,  le=50.0,  description="Ambient temperature °C")
    Humidity:           float     = Field(..., ge=0.0,  le=100.0, description="Relative humidity %")
    Activity:           float     = Field(..., ge=1.0,  le=10.0,  description="Activity score 1–10")
    Pregnant:           int       = Field(..., ge=0,    le=1,     description="1 = pregnant, 0 = not")
    Previous_Milk_Yield: float    = Field(..., ge=0.0,  le=80.0,  description="Yesterday's milk yield (litres)")

    model_config = {"json_schema_extra": {
        "example": {
            "Breed": "Holstein",
            "Age": 4.5,
            "Weight": 650,
            "Feed_Intake": 22.0,
            "Water_Intake": 85.0,
            "Temperature": 22.0,
            "Humidity": 60.0,
            "Activity": 7.5,
            "Pregnant": 0,
            "Previous_Milk_Yield": 28.0,
        }
    }}


class FeedRecommendationRequest(BaseModel):
    """
    Request body for the /feed-recommendation endpoint.
    Extends CowFeatures with an optional target milk yield.
    """

    Breed:              BreedType = Field(..., description="Cow breed")
    Age:                float     = Field(..., ge=0.5,  le=20.0)
    Weight:             float     = Field(..., ge=200,  le=1200)
    Feed_Intake:        float     = Field(..., ge=1.0,  le=60.0)
    Water_Intake:       float     = Field(..., ge=10.0, le=200.0)
    Temperature:        float     = Field(..., ge=0.0,  le=50.0)
    Humidity:           float     = Field(..., ge=0.0,  le=100.0)
    Activity:           float     = Field(..., ge=1.0,  le=10.0)
    Pregnant:           int       = Field(..., ge=0,    le=1)
    Previous_Milk_Yield: float    = Field(..., ge=0.0,  le=80.0)
    Target_Milk_Yield:  Optional[float] = Field(
        None, ge=0.0, le=80.0,
        description="Optional target yield (litres); if omitted the system auto-sets +10%"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "Breed": "Holstein",
            "Age": 4.5,
            "Weight": 650,
            "Feed_Intake": 20.0,
            "Water_Intake": 80.0,
            "Temperature": 24.0,
            "Humidity": 65.0,
            "Activity": 6.0,
            "Pregnant": 0,
            "Previous_Milk_Yield": 25.0,
            "Target_Milk_Yield": 28.0,
        }
    }}


# ═════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═════════════════════════════════════════════════════════════════════════════

class StatusResponse(BaseModel):
    """Response for GET /"""
    status:  str
    message: str
    version: str


class MilkPredictionResponse(BaseModel):
    """Response for POST /predict-milk"""
    predicted_milk_yield: float = Field(..., description="Predicted litres tomorrow")
    unit:                 str   = Field("litres/day")
    confidence_note:      str


class HealthPredictionResponse(BaseModel):
    """Response for POST /predict-health"""
    health_status:    HealthType
    confidence:       float = Field(..., description="Model confidence score 0–1")
    recommendation:   str   = Field(..., description="Action recommendation based on health")


class FeedRecommendationResponse(BaseModel):
    """Response for POST /feed-recommendation"""
    current_feed_intake:    float
    recommended_feed_intake: float
    feed_adjustment_kg:     float  = Field(..., description="Positive = increase, negative = decrease")
    current_predicted_milk:  float
    expected_milk_after:     float
    expected_milk_increase:  float
    advice:                  str


class DashboardSummaryResponse(BaseModel):
    """Response for GET /dashboard-summary"""
    total_cows:          int
    total_milk_today:    float = Field(..., description="Sum of Previous_Milk_Yield (litres)")
    average_milk:        float
    healthy_cows:        int
    unhealthy_cows:      int
    average_temperature: float
    average_feed_intake: float
    health_percentage:   float = Field(..., description="% of cows that are healthy")


class MilkTrendPoint(BaseModel):
    """A single data point in the milk trend series."""
    label:      str
    milk_yield: float


class MilkTrendResponse(BaseModel):
    """Response for GET /milk-trend"""
    trend: List[MilkTrendPoint]


class HealthDistributionResponse(BaseModel):
    """Response for GET /health-distribution"""
    Healthy:   int
    Unhealthy: int
    total:     int


class FeedVsMilkPoint(BaseModel):
    """A single {feed_intake, milk_yield} pair for scatter / line charts."""
    Cow_ID:      str
    Feed_Intake: float
    Milk_Yield:  float


class FeedVsMilkResponse(BaseModel):
    """Response for GET /feed-vs-milk"""
    data: List[FeedVsMilkPoint]
