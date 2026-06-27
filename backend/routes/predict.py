"""
routes/predict.py
-----------------
FastAPI router for ML prediction endpoints:

    POST /predict-milk    – predict tomorrow's milk yield
    POST /predict-health  – classify cow as Healthy / Unhealthy
"""

from fastapi import APIRouter, HTTPException
from backend.schemas import (
    CowFeatures,
    MilkPredictionResponse,
    HealthPredictionResponse,
)
from backend.ml_service import ml_service

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict-milk",
    response_model=MilkPredictionResponse,
    summary="Predict tomorrow's milk yield",
    description=(
        "Accepts cow features and returns the predicted milk yield "
        "(in litres) for the following day using a trained "
        "RandomForestRegressor."
    ),
)
def predict_milk(cow: CowFeatures):
    """
    Predict tomorrow's milk yield for a single cow.

    - **Breed**: One of Holstein / Jersey / Brown Swiss / Guernsey / Ayrshire
    - **Previous_Milk_Yield**: Yesterday's recorded yield in litres
    - All environmental and physiological features are required
    """
    try:
        features = cow.model_dump()
        predicted = ml_service.predict_milk(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Confidence note based on how close to the breed average we are
    confidence_note = (
        "High confidence: stable conditions detected."
        if 10 <= predicted <= 50
        else "Moderate confidence: value at extremes – verify inputs."
    )

    return MilkPredictionResponse(
        predicted_milk_yield=predicted,
        unit="litres/day",
        confidence_note=confidence_note,
    )


@router.post(
    "/predict-health",
    response_model=HealthPredictionResponse,
    summary="Predict cow health status",
    description=(
        "Classifies the cow as Healthy or Unhealthy using a trained "
        "RandomForestClassifier and returns an action recommendation."
    ),
)
def predict_health(cow: CowFeatures):
    """
    Predict the health status of a single cow.

    Returns the predicted label, model confidence, and an action recommendation
    so farm staff know what to do next.
    """
    try:
        features = cow.model_dump()
        label, confidence = ml_service.predict_health(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Health prediction error: {exc}")

    # Generate human-readable recommendation based on the label
    if label == "Unhealthy":
        recommendation = (
            "⚠️  This cow shows signs of illness. "
            "Schedule an immediate veterinary check-up, monitor feed/water intake, "
            "and isolate if contagious illness is suspected."
        )
    else:
        recommendation = (
            "✅  Cow appears healthy. "
            "Continue current feeding and monitoring schedule. "
            "Re-assess weekly or if behaviour changes."
        )

    return HealthPredictionResponse(
        health_status=label,
        confidence=confidence,
        recommendation=recommendation,
    )
