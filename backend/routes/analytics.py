"""
routes/analytics.py
--------------------
FastAPI router for dashboard and chart analytics endpoints:

    GET /dashboard-summary      – KPI summary card data
    GET /milk-trend             – milk production history per breed
    GET /health-distribution    – healthy vs. unhealthy counts
    GET /feed-vs-milk           – scatter chart data (feed intake vs. milk yield)

All endpoints read directly from the CSV dataset so the frontend always
reflects the real data without needing a separate database.
"""

from __future__ import annotations

import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.schemas import (
    DashboardSummaryResponse,
    MilkTrendResponse,
    MilkTrendPoint,
    HealthDistributionResponse,
    FeedVsMilkResponse,
    FeedVsMilkPoint,
)

router = APIRouter(tags=["Analytics"])

# ── Dataset path ──────────────────────────────────────────────────────────────
# routes/analytics.py lives at backend/routes/analytics.py
# backend/  →  dirname of dirname of this file
_ROUTES_DIR  = os.path.dirname(os.path.abspath(__file__))   # backend/routes/
_BACKEND_DIR = os.path.dirname(_ROUTES_DIR)                 # backend/
DATASET_PATH = os.path.join(_BACKEND_DIR, "dataset", "dataset.csv")


def _load_df() -> pd.DataFrame:
    """
    Load the CSV dataset.  Raises HTTP 503 if the file is missing so the
    frontend can display a helpful error rather than a generic 500.
    """
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                "Dataset file not found.  "
                "Run `python backend/dataset/generate_dataset.py` to create it."
            ),
        )
    return pd.read_csv(DATASET_PATH)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/dashboard-summary",
    response_model=DashboardSummaryResponse,
    summary="Farm KPI summary",
    description=(
        "Returns high-level key performance indicators for the dashboard: "
        "total cows, milk production, health breakdown, temperature, and feed."
    ),
)
def dashboard_summary():
    """
    Aggregated KPI summary consumed by the React dashboard header cards.

    Metrics
    -------
    - total_cows          : number of records in the dataset
    - total_milk_today    : sum of Previous_Milk_Yield (today's production)
    - average_milk        : mean milk per cow
    - healthy_cows        : count of Health_Status == "Healthy"
    - unhealthy_cows      : count of Health_Status == "Unhealthy"
    - average_temperature : mean ambient temperature across the herd
    - average_feed_intake : mean daily feed per cow
    - health_percentage   : healthy_cows / total_cows × 100
    """
    df = _load_df()

    total_cows       = len(df)
    total_milk       = round(float(df["Previous_Milk_Yield"].sum()), 2)
    avg_milk         = round(float(df["Previous_Milk_Yield"].mean()), 2)
    healthy_count    = int((df["Health_Status"] == "Healthy").sum())
    unhealthy_count  = int((df["Health_Status"] == "Unhealthy").sum())
    avg_temp         = round(float(df["Temperature"].mean()), 2)
    avg_feed         = round(float(df["Feed_Intake"].mean()), 2)
    health_pct       = round((healthy_count / total_cows) * 100, 1)

    return DashboardSummaryResponse(
        total_cows=total_cows,
        total_milk_today=total_milk,
        average_milk=avg_milk,
        healthy_cows=healthy_count,
        unhealthy_cows=unhealthy_count,
        average_temperature=avg_temp,
        average_feed_intake=avg_feed,
        health_percentage=health_pct,
    )


@router.get(
    "/milk-trend",
    response_model=MilkTrendResponse,
    summary="Milk production trend by breed",
    description=(
        "Returns average milk yield grouped by breed, suitable for "
        "rendering a bar or line chart on the frontend."
    ),
)
def milk_trend():
    """
    Milk production trend endpoint.

    Groups cows by Breed and returns the mean Previous_Milk_Yield for each,
    giving the React frontend data to render a breed-level trend chart.
    The data is sorted in descending yield order so the highest-producing
    breeds appear first.
    """
    df = _load_df()

    trend_df = (
        df.groupby("Breed")["Previous_Milk_Yield"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )

    trend_points = [
        MilkTrendPoint(label=row["Breed"], milk_yield=row["Previous_Milk_Yield"])
        for _, row in trend_df.iterrows()
    ]

    return MilkTrendResponse(trend=trend_points)


@router.get(
    "/health-distribution",
    response_model=HealthDistributionResponse,
    summary="Healthy vs. unhealthy cow count",
    description=(
        "Returns the count of Healthy and Unhealthy cows. "
        "Suitable for rendering a pie or donut chart."
    ),
)
def health_distribution():
    """
    Health distribution endpoint.

    Returns total counts for each health category.  Used by the React
    frontend to render a pie/donut chart on the dashboard.
    """
    df = _load_df()

    counts     = df["Health_Status"].value_counts()
    healthy    = int(counts.get("Healthy", 0))
    unhealthy  = int(counts.get("Unhealthy", 0))

    return HealthDistributionResponse(
        Healthy=healthy,
        Unhealthy=unhealthy,
        total=healthy + unhealthy,
    )


@router.get(
    "/feed-vs-milk",
    response_model=FeedVsMilkResponse,
    summary="Feed intake vs. milk yield data",
    description=(
        "Returns per-cow feed intake and milk yield pairs for scatter plot "
        "visualisation.  Optionally limited to the first 100 cows for "
        "performance."
    ),
)
def feed_vs_milk(limit: int = 100):
    """
    Feed vs. milk scatter chart data endpoint.

    Parameters
    ----------
    limit : int  – maximum number of data points to return (default 100)

    Returns one data point per cow with Cow_ID, Feed_Intake, and
    Previous_Milk_Yield.  The React frontend renders this as a scatter plot
    to visualise the correlation between feeding and production.
    """
    df = _load_df()

    # Cap at the dataset size and the requested limit
    sample = df[["Cow_ID", "Feed_Intake", "Previous_Milk_Yield"]].head(
        min(limit, len(df))
    )

    data_points = [
        FeedVsMilkPoint(
            Cow_ID=row["Cow_ID"],
            Feed_Intake=round(float(row["Feed_Intake"]), 2),
            Milk_Yield=round(float(row["Previous_Milk_Yield"]), 2),
        )
        for _, row in sample.iterrows()
    ]

    return FeedVsMilkResponse(data=data_points)
