"""
routes/feed.py
--------------
FastAPI router for the feed optimisation endpoint:

    POST /feed-recommendation
        Analyses current feed intake vs. a target milk yield and returns
        a recommended feed adjustment along with the expected milk increase.

Algorithm
---------
The endpoint uses a simple binary-search approach over feed_intake to find
the optimal feed level that maximises milk yield without over-feeding:

1. Predict baseline milk yield at the current feed intake.
2. If Target_Milk_Yield is not supplied, set target = baseline + 10 %.
3. Search feed values in range [current * 0.5, current * 1.5] (bounded by
   safety limits) and predict milk for each; pick the level closest to target.
4. Return the delta and practical farming advice.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from backend.schemas import FeedRecommendationRequest, FeedRecommendationResponse
from backend.ml_service import ml_service

router = APIRouter(tags=["Feed Optimisation"])

# ── Safety bounds for feed intake ─────────────────────────────────────────────
MIN_FEED_KG = 5.0    # minimum safe daily feed
MAX_FEED_KG = 55.0   # maximum safe daily feed (prevent bloat / acidosis)

# ── Search resolution ─────────────────────────────────────────────────────────
FEED_SEARCH_STEPS = 100   # number of candidate feed levels to evaluate


@router.post(
    "/feed-recommendation",
    response_model=FeedRecommendationResponse,
    summary="Get optimal feed recommendation",
    description=(
        "Given current cow features and an optional target milk yield, "
        "returns the recommended daily feed intake that best achieves the target, "
        "plus the expected milk increase."
    ),
)
def feed_recommendation(req: FeedRecommendationRequest):
    """
    Feed optimisation endpoint.

    - If **Target_Milk_Yield** is omitted the system targets baseline + 10 %.
    - The recommendation is always bounded within safe feed limits.
    - Returns both the feed delta (kg) and the expected milk increase (litres).
    """
    try:
        features = req.model_dump(exclude={"Target_Milk_Yield"})

        # ── Step 1: baseline prediction at current feed ────────────────────
        current_milk = ml_service.predict_milk(features)

        # ── Step 2: determine target ───────────────────────────────────────
        target_milk = req.Target_Milk_Yield or round(current_milk * 1.10, 2)

        # ── Step 3: grid search over candidate feed levels ─────────────────
        current_feed = req.Feed_Intake
        low_feed     = max(MIN_FEED_KG, current_feed * 0.5)
        high_feed    = min(MAX_FEED_KG, current_feed * 1.5)

        candidate_feeds = np.linspace(low_feed, high_feed, FEED_SEARCH_STEPS)

        best_feed  = current_feed
        best_milk  = current_milk
        best_delta = abs(current_milk - target_milk)

        for feed_val in candidate_feeds:
            predicted = ml_service.predict_milk_for_feed(features, float(feed_val))
            delta     = abs(predicted - target_milk)
            if delta < best_delta:
                best_delta = delta
                best_feed  = float(feed_val)
                best_milk  = predicted

        recommended_feed   = round(best_feed, 2)
        expected_milk      = round(best_milk, 2)
        feed_adjustment    = round(recommended_feed - current_feed, 2)
        milk_increase      = round(expected_milk - current_milk, 2)

        # ── Step 4: generate practical advice ─────────────────────────────
        advice = _build_advice(feed_adjustment, milk_increase, req)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feed recommendation error: {exc}")

    return FeedRecommendationResponse(
        current_feed_intake=round(current_feed, 2),
        recommended_feed_intake=recommended_feed,
        feed_adjustment_kg=feed_adjustment,
        current_predicted_milk=current_milk,
        expected_milk_after=expected_milk,
        expected_milk_increase=milk_increase,
        advice=advice,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_advice(
    feed_adj: float,
    milk_inc: float,
    req: FeedRecommendationRequest,
) -> str:
    """Build a human-readable advice string based on the recommendation."""

    lines = []

    if feed_adj > 2.0:
        lines.append(
            f"📈  Increase daily feed by {feed_adj:.1f} kg to boost milk yield "
            f"by ~{milk_inc:.1f} litres/day."
        )
        lines.append(
            "    Gradually introduce the change over 3–5 days to avoid "
            "digestive upset."
        )
    elif feed_adj < -2.0:
        lines.append(
            f"📉  Reduce daily feed by {abs(feed_adj):.1f} kg. "
            "Current feed level may be in excess of metabolic needs."
        )
        lines.append(
            "    Ensure the cow still meets minimum dry matter intake requirements."
        )
    else:
        lines.append(
            "✅  Current feed intake is close to optimal. "
            f"Minor adjustment of {feed_adj:+.1f} kg may yield a small improvement."
        )

    # Extra contextual tips
    if req.Temperature > 30.0:
        lines.append(
            "🌡️  High temperature detected – provide shade, extra water, and "
            "consider total mixed ration (TMR) to counteract heat stress."
        )
    if req.Pregnant:
        lines.append(
            "🐄  Pregnant cow detected – ensure adequate calcium and energy "
            "supplements in the transition period (3 weeks pre/post calving)."
        )
    if req.Activity < 4.0:
        lines.append(
            "⚠️  Low activity score – investigate for lameness or illness "
            "before increasing feed levels."
        )

    return "  ".join(lines)
