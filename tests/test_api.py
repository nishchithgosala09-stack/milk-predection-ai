"""
tests/test_api.py
-----------------
Integration tests for the DairyVision AI FastAPI backend.

These tests use FastAPI's TestClient (built on httpx) to exercise every
endpoint end-to-end — including request validation, response shapes, and
HTTP status codes.

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from main import app

# ── TestClient (synchronous) ──────────────────────────────────────────────────
client = TestClient(app)

# ── Reusable valid cow payload ─────────────────────────────────────────────────
VALID_COW = {
    "Breed": "Holstein",
    "Age": 4.5,
    "Weight": 650.0,
    "Feed_Intake": 22.0,
    "Water_Intake": 85.0,
    "Temperature": 22.0,
    "Humidity": 60.0,
    "Activity": 7.5,
    "Pregnant": 0,
    "Previous_Milk_Yield": 28.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Status Endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestRootEndpoint:
    def test_get_root_returns_200(self):
        """GET / should return 200 OK with running status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_root_response_shape(self):
        """Response must contain status, message, and version fields."""
        data = client.get("/").json()
        assert "status"  in data
        assert "message" in data
        assert "version" in data

    def test_get_root_status_is_running(self):
        """Status value should be 'running'."""
        data = client.get("/").json()
        assert data["status"] == "running"


# ══════════════════════════════════════════════════════════════════════════════
# Milk Prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictMilk:
    def test_predict_milk_returns_200(self):
        """POST /predict-milk should return 200 with a valid payload."""
        response = client.post("/predict-milk", json=VALID_COW)
        assert response.status_code == 200

    def test_predict_milk_response_shape(self):
        """Response must contain predicted_milk_yield, unit, confidence_note."""
        data = client.post("/predict-milk", json=VALID_COW).json()
        assert "predicted_milk_yield" in data
        assert "unit"                 in data
        assert "confidence_note"      in data

    def test_predict_milk_yield_is_positive(self):
        """Predicted milk yield should always be a positive number."""
        data = client.post("/predict-milk", json=VALID_COW).json()
        assert data["predicted_milk_yield"] >= 0

    def test_predict_milk_unit_is_litres(self):
        """Unit field should be 'litres/day'."""
        data = client.post("/predict-milk", json=VALID_COW).json()
        assert data["unit"] == "litres/day"

    def test_predict_milk_invalid_breed_returns_422(self):
        """Sending an unknown breed should return 422 Unprocessable Entity."""
        payload = {**VALID_COW, "Breed": "Unicorn"}
        response = client.post("/predict-milk", json=payload)
        assert response.status_code == 422

    def test_predict_milk_missing_field_returns_422(self):
        """Missing a required field should return 422."""
        payload = {k: v for k, v in VALID_COW.items() if k != "Age"}
        response = client.post("/predict-milk", json=payload)
        assert response.status_code == 422

    def test_predict_milk_pregnant_cow(self):
        """Pregnant cow (Pregnant=1) should still return a valid yield."""
        payload = {**VALID_COW, "Pregnant": 1}
        response = client.post("/predict-milk", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_milk_yield"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Health Prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictHealth:
    def test_predict_health_returns_200(self):
        """POST /predict-health should return 200 OK."""
        response = client.post("/predict-health", json=VALID_COW)
        assert response.status_code == 200

    def test_predict_health_response_shape(self):
        """Response must contain health_status, confidence, recommendation."""
        data = client.post("/predict-health", json=VALID_COW).json()
        assert "health_status"   in data
        assert "confidence"      in data
        assert "recommendation"  in data

    def test_predict_health_label_is_valid(self):
        """Health status must be either 'Healthy' or 'Unhealthy'."""
        data = client.post("/predict-health", json=VALID_COW).json()
        assert data["health_status"] in ("Healthy", "Unhealthy")

    def test_predict_health_confidence_range(self):
        """Confidence must be between 0 and 1 (inclusive)."""
        data = client.post("/predict-health", json=VALID_COW).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_health_unhealthy_cow(self):
        """A cow with extreme stress indicators should return a valid response."""
        payload = {**VALID_COW, "Temperature": 39.0, "Activity": 1.5, "Previous_Milk_Yield": 5.0}
        response = client.post("/predict-health", json=payload)
        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# Feed Recommendation
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedRecommendation:
    def test_feed_recommendation_returns_200(self):
        """POST /feed-recommendation should return 200."""
        response = client.post("/feed-recommendation", json=VALID_COW)
        assert response.status_code == 200

    def test_feed_recommendation_response_shape(self):
        """Response must contain all expected fields."""
        data = client.post("/feed-recommendation", json=VALID_COW).json()
        required_fields = [
            "current_feed_intake",
            "recommended_feed_intake",
            "feed_adjustment_kg",
            "current_predicted_milk",
            "expected_milk_after",
            "expected_milk_increase",
            "advice",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_feed_recommendation_with_target(self):
        """Should accept and use an explicit Target_Milk_Yield."""
        payload = {**VALID_COW, "Target_Milk_Yield": 30.0}
        response = client.post("/feed-recommendation", json=payload)
        assert response.status_code == 200

    def test_feed_recommendation_bounds_respected(self):
        """Recommended feed must stay within MIN and MAX safe bounds."""
        data = client.post("/feed-recommendation", json=VALID_COW).json()
        assert 5.0 <= data["recommended_feed_intake"] <= 55.0

    def test_feed_recommendation_advice_is_string(self):
        """Advice field must be a non-empty string."""
        data = client.post("/feed-recommendation", json=VALID_COW).json()
        assert isinstance(data["advice"], str)
        assert len(data["advice"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard Summary
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardSummary:
    def test_dashboard_summary_returns_200(self):
        """GET /dashboard-summary should return 200."""
        response = client.get("/dashboard-summary")
        assert response.status_code == 200

    def test_dashboard_summary_response_shape(self):
        """Response must contain all KPI fields."""
        data = client.get("/dashboard-summary").json()
        required = [
            "total_cows", "total_milk_today", "average_milk",
            "healthy_cows", "unhealthy_cows",
            "average_temperature", "average_feed_intake", "health_percentage",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_dashboard_cows_count_positive(self):
        """Total cows must be a positive integer."""
        data = client.get("/dashboard-summary").json()
        assert data["total_cows"] > 0

    def test_dashboard_health_percentage_valid(self):
        """Health percentage must be between 0 and 100."""
        data = client.get("/dashboard-summary").json()
        assert 0.0 <= data["health_percentage"] <= 100.0

    def test_dashboard_counts_add_up(self):
        """Healthy + Unhealthy must equal total cows."""
        data = client.get("/dashboard-summary").json()
        assert data["healthy_cows"] + data["unhealthy_cows"] == data["total_cows"]


# ══════════════════════════════════════════════════════════════════════════════
# Milk Trend
# ══════════════════════════════════════════════════════════════════════════════

class TestMilkTrend:
    def test_milk_trend_returns_200(self):
        response = client.get("/milk-trend")
        assert response.status_code == 200

    def test_milk_trend_has_trend_list(self):
        data = client.get("/milk-trend").json()
        assert "trend" in data
        assert isinstance(data["trend"], list)

    def test_milk_trend_points_have_label_and_yield(self):
        data = client.get("/milk-trend").json()
        for point in data["trend"]:
            assert "label"      in point
            assert "milk_yield" in point
            assert point["milk_yield"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Health Distribution
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthDistribution:
    def test_health_distribution_returns_200(self):
        response = client.get("/health-distribution")
        assert response.status_code == 200

    def test_health_distribution_response_shape(self):
        data = client.get("/health-distribution").json()
        assert "Healthy"   in data
        assert "Unhealthy" in data
        assert "total"     in data

    def test_health_distribution_total_correct(self):
        data = client.get("/health-distribution").json()
        assert data["Healthy"] + data["Unhealthy"] == data["total"]


# ══════════════════════════════════════════════════════════════════════════════
# Feed vs. Milk
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedVsMilk:
    def test_feed_vs_milk_returns_200(self):
        response = client.get("/feed-vs-milk")
        assert response.status_code == 200

    def test_feed_vs_milk_default_limit(self):
        """Default limit is 100 data points."""
        data = client.get("/feed-vs-milk").json()
        assert len(data["data"]) <= 100

    def test_feed_vs_milk_custom_limit(self):
        """Custom limit parameter should be respected."""
        data = client.get("/feed-vs-milk?limit=20").json()
        assert len(data["data"]) <= 20

    def test_feed_vs_milk_point_shape(self):
        data = client.get("/feed-vs-milk").json()
        for point in data["data"]:
            assert "Cow_ID"      in point
            assert "Feed_Intake" in point
            assert "Milk_Yield"  in point
