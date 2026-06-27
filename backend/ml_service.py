"""
ml_service.py
-------------
Singleton ML service that loads the trained models and provides
inference methods consumed by the FastAPI route handlers.

Design decisions
----------------
* Models are loaded ONCE at application startup (lifespan event in main.py)
  to avoid repeated disk I/O on every request.
* A shared `_prepare_features()` helper applies the same preprocessing
  pipeline (encoding + scaling) used during training so predictions are
  always consistent.
* All public methods raise `RuntimeError` if called before models are loaded,
  giving a clear error message rather than a cryptic attribute error.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional

# ── Path helpers ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class MLService:
    """
    Wraps both trained Random Forest models and exposes prediction helpers.

    Attributes
    ----------
    milk_model     : RandomForestRegressor   – predicts tomorrow's milk (litres)
    health_model   : RandomForestClassifier  – predicts Healthy / Unhealthy
    label_encoder  : LabelEncoder            – decodes health class labels
    scaler         : StandardScaler          – normalises numeric features
    feature_cols   : list[str]               – ordered column list from training
    """

    def __init__(self):
        self.milk_model    = None
        self.health_model  = None
        self.label_encoder = None
        self.scaler        = None
        self.feature_cols  = None
        self._loaded       = False

    # ── Numeric feature names (must match train_models.py) ────────────────────
    NUMERIC_FEATURES = [
        "Age", "Weight", "Feed_Intake", "Water_Intake",
        "Temperature", "Humidity", "Activity", "Pregnant",
        "Previous_Milk_Yield",
    ]
    CATEGORICAL_FEATURES = ["Breed"]

    def load(self) -> None:
        """
        Load all model artifacts from disk.
        Called once during FastAPI's startup lifespan event.
        """
        def _path(filename: str) -> str:
            return os.path.join(MODELS_DIR, filename)

        self.milk_model    = joblib.load(_path("milk_model.pkl"))
        self.health_model  = joblib.load(_path("health_model.pkl"))
        self.label_encoder = joblib.load(_path("label_encoder.pkl"))
        self.scaler        = joblib.load(_path("scaler.pkl"))
        self.feature_cols  = joblib.load(_path("feature_columns.pkl"))
        self._loaded       = True

        print("✅  ML models loaded successfully.")

    def _check_loaded(self) -> None:
        """Guard: raises RuntimeError if models haven't been loaded yet."""
        if not self._loaded:
            raise RuntimeError(
                "ML models are not loaded. "
                "Ensure MLService.load() is called at application startup."
            )

    def _prepare_features(self, data: dict) -> pd.DataFrame:
        """
        Convert a raw feature dictionary into a model-ready DataFrame.

        Steps
        -----
        1. Build a single-row DataFrame with the raw features.
        2. One-hot encode the Breed column.
        3. Align columns to match those seen during training (fill missing with 0).
        4. Scale numeric features using the saved StandardScaler.

        Parameters
        ----------
        data : dict  – keys matching CowFeatures field names

        Returns
        -------
        pd.DataFrame – single-row, ready for model.predict()
        """
        # Step 1 – raw row
        row = pd.DataFrame([data])

        # Step 2 – one-hot encode Breed
        row_encoded = pd.get_dummies(row, columns=self.CATEGORICAL_FEATURES, drop_first=False)

        # Step 3 – align columns (add missing breed dummy cols as 0)
        for col in self.feature_cols:
            if col not in row_encoded.columns:
                row_encoded[col] = 0
        row_encoded = row_encoded[self.feature_cols]   # enforce order

        # Step 4 – scale numeric features
        row_encoded[self.NUMERIC_FEATURES] = self.scaler.transform(
            row_encoded[self.NUMERIC_FEATURES]
        )

        return row_encoded

    # ── Public inference methods ───────────────────────────────────────────────

    def predict_milk(self, features: dict) -> float:
        """
        Predict tomorrow's milk yield (litres/day).

        Parameters
        ----------
        features : dict – raw cow features

        Returns
        -------
        float – predicted litres (rounded to 2 dp)
        """
        self._check_loaded()
        X = self._prepare_features(features)
        prediction = self.milk_model.predict(X)[0]
        return round(float(max(0.0, prediction)), 2)

    def predict_health(self, features: dict) -> tuple[str, float]:
        """
        Predict health status and confidence score.

        Returns
        -------
        (health_label, confidence)  e.g. ("Healthy", 0.87)
        """
        self._check_loaded()
        X = self._prepare_features(features)

        class_idx   = self.health_model.predict(X)[0]
        probabilities = self.health_model.predict_proba(X)[0]
        confidence  = round(float(max(probabilities)), 4)
        label       = self.label_encoder.inverse_transform([class_idx])[0]

        return str(label), confidence

    def predict_milk_for_feed(self, features: dict, new_feed: float) -> float:
        """
        Predict milk yield if feed_intake were changed to `new_feed`.
        Used internally by the feed-recommendation logic.
        """
        modified = dict(features)
        modified["Feed_Intake"] = new_feed
        return self.predict_milk(modified)


# ── Global singleton instance ─────────────────────────────────────────────────
# Routes import this object; main.py calls ml_service.load() at startup.
ml_service = MLService()
