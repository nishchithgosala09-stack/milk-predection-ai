"""
train_models.py
---------------
Trains two ML models on the dairy farm dataset and saves them with Joblib:

    1. milk_model.pkl    → RandomForestRegressor  (predict tomorrow's milk yield)
    2. health_model.pkl  → RandomForestClassifier (predict healthy / unhealthy)

Run this script once (or whenever you retrain):
    python train_models.py

Outputs
-------
    backend/models/milk_model.pkl
    backend/models/health_model.pkl
    backend/models/label_encoder.pkl   (for health target encoding)
    backend/models/scaler.pkl          (StandardScaler for numeric features)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    classification_report,
    accuracy_score,
)
print("✅ Imports completed")
print("Starting training...")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature columns used by BOTH models ───────────────────────────────────────
# Cow_ID is dropped (identifier), Health_Status / Milk_Yield_Tomorrow are targets.
NUMERIC_FEATURES = [
    "Age", "Weight", "Feed_Intake", "Water_Intake",
    "Temperature", "Humidity", "Activity", "Pregnant",
    "Previous_Milk_Yield",
]

CATEGORICAL_FEATURES = ["Breed"]   # will be one-hot encoded

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES   # order matters for UI


def load_and_prepare(df: pd.DataFrame):
    """
    Encode categorical columns and return the feature matrix X,
    plus the fitted encoders / scaler for persistence.

    Returns
    -------
    X        : pd.DataFrame  – model-ready feature matrix
    encoder  : LabelEncoder  – fitted on Health_Status target
    scaler   : StandardScaler – fitted on numeric features
    breed_dummies_cols : list[str] – column names produced by get_dummies
    """

    # ── One-hot encode Breed ──────────────────────────────────────────────────
    df_encoded = pd.get_dummies(df[ALL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False)

    # ── Scale numeric features ────────────────────────────────────────────────
    scaler = StandardScaler()
    df_encoded[NUMERIC_FEATURES] = scaler.fit_transform(df_encoded[NUMERIC_FEATURES])

    # ── Encode Health_Status target ───────────────────────────────────────────
    encoder = LabelEncoder()
    y_health = encoder.fit_transform(df["Health_Status"])   # Healthy=0, Unhealthy=1 (alphabetical)

    y_milk = df["Milk_Yield_Tomorrow"].values

    return df_encoded, y_milk, y_health, encoder, scaler, list(df_encoded.columns)


def train_milk_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the milk-yield regression model."""
    print("\n──────────────────────────────────────────")
    print("🥛  Training Milk Yield Regressor …")
    print("──────────────────────────────────────────")

    model = RandomForestRegressor(
        n_estimators=200,       # 200 trees for stability
        max_depth=15,           # prevent over-fitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,              # use all CPU cores
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"   MAE  : {mae:.3f} litres")
    print(f"   R²   : {r2:.4f}")

    return model


def train_health_model(X_train, X_test, y_train, y_test, class_names):
    """Train and evaluate the health-status classification model."""
    print("\n──────────────────────────────────────────")
    print("🐄  Training Health Classifier …")
    print("──────────────────────────────────────────")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handle class imbalance
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"   Accuracy : {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    return model


def main():
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        sys.exit(
            f"❌  Dataset not found at {DATASET_PATH}\n"
            "    Run:  python dataset/generate_dataset.py  first."
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"📂  Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    # ── 2. Prepare features ───────────────────────────────────────────────────
    X, y_milk, y_health, encoder, scaler, feature_cols = load_and_prepare(df)

    # ── 3. Train/test split (80/20) ───────────────────────────────────────────
    X_train, X_test, ym_train, ym_test, yh_train, yh_test = train_test_split(
        X, y_milk, y_health, test_size=0.2, random_state=42, stratify=y_health
    )

    print(f"   Train size : {len(X_train)}  |  Test size : {len(X_test)}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    class_names = list(encoder.classes_)
    milk_model   = train_milk_model(X_train, X_test, ym_train, ym_test)
    health_model = train_health_model(X_train, X_test, yh_train, yh_test, class_names)

    # ── 5. Save artifacts ─────────────────────────────────────────────────────
    joblib.dump(milk_model,   os.path.join(MODELS_DIR, "milk_model.pkl"))
    joblib.dump(health_model, os.path.join(MODELS_DIR, "health_model.pkl"))
    joblib.dump(encoder,      os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
    # Save the ordered feature column list so inference can align columns
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    print("\n✅  Models saved to backend/models/")
    print("   milk_model.pkl | health_model.pkl | label_encoder.pkl | scaler.pkl | feature_columns.pkl")


if __name__ == "__main__":
    main()
