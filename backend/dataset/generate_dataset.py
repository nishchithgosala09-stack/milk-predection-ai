"""
generate_dataset.py
-------------------
Script to generate a realistic synthetic dairy farm dataset (dataset.csv).
This script creates 500 cow records with realistic correlations between
features like breed, age, weight, feed intake, and milk yield.

Run this once to create the dataset before training the models.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_COWS = 500  # number of records

# ── Breed definitions ─────────────────────────────────────────────────────────
BREEDS = ["Holstein", "Jersey", "Brown Swiss", "Guernsey", "Ayrshire"]

# Base milk yield (litres/day) per breed (realistic averages)
BREED_MILK_BASE = {
    "Holstein":    28.0,
    "Jersey":      18.0,
    "Brown Swiss": 22.0,
    "Guernsey":    20.0,
    "Ayrshire":    21.0,
}

# Average weight (kg) per breed
BREED_WEIGHT_BASE = {
    "Holstein":    650,
    "Jersey":      450,
    "Brown Swiss": 600,
    "Guernsey":    500,
    "Ayrshire":    540,
}


def generate_record(cow_id: int) -> dict:
    """Generate a single realistic cow record."""

    # ── Breed & Age ──────────────────────────────────────────────────────────
    breed = np.random.choice(BREEDS)
    age   = round(np.random.uniform(1.5, 10.0), 1)   # years (1.5–10)

    # ── Weight ───────────────────────────────────────────────────────────────
    base_weight = BREED_WEIGHT_BASE[breed]
    weight = round(np.random.normal(base_weight, base_weight * 0.05), 1)
    weight = max(350, weight)  # floor at 350 kg

    # ── Feed & Water Intake ───────────────────────────────────────────────────
    # Feed intake (kg/day) correlates with weight
    feed_intake  = round(np.random.normal(weight * 0.033, 1.5), 2)
    feed_intake  = max(5.0, feed_intake)

    # Water intake (litres/day) ≈ 3–5× feed intake
    water_intake = round(np.random.normal(feed_intake * 4.0, 5.0), 2)
    water_intake = max(20.0, water_intake)

    # ── Environmental Variables ───────────────────────────────────────────────
    temperature = round(np.random.uniform(10.0, 40.0), 1)   # °C
    humidity    = round(np.random.uniform(30.0, 95.0), 1)   # %

    # ── Activity ─────────────────────────────────────────────────────────────
    activity = round(np.random.uniform(1.0, 10.0), 1)        # 1–10 scale

    # ── Pregnancy ────────────────────────────────────────────────────────────
    # ~25% of cows are pregnant at any given time
    pregnant = int(np.random.random() < 0.25)

    # ── Previous Milk Yield (yesterday) ──────────────────────────────────────
    base_yield = BREED_MILK_BASE[breed]

    # Age effect: peak at 4–6 years, drops off at extremes
    age_factor = 1.0 - 0.04 * abs(age - 5.0)

    # Feed effect: normalised around the mean feed intake for this weight
    mean_feed    = weight * 0.033
    feed_factor  = 1.0 + 0.02 * (feed_intake - mean_feed)

    # Temperature stress: optimal ≈ 18°C, drops above 25°C
    temp_factor  = 1.0 - max(0, (temperature - 25.0) * 0.01)

    # Pregnancy reduces yield by ~10%
    preg_factor  = 0.90 if pregnant else 1.0

    # Combine factors with random noise
    prev_milk = base_yield * age_factor * feed_factor * temp_factor * preg_factor
    prev_milk = round(max(3.0, prev_milk + np.random.normal(0, 1.5)), 2)

    # ── Health Status ─────────────────────────────────────────────────────────
    # Unhealthy if: very high temp, very low activity, low milk, extreme humidity
    unhealthy_score = 0
    if temperature > 35.0:
        unhealthy_score += 2
    if activity < 3.0:
        unhealthy_score += 2
    if prev_milk < base_yield * 0.6:
        unhealthy_score += 2
    if humidity > 85.0:
        unhealthy_score += 1
    if feed_intake < mean_feed * 0.7:
        unhealthy_score += 1

    # Add stochastic element: some naturally healthy/sick cows
    unhealthy_score += np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
    health_status = "Unhealthy" if unhealthy_score >= 3 else "Healthy"

    # ── Tomorrow's Milk Yield ─────────────────────────────────────────────────
    # Tomorrow correlates strongly with today, modified by health & environment
    health_penalty = 0.75 if health_status == "Unhealthy" else 1.0
    milk_tomorrow  = prev_milk * health_penalty
    milk_tomorrow  = round(max(1.0, milk_tomorrow + np.random.normal(0, 0.8)), 2)

    return {
        "Cow_ID":             f"COW_{cow_id:04d}",
        "Breed":              breed,
        "Age":                age,
        "Weight":             weight,
        "Feed_Intake":        feed_intake,
        "Water_Intake":       water_intake,
        "Temperature":        temperature,
        "Humidity":           humidity,
        "Activity":           activity,
        "Pregnant":           pregnant,
        "Health_Status":      health_status,
        "Previous_Milk_Yield": prev_milk,
        "Milk_Yield_Tomorrow": milk_tomorrow,
    }


def main():
    """Generate and save the dataset."""
    records = [generate_record(i + 1) for i in range(N_COWS)]
    df = pd.DataFrame(records)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"✅  Dataset saved → {out_path}")
    print(f"    Rows         : {len(df)}")
    print(f"    Columns      : {list(df.columns)}")
    print(f"\n    Health dist  :\n{df['Health_Status'].value_counts().to_string()}")
    print(f"\n    Milk yield   :\n{df['Milk_Yield_Tomorrow'].describe().round(2).to_string()}")


if __name__ == "__main__":
    main()
