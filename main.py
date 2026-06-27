"""
main.py
-------
DairyVision AI – FastAPI Application Entry Point

This is the root of the DairyVision AI backend.  It:
    1. Configures the FastAPI app with metadata and CORS.
    2. Uses a lifespan context manager to load ML models at startup.
    3. Registers all route modules under their respective prefixes.
    4. Exposes a GET / status endpoint.

Usage
-----
    # From the project root (Milk Prediction Ai/)
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs
----------------
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Internal imports ──────────────────────────────────────────────────────────
from backend.ml_service import ml_service
from backend.routes import predict, feed, analytics
from backend.schemas import StatusResponse

# ── Application metadata ──────────────────────────────────────────────────────
APP_TITLE       = "DairyVision AI"
APP_DESCRIPTION = """
## 🐄 DairyVision AI – Dairy Farm Management System

An AI-powered REST API that helps dairy farmers:
- **Predict tomorrow's milk production** using a Random Forest Regressor
- **Detect unhealthy cows** using a Random Forest Classifier
- **Optimise feed intake** via intelligent recommendation engine
- **Analyse herd trends** through aggregated analytics endpoints

### Tech Stack
- **Python** + **FastAPI**
- **Scikit-learn** (RandomForestRegressor + RandomForestClassifier)
- **Pandas** for data processing
- **Joblib** for model serialisation

### Dataset
500-record synthetic dairy farm dataset with 13 features per cow.

---
*Ready to connect with a Lovable AI React frontend.*
"""
APP_VERSION     = "1.0.0"
APP_CONTACT     = {"name": "DairyVision AI", "email": "support@dairyvision.ai"}


# ── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.

    Startup  → loads both ML models into memory so they are instantly
               available for every request without per-request I/O.
    Shutdown → nothing special required (models are in-memory only).
    """
    print("🚀  DairyVision AI starting up …")
    try:
        ml_service.load()
    except FileNotFoundError as exc:
        print(
            f"⚠️   Model files not found: {exc}\n"
            "    Run:  python backend/train_models.py  to train and save models."
        )
    yield  # application runs here
    print("👋  DairyVision AI shutting down …")


# ── FastAPI app instance ───────────────────────────────────────────────────────
app = FastAPI(
    title=APP_DESCRIPTION,
    version=APP_VERSION,
    contact=APP_CONTACT,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS middleware ────────────────────────────────────────────────────────────
# Allows the React frontend (Lovable AI) to call this API from any origin
# during development.  Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get(
    "/",
    response_model=StatusResponse,
    summary="API health check",
    description="Returns the current status of the DairyVision AI API.",
    tags=["Status"],
)
def root():
    """
    GET /
    -----
    Simple liveness probe.  Returns API name, version, and running status.
    Used by monitoring tools and the React frontend to verify connectivity.
    """
    return StatusResponse(
        status="running",
        message="Welcome to DairyVision AI – Dairy Farm Management System 🐄",
        version=APP_VERSION,
    )


# ── Register routers ──────────────────────────────────────────────────────────
# Each router handles a logical group of endpoints.

# Prediction endpoints  →  POST /predict-milk  |  POST /predict-health
app.include_router(predict.router)

# Feed optimisation     →  POST /feed-recommendation
app.include_router(feed.router)

# Analytics / dashboard →  GET /dashboard-summary  |  /milk-trend
#                           GET /health-distribution |  /feed-vs-milk
app.include_router(analytics.router)


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # auto-reload on file changes (dev only)
        log_level="info",
    )
