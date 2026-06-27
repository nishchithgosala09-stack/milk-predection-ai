@echo off
:: ============================================================
::  DairyVision AI – One-Click Setup Script (Windows)
::  Run this from the project root:  setup.bat
:: ============================================================

echo.
echo ╔═══════════════════════════════════════════════╗
echo ║        DairyVision AI – Setup Script          ║
echo ╚═══════════════════════════════════════════════╝
echo.

:: ── Step 1: Create virtual environment ──────────────────────────────────────
echo [1/4] Creating Python virtual environment ...
python -m venv venv
if errorlevel 1 (
    echo ❌  Failed to create virtual environment. Is Python 3.10+ installed?
    pause
    exit /b 1
)
echo ✅  Virtual environment created.
echo.

:: ── Step 2: Activate and install dependencies ────────────────────────────────
echo [2/4] Installing dependencies from requirements.txt ...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌  Dependency installation failed.
    pause
    exit /b 1
)
echo ✅  Dependencies installed.
echo.

:: ── Step 3: Generate the dataset ─────────────────────────────────────────────
echo [3/4] Generating synthetic dataset (500 cows) ...
python -m backend.dataset.generate_dataset
if errorlevel 1 (
    echo ❌  Dataset generation failed.
    pause
    exit /b 1
)
echo ✅  Dataset saved to backend/dataset/dataset.csv
echo.

:: ── Step 4: Train models ──────────────────────────────────────────────────────
echo [4/4] Training ML models (this may take ~30 seconds) ...
python backend/train_models.py
if errorlevel 1 (
    echo ❌  Model training failed.
    pause
    exit /b 1
)
echo ✅  Models saved to backend/models/
echo.

:: ── Done ─────────────────────────────────────────────────────────────────────
echo ╔═══════════════════════════════════════════════════════════╗
echo ║  ✅  Setup complete!  Start the server with:              ║
echo ║                                                           ║
echo ║      venv\Scripts\activate.bat                           ║
echo ║      uvicorn main:app --reload --host 0.0.0.0 --port 8000 ║
echo ║                                                           ║
echo ║  Docs → http://localhost:8000/docs                        ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
pause
