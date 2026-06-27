# 🐄 DairyVision AI – Dairy Farm Management System

> AI-powered FastAPI backend for predicting milk production, detecting cow health issues, optimising feed intake, and analysing herd trends.

---

## 📁 Project Structure

```
Milk Prediction Ai/
│
├── main.py                          # FastAPI app entry point (run this)
├── requirements.txt                 # Python dependencies
├── setup.bat                        # One-click Windows setup script
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── ml_service.py                # Singleton ML model loader & inference engine
│   ├── schemas.py                   # Pydantic request/response models
│   ├── train_models.py              # Train & save both ML models
│   │
│   ├── dataset/
│   │   ├── generate_dataset.py      # Generates synthetic dataset.csv (500 cows)
│   │   └── dataset.csv              # ← generated at runtime
│   │
│   ├── models/
│   │   ├── milk_model.pkl           # ← saved after training
│   │   ├── health_model.pkl         # ← saved after training
│   │   ├── label_encoder.pkl        # ← saved after training
│   │   ├── scaler.pkl               # ← saved after training
│   │   └── feature_columns.pkl      # ← saved after training
│   │
│   └── routes/
│       ├── __init__.py
│       ├── predict.py               # POST /predict-milk, POST /predict-health
│       ├── feed.py                  # POST /feed-recommendation
│       └── analytics.py            # GET /dashboard-summary, /milk-trend, etc.
│
└── tests/
    └── test_api.py                  # pytest integration tests
```

---

## 🚀 Quick Start

### Option A – One-Click Setup (Windows)
```bat
setup.bat
```

### Option B – Manual Setup

#### 1. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

#### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

#### 3. Generate the synthetic dataset
```powershell
python backend/dataset/generate_dataset.py
```

#### 4. Train the ML models
```powershell
python backend/train_models.py
```

#### 5. Start the API server
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | API health check & version |
| `POST` | `/predict-milk` | Predict tomorrow's milk yield (litres) |
| `POST` | `/predict-health` | Classify cow as Healthy / Unhealthy |
| `POST` | `/feed-recommendation` | Optimal feed intake recommendation |
| `GET`  | `/dashboard-summary` | KPI summary for dashboard |
| `GET`  | `/milk-trend` | Milk production trend by breed |
| `GET`  | `/health-distribution` | Healthy vs. unhealthy counts |
| `GET`  | `/feed-vs-milk` | Feed intake vs. milk scatter data |

### Interactive Docs
- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

---

## 📋 Request / Response Examples

### POST `/predict-milk`

**Request:**
```json
{
  "Breed": "Holstein",
  "Age": 4.5,
  "Weight": 650,
  "Feed_Intake": 22.0,
  "Water_Intake": 85.0,
  "Temperature": 22.0,
  "Humidity": 60.0,
  "Activity": 7.5,
  "Pregnant": 0,
  "Previous_Milk_Yield": 28.0
}
```

**Response:**
```json
{
  "predicted_milk_yield": 27.84,
  "unit": "litres/day",
  "confidence_note": "High confidence: stable conditions detected."
}
```

---

### POST `/predict-health`

**Response:**
```json
{
  "health_status": "Healthy",
  "confidence": 0.92,
  "recommendation": "✅ Cow appears healthy. Continue current feeding and monitoring schedule."
}
```

---

### POST `/feed-recommendation`

**Request:** (same as CowFeatures + optional `Target_Milk_Yield`)

**Response:**
```json
{
  "current_feed_intake": 20.0,
  "recommended_feed_intake": 23.5,
  "feed_adjustment_kg": 3.5,
  "current_predicted_milk": 25.2,
  "expected_milk_after": 27.8,
  "expected_milk_increase": 2.6,
  "advice": "📈 Increase daily feed by 3.5 kg to boost milk yield by ~2.6 litres/day."
}
```

---

### GET `/dashboard-summary`

**Response:**
```json
{
  "total_cows": 500,
  "total_milk_today": 9842.5,
  "average_milk": 19.68,
  "healthy_cows": 378,
  "unhealthy_cows": 122,
  "average_temperature": 24.8,
  "average_feed_intake": 20.3,
  "health_percentage": 75.6
}
```

---

## 🤖 ML Models

| Model | Type | Target | Algorithm |
|-------|------|--------|-----------|
| `milk_model.pkl` | Regression | `Milk_Yield_Tomorrow` (litres) | RandomForestRegressor (200 trees) |
| `health_model.pkl` | Classification | `Health_Status` (Healthy/Unhealthy) | RandomForestClassifier (200 trees) |

### Features Used by Both Models
| Feature | Type | Description |
|---------|------|-------------|
| `Breed` | Categorical | One of 5 breeds (one-hot encoded) |
| `Age` | Float | Years (0.5–20) |
| `Weight` | Float | Body weight kg |
| `Feed_Intake` | Float | Daily feed kg |
| `Water_Intake` | Float | Daily water litres |
| `Temperature` | Float | Ambient temperature °C |
| `Humidity` | Float | Relative humidity % |
| `Activity` | Float | Activity score 1–10 |
| `Pregnant` | Int | 1 = pregnant, 0 = not |
| `Previous_Milk_Yield` | Float | Yesterday's yield litres |

---

## 🧪 Running Tests

```powershell
# From the project root
pytest tests/ -v
```

---

## 🔌 Connecting a React Frontend (Lovable AI)

The API uses **CORS wildcard** (`allow_origins=["*"]`) which means any React frontend can call it directly. For Lovable AI, point your API calls to:

```
http://localhost:8000
```

All endpoints return clean JSON following the response models in `backend/schemas.py`.

For production, restrict CORS in `main.py`:
```python
allow_origins=["https://your-lovable-app.lovable.app"]
```

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Server host | `main.py` uvicorn call | `0.0.0.0` |
| Server port | `main.py` uvicorn call | `8000` |
| CORS origins | `main.py` middleware | `["*"]` |
| Dataset size | `generate_dataset.py` `N_COWS` | `500` |
| RF trees | `train_models.py` | `200` |
| Feed search steps | `routes/feed.py` | `100` |

---

## 🛠️ Troubleshooting

### `ImportError: DLL load failed while importing _mt19937`
This is caused by an **Application Control policy** (common on Lenovo enterprise devices) blocking numpy DLL files. Solutions:
1. Run as Administrator
2. Whitelist the `site-packages` DLLs in your Application Control policy
3. Use a virtual environment and add its path to the allowlist
4. Contact your IT administrator to whitelist numpy DLLs

### `FileNotFoundError: models/*.pkl`
Run training first: `python backend/train_models.py`

### `FileNotFoundError: dataset.csv`
Generate the dataset first: `python backend/dataset/generate_dataset.py`

---

## 📄 License
MIT License – DairyVision AI
