import os
import json
import pickle
import re
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor

import dateparser  # <- FIX: bullet-proof date extraction
from google import genai

# ==============================
# Gemini client
# ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# ==============================
# Feature definitions
# ==============================

NUMERIC_FEATURES = [
    "is_weekend","lag_35","lag_42","lag_49","rolling_mean_35_7","rolling_std_35_7",
    "dept_day_share","dept_mean_encoded","store_dept_mean_encoded","rel_cases_to_baseline",
    "store_cases_7","store_cases_28","store_volatility_28","relative_to_store","shock_ratio",
    "dept_volatility_28_within_store","trucks_lag_35","trucks_lag_42",
    "trucks_3_rolling_mean_35","trucks_7_rolling_mean_35","trucks_3_rolling_std_35",
    "trucks_7_rolling_std_35","weekly_truck_mean_lag35","truck_trend_3_vs_7",
    "trucks_target_28d","diesel_price","diesel_5w_mean","diesel_5w_max",
    "diesel_region_minus_us","cpi_level","cpi_6m_mean","dept_weekend_mean",
    "store_truck_mode","store_truck_mean","store_truck_std","store_truck_ever_3plus",
    "store_truck_ever_1or4","store_almost_fixed_2","store_can_do_3_trucks",
    "store_never_extreme","store_truck_target_enc","cases_prophet","trucks_prophet",
]

CATEGORICAL_FEATURES = [
    "dept_id","store_id","gmm_name","dmm_name","dept_desc","state_name",
    "day_of_week","holiday_name","dept_near_holiday_5","dept_near_holiday_10",
    "dept_weekday","dept_holiday_interact",
]

# ==============================
# Load feature means JSON
# ==============================

with open("feature_means.json","r") as f:
    NUMERIC_FEATURE_MEANS = json.load(f)["numeric_feature_means"]

# ==============================
# Load historical date/trucks/cases
# ==============================

df_hist = pd.read_csv("date_trucks_cases.csv", parse_dates=["dt"])
df_hist = df_hist.sort_values("dt").reset_index(drop=True)

MIN_DATE = df_hist["dt"].min()
MAX_DATE = df_hist["dt"].max()

# ==============================
# Load models
# ==============================

with open("best_trucks_ts_cv.pkl","rb") as f:
    trucks_model = pickle.load(f)

cases_model = CatBoostRegressor()
cases_model.load_model("best_cases_catboost_ts_cv.cbm")

# ==============================
# FastAPI
# ==============================

app = FastAPI(
    title="Trucks & Cases Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

class QueryRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    date: str
    source: str
    Cases: float
    trucks: float
    raw_extracted: Dict[str,Any]

# ==============================
# UNIVERSAL DATE EXTRACTION (THE FIX)
# ==============================

def robust_date_parse(text: str) -> datetime:
    """
    Bullet-proof date parser using dateparser library.
    Handles:
    - Jan 1 2025
    - 1st January 2025
    - 01/01/25
    - 2025-01-01
    - etc.
    """
    dt = dateparser.parse(text)
    if dt is None:
        raise ValueError(f"Could not parse any date from: {text}")
    return dt

# ==============================
# Gemini extractor
# ==============================

EXTRACTION_PROMPT = """
Return JSON with keys:
"dt", "state_name", "store_id", "dept_id"
All others are allowed but optional.
Output ONLY JSON. If a field is missing, set to null.
"""

def extract_with_gemini(q: str) -> Dict[str,Any]:
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"role":"user","parts":[EXTRACTION_PROMPT + "\nQuery:\n" + q]}]
        )
        text = resp.candidates[0].content.parts[0].text.strip()

        # extract JSON
        start = text.find("{")
        end   = text.rfind("}")
        if start==-1 or end==-1:
            return {}
        
        return json.loads(text[start:end+1])

    except:
        return {}

# ==============================
# Build feature row
# ==============================

def build_feature_row(extracted: Dict[str,Any], dt: datetime):
    row = {}

    # weekend flag
    row["is_weekend"] = 1 if dt.weekday()>=5 else 0

    # numeric feature defaults
    for col in NUMERIC_FEATURES:
        if col=="is_weekend": continue
        val = extracted.get(col)
        if val is None:
            val = NUMERIC_FEATURE_MEANS.get(col,0.0)
        row[col] = val

    # categorical values included only in raw_extracted; NOT passed to models
    for col in CATEGORICAL_FEATURES:
        row[col] = extracted.get(col)

    return pd.DataFrame([row])

# ==============================
# Historical lookup
# ==============================

def find_historical(dt: datetime):
    if dt < MIN_DATE or dt > MAX_DATE:
        return None
    rows = df_hist[df_hist["dt"]==dt]
    if len(rows)==0: return None
    row = rows.iloc[0]
    return {"trucks":float(row["trucks"]), "cases":float(row["cases"])}

# ==============================
# Prediction endpoint
# ==============================

@app.post("/predict", response_model=PredictionResponse)
def predict(req: QueryRequest):

    # 1) Extract with Gemini
    extracted = extract_with_gemini(req.query) or {}

    # 2) Parse date robustly from ENTIRE query (not from extractor)
    try:
        dt = robust_date_parse(req.query)
    except Exception as e:
        raise HTTPException(400, f"Invalid date: {e}")

    # 3) Check historical values
    hist = find_historical(dt)
    if hist:
        return PredictionResponse(
            date=dt.date().isoformat(),
            source="historical",
            Cases=hist["cases"],
            trucks=hist["trucks"],
            raw_extracted=extracted
        )

    # 4) Model prediction
    features = build_feature_row(extracted, dt)
    X = features[NUMERIC_FEATURES]

    try:
        trucks_pred = float(trucks_model.predict(X)[0])
        cases_pred  = float(cases_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Model prediction error: {e}")

    # trucks rounded 1–4
    trucks_final = max(1, min(4, int(round(trucks_pred))))

    return PredictionResponse(
        date=dt.date().isoformat(),
        source="model",
        Cases=cases_pred,
        trucks=trucks_final,
        raw_extracted=extracted
    )

# ==============================
# Health endpoints
# ==============================

@app.get("/health")
def health(): return {"ok":True}

@app.get("/")
def root(): return {"status":"ok","message":"Trucks & Cases Prediction API"}
