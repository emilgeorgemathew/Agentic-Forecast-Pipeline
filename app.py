import os
import json
import pickle
import re
from datetime import datetime

import pandas as pd
import dateparser

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from google import genai
from catboost import CatBoostRegressor


# =============================================================================
# CONFIG
# =============================================================================

NUMERIC_FEATURES = [
    "is_weekend","lag_35","lag_42","lag_49","rolling_mean_35_7",
    "rolling_std_35_7","dept_day_share","dept_mean_encoded",
    "store_dept_mean_encoded","rel_cases_to_baseline","store_cases_7",
    "store_cases_28","store_volatility_28","relative_to_store",
    "shock_ratio","dept_volatility_28_within_store","trucks_lag_35",
    "trucks_lag_42","trucks_3_rolling_mean_35","trucks_7_rolling_mean_35",
    "trucks_3_rolling_std_35","trucks_7_rolling_std_35",
    "weekly_truck_mean_lag35","truck_trend_3_vs_7","trucks_target_28d",
    "diesel_price","diesel_5w_mean","diesel_5w_max",
    "diesel_region_minus_us","cpi_level","cpi_6m_mean",
    "dept_weekend_mean","store_truck_mode","store_truck_mean",
    "store_truck_std","store_truck_ever_3plus","store_truck_ever_1or4",
    "store_almost_fixed_2","store_can_do_3_trucks","store_never_extreme",
    "store_truck_target_enc","cases_prophet","trucks_prophet"
]

CATEGORICAL_FEATURES = [
    "dept_id","store_id","gmm_name","dmm_name","dept_desc",
    "state_name","day_of_week","holiday_name","dept_near_holiday_5",
    "dept_near_holiday_10","dept_weekday","dept_holiday_interact"
]

DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}",
    r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*,?\s+\d{2,4}"
]


# =============================================================================
# LOAD FILES
# =============================================================================

with open("feature_means.json","r") as f:
    NUMERIC_FEATURE_MEANS = json.load(f)["numeric_feature_means"]

df_hist = pd.read_csv("date_trucks_cases.csv", parse_dates=["dt"]).sort_values("dt")
MIN_DATE, MAX_DATE = df_hist["dt"].min(), df_hist["dt"].max()

with open("best_trucks_ts_cv.pkl","rb") as f:
    trucks_model = pickle.load(f)

cases_model = CatBoostRegressor()
cases_model.load_model("best_cases_catboost_ts_cv.cbm")


# =============================================================================
# GEMINI CLIENT (CORRECT NEW SDK)
# =============================================================================

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EXTRACT_PROMPT = """
Extract this query into structured JSON with EXACT keys:

["dt","state_name","store_id","dept_id","gmm_name","dmm_name","dept_desc",
"day_of_week","holiday_name","dept_near_holiday_5","dept_near_holiday_10",
"dept_weekday","dept_holiday_interact"].

State names → US 2-letter code (Maryland → MD).
If missing → null.
Return ONLY JSON.
"""

def extract_with_gemini(query: str) -> dict:
    try:
        res = client.responses.generate(
            model="gemini-1.5-flash-latest",
            input=[EXTRACT_PROMPT, query]
        )
        return json.loads(res.output_text)
    except:
        return {}


# =============================================================================
# DATE PARSING (FINAL FIXED VERSION)
# =============================================================================

def extract_date_substring(text: str):
    t = text.lower()
    for p in DATE_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def robust_date_parse(text: str) -> datetime:
    date_str = extract_date_substring(text)
    if not date_str:
        raise ValueError(f"Could not parse any date from: {text}")
    dt = dateparser.parse(date_str)
    if not dt:
        raise ValueError(f"Could not parse: {date_str}")
    return dt


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

def get_historical(dt: datetime):
    if dt < MIN_DATE or dt > MAX_DATE:
        return None
    r = df_hist[df_hist["dt"] == dt]
    if r.empty:
        return None
    row = r.iloc[0]
    return {"trucks": float(row["trucks"]), "cases": float(row["cases"])}

def build_features(ex: dict, dt: datetime):
    row = {"is_weekend": 1 if dt.weekday() >= 5 else 0}
    for c in NUMERIC_FEATURES:
        if c == "is_weekend": continue
        row[c] = ex.get(c, NUMERIC_FEATURE_MEANS.get(c, 0.0))
    for c in CATEGORICAL_FEATURES:
        row[c] = ex.get(c)
    return pd.DataFrame([row])

def run_models(feat: pd.DataFrame):
    t = float(trucks_model.predict(feat)[0])
    t = max(1, min(4, round(t)))
    c = float(cases_model.predict(feat)[0])
    return t, c


# =============================================================================
# FASTAPI
# =============================================================================

class QueryRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    date: str
    source: str
    Cases: float
    trucks: float
    raw_extracted: dict

app = FastAPI(title="Forecast API", version="1.0.0")

@app.get("/")
def root():
    return {"status":"ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_api(req: QueryRequest):

    ex = extract_with_gemini(req.query)

    try:
        dt = robust_date_parse(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    hist = get_historical(dt)
    if hist:
        return PredictionResponse(
            date=dt.date().isoformat(),
            source="historical",
            Cases=hist["cases"],
            trucks=hist["trucks"],
            raw_extracted=ex
        )

    feat = build_features(ex, dt)
    try:
        trucks, cases = run_models(feat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Prediction Error: {e}")

    return PredictionResponse(
        date=dt.date().isoformat(),
        source="model",
        Cases=cases,
        trucks=trucks,
        raw_extracted=ex
    )
