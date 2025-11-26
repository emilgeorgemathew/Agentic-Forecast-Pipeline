import os
import json
import pickle
import re
from datetime import datetime

import pandas as pd
import dateparser

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from google import genai
from catboost import CatBoostRegressor


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

NUMERIC_FEATURES = [
    "is_weekend", "lag_35", "lag_42", "lag_49", "rolling_mean_35_7",
    "rolling_std_35_7", "dept_day_share", "dept_mean_encoded",
    "store_dept_mean_encoded", "rel_cases_to_baseline", "store_cases_7",
    "store_cases_28", "store_volatility_28", "relative_to_store",
    "shock_ratio", "dept_volatility_28_within_store", "trucks_lag_35",
    "trucks_lag_42", "trucks_3_rolling_mean_35", "trucks_7_rolling_mean_35",
    "trucks_3_rolling_std_35", "trucks_7_rolling_std_35",
    "weekly_truck_mean_lag35", "truck_trend_3_vs_7", "trucks_target_28d",
    "diesel_price", "diesel_5w_mean", "diesel_5w_max",
    "diesel_region_minus_us", "cpi_level", "cpi_6m_mean",
    "dept_weekend_mean", "store_truck_mode", "store_truck_mean",
    "store_truck_std", "store_truck_ever_3plus", "store_truck_ever_1or4",
    "store_almost_fixed_2", "store_can_do_3_trucks", "store_never_extreme",
    "store_truck_target_enc", "cases_prophet", "trucks_prophet",
]

CATEGORICAL_FEATURES = [
    "dept_id", "store_id", "gmm_name", "dmm_name", "dept_desc",
    "state_name", "day_of_week", "holiday_name", "dept_near_holiday_5",
    "dept_near_holiday_10", "dept_weekday", "dept_holiday_interact",
]

DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                        # 2025-03-07
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                  # 1/7/2025
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}",
    r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*,?\s+\d{2,4}",
]


# =============================================================================
# LOAD FILES
# =============================================================================

FEATURE_MEANS_PATH = "feature_means.json"
DATE_TRUCKS_CASES_PATH = "date_trucks_cases.csv"
TRUCKS_MODEL_PATH = "best_trucks_ts_cv.pkl"
CASES_MODEL_PATH = "best_cases_catboost_ts_cv.cbm"

# Load feature means
with open(FEATURE_MEANS_PATH, "r") as f:
    means_payload = json.load(f)

NUMERIC_FEATURE_MEANS = means_payload["numeric_feature_means"]

# Load historical data
df_hist = pd.read_csv(DATE_TRUCKS_CASES_PATH, parse_dates=["dt"])
df_hist = df_hist.sort_values("dt")
MIN_DATE = df_hist["dt"].min()
MAX_DATE = df_hist["dt"].max()

# Load models
with open(TRUCKS_MODEL_PATH, "rb") as f:
    trucks_model = pickle.load(f)

cases_model = CatBoostRegressor()
cases_model.load_model(CASES_MODEL_PATH)


# =============================================================================
# GEMINI CLIENT (NEW SDK v1beta)
# =============================================================================

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

EXTRACT_PROMPT = """
Extract structured info from this query.

Return ONLY JSON with keys:
["dt","state_name","store_id","dept_id","gmm_name","dmm_name","dept_desc",
"day_of_week","holiday_name","dept_near_holiday_5","dept_near_holiday_10",
"dept_weekday","dept_holiday_interact"].

Infer US state codes (e.g., Maryland → MD).
If unknown → set null.
"""


def extract_with_gemini(user_query: str) -> dict:
    """Call Gemini using the new google-genai SDK."""
    try:
        response = client.responses.generate(
            model="gemini-1.5-flash-latest",
            input=[EXTRACT_PROMPT, user_query],
        )
        text = response.output_text
        return json.loads(text)
    except:
        return {}    # fallback if model fails


# =============================================================================
# DATE EXTRACTION
# =============================================================================

def extract_date_substring(text: str):
    text = text.lower()
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def robust_date_parse(full_query: str) -> datetime:
    date_text = extract_date_substring(full_query)
    if not date_text:
        raise ValueError(f"Could not parse any date from: {full_query}")

    dt = dateparser.parse(date_text)
    if not dt:
        raise ValueError(f"Could not parse date substring: {date_text}")

    return dt


# =============================================================================
# MODEL PIPELINE HELPERS
# =============================================================================

def get_historical(dt: datetime):
    if dt < MIN_DATE or dt > MAX_DATE:
        return None
    rowset = df_hist[df_hist["dt"] == dt]
    if rowset.empty:
        return None
    row = rowset.iloc[0]
    return {"trucks": float(row["trucks"]), "cases": float(row["cases"])}


def build_feature_row(extracted: dict, dt: datetime):
    row = {}

    # Weekend
    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0

    # Numeric features
    for col in NUMERIC_FEATURES:
        if col == "is_weekend":
            continue
        val = extracted.get(col)
        if val is None:
            val = NUMERIC_FEATURE_MEANS.get(col, 0.0)
        row[col] = val

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        row[col] = extracted.get(col)

    return pd.DataFrame([row])


def predict_models(feature_row: pd.DataFrame):
    t_pred = float(trucks_model.predict(feature_row)[0])
    t_clamped = max(1, min(4, round(t_pred)))
    c_pred = float(cases_model.predict(feature_row)[0])
    return {"trucks": float(t_clamped), "cases": float(c_pred)}


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


app = FastAPI(title="Trucks & Cases Prediction API", version="1.0.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "Trucks & Cases Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_api(req: QueryRequest):

    # 1. Gemini extraction
    extracted = extract_with_gemini(req.query)

    # 2. Date parsing
    try:
        dt = robust_date_parse(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    # 3. Historical lookup
    hist = get_historical(dt)
    if hist:
        return PredictionResponse(
            date=dt.date().isoformat(),
            source="historical",
            Cases=hist["cases"],
            trucks=hist["trucks"],
            raw_extracted=extracted
        )

    # 4. Model prediction
    try:
        feat = build_feature_row(extracted, dt)
        out = predict_models(feat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Prediction Error: {e}")

    return PredictionResponse(
        date=dt.date().isoformat(),
        source="model",
        Cases=out["cases"],
        trucks=out["trucks"],
        raw_extracted=extracted
    )
