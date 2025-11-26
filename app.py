# -------------------------------
# FINAL WORKING BACKEND (NO GEMINI)
# --------------------------------

import os
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
import dateparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle

# -----------------------------------------------------
# Load models, dataset, feature means, categorical maps
# -----------------------------------------------------
CASES_MODEL = pickle.load(open("best_cases_ts_cv.pkl", "rb"))
TRUCKS_MODEL = pickle.load(open("best_trucks_ts_cv.pkl", "rb"))

df = pd.read_csv("merged_data_new.csv", low_memory=False)
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

feature_means = json.load(open("feature_means.json", "r"))

CATEGORICAL_FEATURES = ["dept_id", "store_id", "gmm_name", "dmm_name",
                        "dept_desc", "state_name", "day_of_week",
                        "holiday_name", "dept_near_holiday_5",
                        "dept_near_holiday_10", "dept_weekday",
                        "dept_holiday_interact"]

NUMERIC_FEATURES = list(feature_means.keys())

MIN_DATE = df["dt"].min().date()
MAX_DATE = df["dt"].max().date()

STATE_MAP = {
    "maryland": "MD", "md": "MD",
    "virginia": "VA", "va": "VA",
    "pennsylvania": "PA", "pa": "PA",
    "delaware": "DE", "de": "DE",
}

# -------------------------
# FASTAPI MODELS
# -------------------------
class QueryRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    date: str
    source: str
    Cases: float
    trucks: float
    raw_extracted: dict


app = FastAPI(title="Trucks & Cases Prediction API")


# ---------------------------------------------------------
# High–accuracy date extractor
# ---------------------------------------------------------
def extract_date(text: str):
    text = text.strip()

    # 1) Direct dateparser
    dt = dateparser.parse(text)
    if dt:
        return dt.date()

    # 2) Regex for formats like 2025/11/17 or 17/11/2025
    patterns = [
        r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})",
        r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})",
        r"(\d{1,2}\s+\w+\s+\d{4})"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            dt = dateparser.parse(m.group(1))
            if dt:
                return dt.date()

    # 3) English natural language like "next monday"
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    if dt:
        return dt.date()

    return None


# ---------------------------------------------------------
# Extract state robustly
# ---------------------------------------------------------
def extract_state(text: str):
    t = text.lower()
    for k, v in STATE_MAP.items():
        if k in t:
            return v
    return None


# ---------------------------------------------------------
# Build feature row for model
# ---------------------------------------------------------
def build_feature_row(dt, state):
    row = {}

    # numeric features → mean
    for col in NUMERIC_FEATURES:
        row[col] = feature_means[col]

    # categorical → smallest/default option
    for col in CATEGORICAL_FEATURES:
        row[col] = "NONE"

    # override some known features
    row["state_name"] = state
    row["day_of_week"] = dt.weekday()

    df_row = pd.DataFrame([row])
    return df_row


# ---------------------------------------------------------
# API Routes
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Trucks & Cases Prediction API"}

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_api(req: QueryRequest):

    query = req.query.lower()

    # ----------- DATE EXTRACTION --------------
    dt = extract_date(req.query)
    if not dt:
        raise HTTPException(400, f"Invalid date: Could not parse any date from: {req.query}")

    # ----------- STATE EXTRACTION -------------
    state = extract_state(req.query)
    if not state:
        state = "MD"  # default fallback

    # ----------- CHECK IF DATE EXISTS ----------
    dt_in_range = (dt >= MIN_DATE) and (dt <= MAX_DATE)

    if dt_in_range:
        row = df[df["dt"] == pd.to_datetime(dt)]
        if not row.empty:
            return PredictionResponse(
                date=str(dt),
                source="historical",
                Cases=float(row.iloc[0]["cases"]),
                trucks=float(row.iloc[0]["trucks"]),
                raw_extracted={"dt": str(dt), "state_name": state},
            )

    # ----------- MODEL PREDICTION --------------
    features = build_feature_row(dt, state)

    try:
        pred_cases = CASES_MODEL.predict(features)[0]
        pred_trucks = TRUCKS_MODEL.predict(features)[0]
    except Exception as e:
        raise HTTPException(500, f"Model prediction failed: {e}")

    return PredictionResponse(
        date=str(dt),
        source="model_forecast",
        Cases=float(pred_cases),
        trucks=float(pred_trucks),
        raw_extracted={"dt": str(dt), "state_name": state},
    )


