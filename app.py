import os
import json
import pickle
import re
from datetime import date
from typing import Dict, Optional, Any

import pandas as pd
import dateparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
from google import genai


# ============================================================
# CONSTANTS / FEATURE DEFINITIONS
# ============================================================

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

# simple state canonicalization; Gemini should mostly output 2-letter codes,
# but we keep this for fallback
STATE_MAP: Dict[str, str] = {
    "md": "MD", "maryland": "MD",
    "va": "VA", "virginia": "VA",
    "nh": "NH", "new hampshire": "NH",
    "de": "DE", "delaware": "DE",
}


# ============================================================
# FILE PATHS / LOADING
# ============================================================

FEATURE_MEANS_PATH = os.environ.get("FEATURE_MEANS_JSON", "feature_means.json")
DATE_TRUCKS_CASES_PATH = os.environ.get("DATE_TRUCKS_CASES_CSV", "date_trucks_cases.csv")
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_ts_cv.pkl")
CASES_MODEL_PATH = os.environ.get("CASES_MODEL_PATH", "best_cases_catboost_ts_cv.cbm")

if not os.path.exists(FEATURE_MEANS_PATH):
    raise RuntimeError(f"feature_means.json not found at {FEATURE_MEANS_PATH}")

with open(FEATURE_MEANS_PATH, "r") as f:
    means_payload = json.load(f)

NUMERIC_FEATURE_MEANS: Dict[str, float] = means_payload.get("numeric_feature_means", {})

if not os.path.exists(DATE_TRUCKS_CASES_PATH):
    raise RuntimeError(f"date_trucks_cases.csv not found at {DATE_TRUCKS_CASES_PATH}")

df_hist = pd.read_csv(DATE_TRUCKS_CASES_PATH, parse_dates=["dt"])
df_hist = df_hist.sort_values("dt").reset_index(drop=True)
MIN_DATE: date = df_hist["dt"].min().date()
MAX_DATE: date = df_hist["dt"].max().date()

if not os.path.exists(TRUCKS_MODEL_PATH):
    raise RuntimeError(f"Trucks model not found at {TRUCKS_MODEL_PATH}")
if not os.path.exists(CASES_MODEL_PATH):
    raise RuntimeError(f"Cases model not found at {CASES_MODEL_PATH}")

with open(TRUCKS_MODEL_PATH, "rb") as f:
    trucks_model = pickle.load(f)

cases_model = CatBoostRegressor()
cases_model.load_model(CASES_MODEL_PATH)


# ============================================================
# GEMINI CLIENT + PROMPT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var is required.")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

client = genai.Client(api_key=GEMINI_API_KEY)

EXTRACT_PROMPT = """
You are a strict JSON information extractor for a retail forecasting system.

Given a natural language query about cases & trucks, extract:

- "dt": date in ISO format "YYYY-MM-DD"
- "state_name": 2-letter US state code (e.g., MD, VA); if you see full state name, convert it.
- "store_id": integer store id if mentioned, otherwise null.

Rules:
- If the user gives a date like "Jan 1 2025", "9th September 2025", "2025/11/17", convert it to "YYYY-MM-DD".
- If the user does NOT specify a date, set "dt" to null.
- If you cannot detect a state, set "state_name" to null.
- Output ONLY a single valid JSON object, no prose, no comments.
"""


def call_gemini_extract(user_query: str) -> Dict[str, Any]:
    """
    Use google-genai v1beta client to get a JSON with dt, state_name, store_id.
    If anything goes wrong (model error, quota, bad JSON), we return {} and
    let local fallback handle it.
    """
    try:
        resp = client.responses.generate(
            model=GEMINI_MODEL_NAME,
            input=[EXTRACT_PROMPT, user_query],
        )
        text = resp.output_text.strip()

        # In case Gemini ever adds extra text (it shouldn't), slice JSON
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        json_str = text[start : end + 1]

        return json.loads(json_str)
    except Exception:
        # silent fallback; we don't crash the API if Gemini dies
        return {}


# ============================================================
# FASTAPI MODELS
# ============================================================

class QueryRequest(BaseModel):
    query: str


class PredictionResponse(BaseModel):
    date: str
    source: str  # "historical" or "model"
    Cases: float
    trucks: float
    raw_extracted: Dict[str, Any]


app = FastAPI(title="Trucks & Cases Prediction API", version="1.0.0")


# ============================================================
# LOCAL FALLBACK HELPERS
# ============================================================

def extract_state_fallback(text: str) -> Optional[str]:
    t = text.lower()
    for key, code in STATE_MAP.items():
        if key in t:
            return code
    return None


def robust_date_parse(full_query: str) -> date:
    """
    Strong parser that tries multiple strategies using dateparser.
    Handles things like:
      - "value for virginia 9th september 2025"
      - "Whats the value for Maryland store 10001 on Jan 1 2025"
      - "2025/11/17"
    """
    # 1. Try parsing the full query
    dt = dateparser.parse(
        full_query,
        settings={"PREFER_DATES_FROM": "future", "DATE_ORDER": "DMY"},
        languages=["en"],
    )
    if dt is not None:
        return dt.date()

    # 2. Try to grab date-like chunks
    patterns = [
        r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}",  # 2025-11-17 or 2025/11/17
        r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",  # 17/11/2025
        r"\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{2,4}",  # 9th September 2025
        r"\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}",  # September 9th, 2025
    ]
    for pat in patterns:
        m = re.search(pat, full_query, flags=re.IGNORECASE)
        if m:
            chunk = m.group(0)
            dt2 = dateparser.parse(
                chunk,
                settings={"PREFER_DATES_FROM": "future", "DATE_ORDER": "DMY"},
                languages=["en"],
            )
            if dt2 is not None:
                return dt2.date()

    raise ValueError(f"Could not parse any date from: {full_query}")


def parse_dt_from_extracted(extracted: Dict[str, Any], original_query: str) -> date:
    """
    If Gemini gave us a dt → try to parse it.
    Otherwise fall back to robust_date_parse on the original text.
    """
    dt_str = extracted.get("dt")
    if dt_str:
        # try ISO first
        try:
            return date.fromisoformat(dt_str[:10])
        except Exception:
            # fall back to dateparser on that string
            dt = dateparser.parse(
                dt_str,
                settings={"PREFER_DATES_FROM": "future", "DATE_ORDER": "DMY"},
                languages=["en"],
            )
            if dt is not None:
                return dt.date()
    # fallback to full-query parsing
    return robust_date_parse(original_query)


def get_historical_values_if_available(dt: date) -> Optional[Dict[str, float]]:
    if dt < MIN_DATE or dt > MAX_DATE:
        return None
    mask = df_hist["dt"].dt.date == dt
    if not mask.any():
        return None
    row = df_hist.loc[mask].iloc[0]
    return {"trucks": float(row["trucks"]), "cases": float(row["cases"])}


def build_feature_row(dt: date, state_name: Optional[str]) -> pd.DataFrame:
    """
    Build a single-row DataFrame for the models:
    - numeric features mostly from means (except is_weekend from dt)
    - categorical features minimal: state_name, day_of_week, others = None
    """
    row: Dict[str, Any] = {}

    # is_weekend from date
    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0

    # other numeric features from means (fallback 0.0)
    for col in NUMERIC_FEATURES:
        if col == "is_weekend":
            continue
        row[col] = NUMERIC_FEATURE_MEANS.get(col, 0.0)

    # categorical features
    for col in CATEGORICAL_FEATURES:
        if col == "state_name":
            row[col] = state_name
        elif col == "day_of_week":
            row[col] = dt.strftime("%A")  # e.g., "Monday"
        else:
            row[col] = None

    return pd.DataFrame([row])


def predict_with_models(feature_row: pd.DataFrame) -> Dict[str, float]:
    trucks_pred = float(trucks_model.predict(feature_row)[0])
    cases_pred = float(cases_model.predict(feature_row)[0])

    # clamp trucks to 1..4 as before
    try:
        trucks_int = int(round(trucks_pred))
    except Exception:
        trucks_int = 1
    trucks_int = max(1, min(4, trucks_int))

    return {"trucks": float(trucks_int), "cases": cases_pred}


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Trucks & Cases Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_trucks_and_cases(req: QueryRequest):
    user_query = req.query or ""

    # 1) Gemini extraction
    extracted = call_gemini_extract(user_query)

    # 2) Date parsing (Gemini dt → robust fallback)
    try:
        dt = parse_dt_from_extracted(extracted, user_query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    # 3) State: Gemini first, fallback local
    state_name = extracted.get("state_name")
    if isinstance(state_name, str):
        state_name = state_name.strip().upper()
        if len(state_name) != 2:
            # maybe it returned 'Maryland' etc
            state_name = STATE_MAP.get(state_name.lower())
    if not state_name:
        state_name = extract_state_fallback(user_query)

    # 4) Historical check
    hist = get_historical_values_if_available(dt)
    if hist is not None:
        return PredictionResponse(
            date=dt.isoformat(),
            source="historical",
            Cases=hist["cases"],
            trucks=hist["trucks"],
            raw_extracted={"dt": extracted.get("dt"), "state_name": state_name, "store_id": extracted.get("store_id")},
        )

    # 5) Model prediction
    try:
        feat_row = build_feature_row(dt, state_name)
        preds = predict_with_models(feat_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return PredictionResponse(
        date=dt.isoformat(),
        source="model",
        Cases=preds["cases"],
        trucks=preds["trucks"],
        raw_extracted={"dt": extracted.get("dt"), "state_name": state_name, "store_id": extracted.get("store_id")},
    )
