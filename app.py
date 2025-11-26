import os
import json
import pickle
import re
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from catboost import CatBoostRegressor

# ==============================
# Feature definitions (must match training)
# ==============================

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

# ==============================
# Load feature means JSON
# ==============================

FEATURE_MEANS_PATH = os.environ.get("FEATURE_MEANS_JSON", "feature_means.json")
if not os.path.exists(FEATURE_MEANS_PATH):
    raise RuntimeError(f"feature_means.json not found at {FEATURE_MEANS_PATH}")

with open(FEATURE_MEANS_PATH, "r") as f:
    means_payload = json.load(f)

NUMERIC_FEATURE_MEANS: Dict[str, float] = means_payload.get("numeric_feature_means", {})

# If some numeric features have no mean (e.g., cases_prophet/trucks_prophet),
# we will default them to 0.0 at feature construction time.

# ==============================
# Load historical date/trucks/cases
# ==============================

DATE_TRUCKS_CASES_PATH = os.environ.get("DATE_TRUCKS_CASES_CSV", "date_trucks_cases.csv")
if not os.path.exists(DATE_TRUCKS_CASES_PATH):
    raise RuntimeError(f"date_trucks_cases.csv not found at {DATE_TRUCKS_CASES_PATH}")

df_hist = pd.read_csv(DATE_TRUCKS_CASES_PATH, parse_dates=["dt"])
df_hist = df_hist.sort_values("dt").reset_index(drop=True)

MIN_DATE = df_hist["dt"].min()
MAX_DATE = df_hist["dt"].max()

# ==============================
# Load models
# ==============================

TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_ts_cv.pkl")
CASES_MODEL_PATH = os.environ.get("CASES_MODEL_PATH", "best_cases_catboost_ts_cv.cbm")

if not os.path.exists(TRUCKS_MODEL_PATH):
    raise RuntimeError(f"Trucks model not found at {TRUCKS_MODEL_PATH}")
if not os.path.exists(CASES_MODEL_PATH):
    raise RuntimeError(f"Cases model not found at {CASES_MODEL_PATH}")

with open(TRUCKS_MODEL_PATH, "rb") as f:
    trucks_model = pickle.load(f)

cases_model = CatBoostRegressor()
cases_model.load_model(CASES_MODEL_PATH)

# ==============================
# FastAPI setup
# ==============================

app = FastAPI(title="Trucks & Cases Prediction API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str


class PredictionResponse(BaseModel):
    date: str
    source: str  # "historical" or "model"
    Cases: float
    trucks: float
    raw_extracted: Dict[str, Any]


# ==============================
# Local extraction helper
# ==============================

def extract_features_local(user_query: str) -> Dict[str, Any]:
    """
    Local extraction. Supports either:
      - A JSON object string (e.g. '{"dt":"2025-11-25","store_id":123}')
      - Space-separated key:value or key=value tokens
        (e.g. 'dt:2025-11-25 store_id:123 state_name:MD')

    Also:
      - If 'Maryland' / 'maryland' appears, sets state_name='MD', etc.
      - If a 2-letter state code appears as a token, uses that.

    If dt is missing, attempts to find an ISO date inside the text.
    """
    s = (user_query or "").strip()
    if not s:
        raise ValueError("Empty query provided for extraction")

    # 1) Try JSON object
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON extraction string: {e}") from e
    else:
        # 2) key:value or key=value parser
        data: Dict[str, Any] = {}
        tokens = re.split(r"\s+", s)
        for token in tokens:
            if ":" in token:
                k, v = token.split(":", 1)
            elif "=" in token:
                k, v = token.split("=", 1)
            else:
                continue
            k = k.strip()
            v = v.strip()
            if not k:
                continue

            if k in ("store_id", "dept_id"):
                try:
                    data[k] = int(v)
                except Exception:
                    data[k] = None
            else:
                data[k] = v

    # 3) If dt missing, look for ISO date yyyy-mm-dd in the whole text
    if "dt" not in data or data.get("dt") in (None, ""):
        m = re.search(r"\d{4}-\d{2}-\d{2}", s)
        if m:
            data["dt"] = m.group(0)

    # 4) State name mapping (full name or 2-letter code)
    if "state_name" not in data or not data.get("state_name"):
        state_map = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
            "california": "CA", "colorado": "CO", "connecticut": "CT",
            "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
            "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
            "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
            "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
            "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
            "montana": "MT", "nebraska": "NE", "nevada": "NV",
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
            "new york": "NY", "north carolina": "NC", "north dakota": "ND",
            "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
            "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
            "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
            "virginia": "VA", "washington": "WA", "west virginia": "WV",
            "wisconsin": "WI", "wyoming": "WY",
        }

        lower_text = s.lower()
        matched_code = None

        # full-name matches first
        for full_name, code in state_map.items():
            if full_name in lower_text:
                matched_code = code
                break

        # if nothing yet, look for 2-letter code tokens
        if not matched_code:
            for token in re.split(r"\W+", s.upper()):
                if token in state_map.values():
                    matched_code = token
                    break

        if matched_code:
            data["state_name"] = matched_code

    return data


# ==============================
# Feature building utilities
# ==============================

def parse_date(dt_str: Optional[str]) -> datetime:
    if dt_str is None:
        raise ValueError("Extracted date (dt) is null")
    try:
        return datetime.fromisoformat(dt_str[:10])
    except Exception as e:
        raise ValueError(f"Invalid dt format from extractor: {dt_str}") from e


def get_historical_values_if_available(dt: datetime) -> Optional[Dict[str, float]]:
    """
    If date within min/max AND exists in historical CSV, return dict with trucks & cases.
    Otherwise return None.
    """
    if dt < MIN_DATE or dt > MAX_DATE:
        return None
    mask = df_hist["dt"] == dt
    if not mask.any():
        return None

    row = df_hist.loc[mask].iloc[0]
    trucks = float(row["trucks"])
    cases = float(row["cases"])
    return {"trucks": trucks, "cases": cases}


def build_feature_row(extracted: Dict[str, Any], dt: datetime) -> pd.DataFrame:
    """
    Build a single-row DataFrame with all required features.
    Numeric features not provided by extractor use the precomputed means.
    Some easy, date-based features (e.g. is_weekend) are computed directly.
    """
    row: Dict[str, Any] = {}

    # Date-based numeric feature
    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0

    # Fill remaining numeric features from extracted (if present) or means
    for col in NUMERIC_FEATURES:
        if col == "is_weekend":
            continue

        value = extracted.get(col, None)

        if value is None:
            # fallback to precomputed mean or 0.0 if not available
            if col in NUMERIC_FEATURE_MEANS:
                value = NUMERIC_FEATURE_MEANS[col]
            else:
                value = 0.0

        row[col] = value

    # Categorical features straight from extracted
    for col in CATEGORICAL_FEATURES:
        row[col] = extracted.get(col, None)

    return pd.DataFrame([row])


def predict_with_models(feature_row: pd.DataFrame) -> Dict[str, float]:
    trucks_pred = float(trucks_model.predict(feature_row)[0])
    cases_pred = float(cases_model.predict(feature_row)[0])

    # Round trucks to nearest integer and clamp to 1..4
    try:
        trucks_rounded = int(round(trucks_pred))
    except Exception:
        trucks_rounded = 1

    trucks_clamped = max(1, min(4, trucks_rounded))

    return {"trucks": float(trucks_clamped), "cases": cases_pred}


# ==============================
# Health + root endpoints
# ==============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "message": "Trucks & Cases Prediction API"}


# ==============================
# Prediction endpoint
# ==============================

@app.post("/predict", response_model=PredictionResponse)
def predict_trucks_and_cases(req: QueryRequest):
    # 1. Extract structured info using local extractor
    try:
        extracted = extract_features_local(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")

    # 2. Parse date
    try:
        dt = parse_date(extracted.get("dt"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date from query: {e}")

    # 3. Check historical data (within min/max and exact date match)
    hist_values = get_historical_values_if_available(dt)
    if hist_values is not None:
        # Historical source wins
        return PredictionResponse(
            date=dt.date().isoformat(),
            source="historical",
            Cases=hist_values["cases"],
            trucks=hist_values["trucks"],
            raw_extracted=extracted,
        )

    # 4. Otherwise, build feature row using means + extracted info
    try:
        feature_row = build_feature_row(extracted, dt)
        preds = predict_with_models(feature_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return PredictionResponse(
        date=dt.date().isoformat(),
        source="model",
        Cases=preds["cases"],
        trucks=preds["trucks"],
        raw_extracted=extracted,
    )
