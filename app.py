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
from dateutil import parser as date_parser
from google import genai

# ==============================
# Gemini client
# ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")

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
# FastAPI setup (docs explicitly enabled)
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
    source: str  # "historical" or "model"
    Cases: float
    trucks: float
    raw_extracted: Dict[str, Any]


# ==============================
# Helpers
# ==============================

def infer_date_from_text(text: str) -> Optional[str]:
    """Fuzzy parse any date string like 'jan 1 2025' -> '2025-01-01'."""
    try:
        dt = date_parser.parse(text, fuzzy=True, dayfirst=False)
        return dt.date().isoformat()
    except Exception:
        return None


STATE_MAP = {
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


def normalize_state_name(raw_state: Optional[str], full_text: str) -> Optional[str]:
    """Ensure state_name is 2-letter code. Use Gemini output OR fallback to text scan."""
    if raw_state:
        s = str(raw_state).strip()
        if len(s) == 2:
            return s.upper()
        low = s.lower()
        if low in STATE_MAP:
            return STATE_MAP[low]

    lower_text = full_text.lower()
    for full_name, code in STATE_MAP.items():
        if full_name in lower_text:
            return code

    for token in re.split(r"\W+", full_text.upper()):
        if token in STATE_MAP.values():
            return token

    return None


# ==============================
# Local fallback extractor
# ==============================

def extract_features_local(user_query: str) -> Dict[str, Any]:
    """
    Fallback extraction:
      - JSON string or key:value / key=value pairs.
      - Fuzzy date parsing.
      - State code mapping.
    """
    s = (user_query or "").strip()
    if not s:
        raise ValueError("Empty query provided for extraction")

    if s.startswith("{"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON extraction string: {e}") from e
    else:
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

    # Date: ISO first, then fuzzy
    if "dt" not in data or not data.get("dt"):
        iso_match = re.search(r"\d{4}-\d{2}-\d{2}", s)
        if iso_match:
            data["dt"] = iso_match.group(0)
        else:
            inferred = infer_date_from_text(s)
            if inferred:
                data["dt"] = inferred

    state = data.get("state_name")
    data["state_name"] = normalize_state_name(state, s)

    return data


# ==============================
# Gemini extraction
# ==============================

EXTRACTION_SYSTEM_PROMPT = """
You are a strict JSON information extractor for a retail logistics forecasting system.

Given a natural language query about trucks and cases,
you MUST return a single JSON object with exactly these keys:

- "dt": date in ISO format "YYYY-MM-DD" (if the user mentions a date in any format, convert it;
  otherwise infer from context or leave null)
- "state_name": 2-letter US state code like "MD", "NH", "CA" (uppercase). If not mentioned, use null.
- "store_id": integer store id if mentioned, otherwise null
- "dept_id": integer department id if mentioned, otherwise null
- "gmm_name": string or null
- "dmm_name": string or null
- "dept_desc": string or null
- "day_of_week": string name like "Monday", "Tuesday" etc or null
- "holiday_name": string holiday name like "Christmas", "Easter", "Labor Day" etc or null
- "dept_near_holiday_5": 0 or 1 or null
- "dept_near_holiday_10": 0 or 1 or null
- "dept_weekday": string categorization like "weekday" or "weekend" or null
- "dept_holiday_interact": string like "holiday", "non_holiday" or null

Rules:
- Output ONLY raw JSON, no backticks, no prose.
- If something is not specified and you cannot safely infer it, set it to null.
- "state_name" MUST be a 2-letter code if you can infer the state (e.g., Maryland -> "MD").
"""


def extract_features_with_gemini(user_query: str) -> Dict[str, Any]:
    prompt = EXTRACTION_SYSTEM_PROMPT + "\n\nUser query:\n" + user_query

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[{"role": "user", "parts": [prompt]}],
    )

    if not resp or not resp.candidates:
        raise ValueError("Gemini returned no candidates")

    text = resp.candidates[0].content.parts[0].text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Gemini extraction not valid JSON: {text}")
        json_str = text[start: end + 1]
        data = json.loads(json_str)

    if not isinstance(data, dict):
        raise ValueError("Gemini output is not a JSON object")

    keys = [
        "dt", "state_name", "store_id", "dept_id", "gmm_name", "dmm_name",
        "dept_desc", "day_of_week", "holiday_name", "dept_near_holiday_5",
        "dept_near_holiday_10", "dept_weekday", "dept_holiday_interact",
    ]
    for k in keys:
        data.setdefault(k, None)

    data["state_name"] = normalize_state_name(data.get("state_name"), user_query)

    if not data.get("dt"):
        inferred = infer_date_from_text(user_query)
        if inferred:
            data["dt"] = inferred

    return data


# ==============================
# Feature building utilities
# ==============================

def parse_date(dt_str: Optional[str]) -> datetime:
    """
    Parse a date string from Gemini or local extractor.

    Handles:
    - ISO '2025-01-01'
    - Natural 'Jan 1 2025', etc.
    """
    if dt_str is None:
        raise ValueError("Extracted date (dt) is null")

    dt_str = str(dt_str).strip()

    # Try ISO first
    try:
        return datetime.fromisoformat(dt_str[:10])
    except Exception:
        pass

    # Fuzzy fallback
    try:
        dt = date_parser.parse(dt_str, fuzzy=True, dayfirst=False)
        return dt
    except Exception as e:
        raise ValueError(f"Could not parse date string: {dt_str}") from e


def get_historical_values_if_available(dt: datetime) -> Optional[Dict[str, float]]:
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
    row: Dict[str, Any] = {}
    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0

    for col in NUMERIC_FEATURES:
        if col == "is_weekend":
            continue

        value = extracted.get(col, None)
        if value is None:
            if col in NUMERIC_FEATURE_MEANS:
                value = NUMERIC_FEATURE_MEANS[col]
            else:
                value = 0.0

        row[col] = value

    for col in CATEGORICAL_FEATURES:
        row[col] = extracted.get(col, None)

    return pd.DataFrame([row])


def predict_with_models(feature_row: pd.DataFrame) -> Dict[str, float]:
    trucks_pred = float(trucks_model.predict(feature_row)[0])
    cases_pred = float(cases_model.predict(feature_row)[0])

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
    # 1. Try Gemini extraction, fallback to local if something goes wrong
    try:
        extracted = extract_features_with_gemini(req.query)
    except Exception:
        extracted = extract_features_local(req.query)

    # 2. Parse date
    try:
        dt = parse_date(extracted.get("dt"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date from query: {e}")

    # 3. Check historical data
    hist_values = get_historical_values_if_available(dt)
    if hist_values is not None:
        return PredictionResponse(
            date=dt.date().isoformat(),
            source="historical",
            Cases=hist_values["cases"],
            trucks=hist_values["trucks"],
            raw_extracted=extracted,
        )

    # 4. Model prediction
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
