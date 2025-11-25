# app.py
import os
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor
from dateutil import parser as dateparser

# ==============================
# Gemini SDK setup (optional)
# ==============================
try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    gemini_model = None  # will raise at runtime if used


# ==============================
# Paths and constants
# ==============================
path_csv = "https://drive.google.com/file/d/1hlt8iiILix5PVGoQJjXdtqdULXwRYkKQ/view?usp=drive_link"
CSV_PATH = "/Users/emilgeorgemathew/Documents/Wallmart/merged_data_new.csv"
CASES_MODEL_PATH = "best_cases_catboost_ts_cv.cbm"
TRUCKS_MODEL_PATH = "best_trucks_ts_cv.pkl"

# These MUST match what you used in training
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

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==============================
# Pydantic models
# ==============================
class PredictOverrides(BaseModel):
    # Free-form overrides placeholder if you want to add things later
    additionalProp1: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    query: str
    overrides: Optional[PredictOverrides] = None


class PredictResponse(BaseModel):
    cases: float
    trucks: float
    metadata: Optional[Dict[str, Any]] = None


# ==============================
# Load data and models at startup
# ==============================
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Ensure dt is date-like so equality works
if "dt" in df.columns:
    df["dt"] = pd.to_datetime(df["dt"]).dt.date

print("Loading CatBoost CASES model...")
cases_model = CatBoostRegressor()
cases_model.load_model(CASES_MODEL_PATH)

print("Loading TRUCKS model (RandomForest)...")
trucks_model = joblib.load(TRUCKS_MODEL_PATH)


# ==============================
# Helper: call Gemini to parse query
# ==============================
def extract_params_from_query(query: str) -> Dict[str, Any]:
    """
    Extracts date/time (and optionally store/dept) from the user's query.
    Uses Gemini when available; otherwise falls back to dateutil fuzzy parsing.
    """
    # Use Gemini if configured
    if gemini_model is not None:
        prompt = (
            "You are a strict JSON parser.\n\n"
            "Extract the following fields from the user query.\n\n"
            "User query:\n"
            f'"""{query}"""\n\n'
            "Return ONLY valid JSON with this exact structure (no extra text):\n\n"
            "{\n"
            '  "date": "YYYY-MM-DD or null if unknown",\n'
            '  "time": "HH:MM or null if not provided",\n'
            '  "store_id": "store id as string or null",\n'
            '  "dept_id": "department id as string or null"\n'
            "}\n"
        )

        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        try:
            if "{" in text and "}" in text:
                text = text[text.find("{"): text.rfind("}") + 1]
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Could not parse JSON from Gemini response: {text}") from e

        date_val = data.get("date")
        time_val = data.get("time")
        store_id = data.get("store_id")
        dept_id = data.get("dept_id")

        return {
            "date": date_val if date_val not in ("", "null", None) else None,
            "time": time_val if time_val not in ("", "null", None) else None,
            "store_id": str(store_id) if store_id not in ("", "null", None) else None,
            "dept_id": str(dept_id) if dept_id not in ("", "null", None) else None,
        }

    # Fallback: use dateutil fuzzy parsing to extract a date/time
    try:
        dt = dateparser.parse(query, fuzzy=True)
        if dt:
            parsed_date = dt.date().isoformat()
            parsed_time = None
            if not (dt.hour == 0 and dt.minute == 0 and dt.second == 0):
                parsed_time = f"{dt.hour:02d}:{dt.minute:02d}"
            return {"date": parsed_date, "time": parsed_time, "store_id": None, "dept_id": None}
    except Exception:
        pass

    return {"date": None, "time": None, "store_id": None, "dept_id": None}


# ==============================
# Helper: build feature row
# ==============================
def get_feature_row(parsed: Dict[str, Any]) -> pd.Series:
    """
    Given parsed {date, store_id, dept_id}, pull the matching row
    from df and return the features in the correct order.
    """
    if parsed["date"] is None:
        raise HTTPException(status_code=400, detail="Could not extract date from query.")
    if parsed["store_id"] is None:
        raise HTTPException(status_code=400, detail="Could not extract store_id from query.")
    if parsed["dept_id"] is None:
        raise HTTPException(status_code=400, detail="Could not extract dept_id from query.")

    try:
        date_val = pd.to_datetime(parsed["date"]).date()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {parsed['date']}")

    # Filter the row
    subset = df[
        (df["dt"] == date_val) &
        (df["store_id"].astype(str) == parsed["store_id"]) &
        (df["dept_id"].astype(str) == parsed["dept_id"])
    ]

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No matching row found in dataset for date={parsed['date']}, "
                   f"store_id={parsed['store_id']}, dept_id={parsed['dept_id']}"
        )

    row = subset.iloc[0]

    # Ensure all required feature columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in subset.columns]
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset missing required feature columns: {missing_cols}"
        )

    # Fill NAs and return
    features = row[FEATURE_COLS].fillna(0)
    return features


# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="Prediction API",
    version="0.1.0",
    description="Predicts cases and trucks from a natural language query.",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # 1. Use Gemini (or fail if not configured) to parse the query
    parsed = extract_params_from_query(req.query)

    # 2. Build feature vector from CSV
    feature_row = get_feature_row(parsed)
    X = feature_row.values.reshape(1, -1).astype(np.float32)

    # 3. Predict with best models
    cases_pred = float(cases_model.predict(X)[0])
    trucks_pred = float(trucks_model.predict(X)[0])

    # 4. Return response
    return PredictResponse(
        cases=cases_pred,
        trucks=trucks_pred,
        metadata={"parsed": parsed}
    )


# ==============================
# Main entrypoint for local run
# ==============================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run("app:app", host=host, port=port, reload=True)
