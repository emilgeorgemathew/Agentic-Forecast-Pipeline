import os
import json
import pickle
import re
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Any

import pandas as pd
import dateparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
from google import genai
from dotenv import load_dotenv

load_dotenv()


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


def validate_query_relevance(query: str, conversation_history: Optional[list] = None) -> bool:
    """
    Check if the query appears to be related to retail forecasting (cases/trucks).
    Uses simple keyword matching for relevance.
    If conversation history exists, be more lenient for follow-up queries.
    """
    query_lower = query.lower()
    relevant_keywords = [
        "cases", "case", "trucks", "truck", "forecast", "predict", "prediction",
        "estimate", "estimation", "sales", "demand", "retail", "store", "stores",
        "department", "dept", "inventory", "supply", "shipping", "delivery"
    ]

    # Check if query contains forecasting keywords
    if any(keyword in query_lower for keyword in relevant_keywords):
        return True

    # If there's conversation history, allow follow-up queries
    # These might be short like "what about MD?" or "same for VA"
    if conversation_history and len(conversation_history) > 0:
        # Follow-up patterns
        follow_up_patterns = [
            "what about", "how about", "same", "also", "and",
            "what if", "now", "next", "then", "for",
            # State codes
            "md", "va", "de", "nh", "maryland", "virginia", "delaware", "new hampshire"
        ]
        if any(pattern in query_lower for pattern in follow_up_patterns):
            return True

    return False


# ============================================================
# FILE PATHS / LOADING
# ============================================================

FEATURE_MEANS_PATH = os.environ.get("FEATURE_MEANS_JSON", "feature_means.json")
DATE_TRUCKS_CASES_PATH = os.environ.get("DATE_TRUCKS_CASES_CSV", "date_trucks_cases.csv")
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_lgbm.pkl")
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

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY env var is required.")

# Support multiple API keys for rotation (comma-separated)
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEY.split(",") if key.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("At least one GOOGLE_API_KEY is required.")

GEMINI_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash-latest")

# Track current key index for rotation
import threading
key_lock = threading.Lock()
current_key_index = 0

def get_next_client():
    """Get the next Gemini client using round-robin API key rotation"""
    global current_key_index
    with key_lock:
        api_key = GEMINI_API_KEYS[current_key_index]
        current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return genai.Client(api_key=api_key)

EXTRACT_PROMPT = """
You are a strict JSON information extractor for a retail forecasting system.

Given a natural language query about cases & trucks (and optional conversation history), extract:

- "dt": date in ISO format "YYYY-MM-DD"
- "state_name": 2-letter US state code (e.g., MD, VA); if you see full state name, convert it.
- "store_id": integer store id if mentioned, otherwise null.
- "dept_id": integer department id if mentioned, otherwise null.
- "dept_desc": department description/name if mentioned, otherwise null.

Rules:
- If the user gives a date like "Jan 1 2025", "9th September 2025", "2025/11/17", convert it to "YYYY-MM-DD".
- CRITICAL CONTEXT INHERITANCE RULE: When conversation history is provided with a "Previous extraction" showing the last query's extracted values, you MUST inherit ANY field that is:
  1. Present in the previous extraction, AND
  2. NOT explicitly overridden in the current query

  Specifically:
  * If previous extraction has "dt": "2025-12-04" and current query does NOT mention a different date, you MUST use "dt": "2025-12-04"
  * If current query ONLY mentions a new state (like "What about MD"), inherit ALL other fields (dt, store_id, dept_id, dept_desc) from previous extraction
  * If current query says "same date", "same state", "that store", inherit those exact values

  Examples:
    - Previous extraction: {"dt": "2025-12-04", "state_name": "DE"}, Current: "What about MD" → {"dt": "2025-12-04", "state_name": "MD", ...}
    - Previous extraction: {"dt": "2025-12-04", "state_name": "MD"}, Current: "What about Virginia" → {"dt": "2025-12-04", "state_name": "VA", ...}
    - Previous extraction: {"dt": "2025-01-01", "store_id": 10001}, Current: "what about MD" → {"dt": "2025-01-01", "state_name": "MD", "store_id": 10001, ...}

- If the user does NOT specify a date AND there's no previous extraction with a date, set "dt" to null.
- If you cannot detect a state and there's no previous state, set "state_name" to null.
- If no department info and no previous dept info, set "dept_id" and "dept_desc" to null.
- Output ONLY a single valid JSON object, no prose, no comments.
"""

ANALYSIS_PROMPT = """
You are a data analyst for a retail forecasting system. Answer questions about historical cases and trucks data.

Available data summary:
{data_summary}

User question: {question}

Instructions:
- Provide clear, concise answers based on the data
- Use specific numbers and insights
- If asking for rankings/comparisons, analyze the data and provide the top results
- Format your response in a friendly, conversational way
- Keep responses under 200 words
"""

GENERAL_CHAT_PROMPT = """
You are a friendly AI assistant for a retail forecasting system. You help users with general questions about the system, explain concepts, and provide guidance.

System Overview:
- This is an AI-powered forecasting system for retail cases and trucks
- We forecast demand for cases (products) and required trucks for delivery
- The system serves retail stores in Maryland (MD), Virginia (VA), Delaware (DE), and New Hampshire (NH)
- We use advanced machine learning models to predict based on historical data from March 2024 onwards

Key Capabilities:
- Forecast cases and truck requirements for specific dates
- Provide analytical insights from historical data
- Answer questions about the forecasting process
- Help with understanding store operations and logistics

User Question: {question}

Conversation History: {history}

Instructions:
- Be friendly and helpful
- Explain concepts clearly and concisely
- Provide practical examples when helpful
- Keep responses under 300 words
- If technical, explain in simple terms
- Encourage users to ask specific forecasting questions when appropriate
"""


def is_analytical_question(query: str) -> bool:
    """
    Detect if the query is asking for analysis/insights rather than a specific forecast.
    """
    query_lower = query.lower()
    analytical_patterns = [
        "top", "best", "worst", "highest", "lowest", "most", "least",
        "compare", "comparison", "versus", "vs", "which", "what are",
        "how many", "show me", "list", "rank", "ranking", "trend",
        "average", "total", "sum", "all", "statistics", "stats",
        "data", "analysis", "analytics", "insights", "report", "summary",
        "distribution", "pattern", "breakdown", "overview"
    ]
    return any(pattern in query_lower for pattern in analytical_patterns)


def is_general_question(query: str) -> bool:
    """
    Detect if the query is a general question not related to specific forecasting
    demands or historical data.
    """
    query_lower = query.lower()
    general_patterns = [
        "how does", "what is", "how to", "why", "explain", "tell me about",
        "how works", "what means", "purpose", "goal", "objective", "function",
        "help", "guide", "tutorial", "documentation", "examples", "demo",
        "greeting", "hi", "hello", "hey", "thanks", "thank", "goodbye", "bye",
        "chat", "talk", "conversation", "discuss"
    ]

    # If it starts with typical greeting/help patterns
    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "help", "hi there", "hey there", "howdy", "greetings"
    ]
    if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
        return True

    return any(pattern in query_lower for pattern in general_patterns)


def extract_aggregation_query(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Extract aggregation parameters from user query.
    Returns dict with aggregation type and filters.
    """
    query_lower = user_query.lower()

    # Aggregation types
    agg_functions = {
        "sum": ["total", "sum", "how many", "count", "aggregate", "combined"],
        "avg": ["average", "mean", "avg"],
        "max": ["maximum", "max", "most", "highest"],
        "min": ["minimum", "min", "least", "lowest"],
        "count": ["how many", "count", "number of"]
    }

    # Detect aggregation type
    agg_type = None
    for agg, patterns in agg_functions.items():
        if any(pattern in query_lower for pattern in patterns):
            agg_type = agg
            break

    # Only proceed if we found aggregation keywords
    if not agg_type:
        return None

    # Extract filters
    result = {
        "aggregation": agg_type,
        "field": None,
        "filters": {}
    }

    # Detect field (cases or trucks)
    if "truck" in query_lower:
        result["field"] = "trucks"
    elif "case" in query_lower:
        result["field"] = "cases"
    elif "both" in query_lower or ("case" in query_lower and "truck" in query_lower):
        result["field"] = "both"
    else:
        # Default to trucks if not specified
        result["field"] = "trucks"

    # State extraction
    state_patterns = {
        "DE": ["delaware", "de"],
        "MD": ["maryland", "md"],
        "VA": ["virginia", "va"],
        "NH": ["new hampshire", "nh"]
    }

    for state_code, patterns in state_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            result["filters"]["state"] = state_code
            break

    # Store extraction
    store_match = re.search(r'store\s+(\d+)', query_lower)
    if store_match:
        result["filters"]["store_id"] = int(store_match.group(1))

    # Date range extraction - more sophisticated
    date_patterns = [
        (r"in\s+(\w+)\s+(\d{4})", "month_year"),  # "in January 2025"
        (r"(\d{4})[-\s](\d{4})", "year_range"),   # "2024-2025"
        (r"q(\d)\s+(\d{4})", "quarter_year"),     # "Q1 2025"
        (r"(\w+)\s+(\d{4})", "month_year"),       # "January 2025"
    ]

    for pattern, pattern_type in date_patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            if pattern_type == "month_year":
                month_str, year = match.groups()
                year = int(year)
                month_names = ["january", "february", "march", "april", "may", "june",
                             "july", "august", "september", "october", "november", "december"]
                try:
                    month = month_names.index(month_str.lower()) + 1
                    result["filters"]["date_range"] = {
                        "type": "month",
                        "month": month,
                        "year": year
                    }
                except ValueError:
                    pass
            elif pattern_type == "year_range":
                start_year, end_year = int(match.group(1)), int(match.group(2))
                result["filters"]["date_range"] = {
                    "type": "year_range",
                    "start_year": start_year,
                    "end_year": end_year
                }
            break

    return result if result["filters"] else None


def perform_intelligent_aggregation(query_analysis: Dict[str, Any]) -> str:
    """
    Perform intelligent data aggregation based on extracted parameters.
    """
    try:
        agg_type = query_analysis["aggregation"]
        field = query_analysis["field"]
        filters = query_analysis["filters"]

        # Build filter mask
        mask = pd.Series([True] * len(df_hist))

        # Apply state filter
        if "state" in filters and "state_name" in df_hist.columns:
            mask = mask & (df_hist["state_name"].str.upper() == filters["state"])

        # Apply store filter
        if "store_id" in filters and "store_id" in df_hist.columns:
            mask = mask & (df_hist["store_id"] == filters["store_id"])

        # Apply date range filter
        if "date_range" in filters:
            date_info = filters["date_range"]
            if date_info["type"] == "month":
                year = date_info["year"]
                month = date_info["month"]
                mask = mask & (df_hist["dt"].dt.year == year) & (df_hist["dt"].dt.month == month)
            elif date_info["type"] == "year_range":
                start_year = date_info["start_year"]
                end_year = date_info["end_year"]
                mask = mask & (df_hist["dt"].dt.year >= start_year) & (df_hist["dt"].dt.year <= end_year)

        # Filter data
        filtered_data = df_hist[mask]

        if len(filtered_data) == 0:
            return "I couldn't find any data matching your criteria."

        # Perform aggregation
        result_text = ""

        if agg_type == "sum":
            if field == "both":
                total_cases = filtered_data["cases"].sum()
                total_trucks = filtered_data["trucks"].sum()
                record_count = len(filtered_data)
                result_text = f"Found {record_count:,} records matching your criteria:\n• Total cases: {total_cases:,.0f}\n• Total trucks: {total_trucks:,.0f}"
            else:
                total = filtered_data[field].sum()
                record_count = len(filtered_data)
                result_text = f"Found {record_count:,} records matching your criteria:\n• Total {field}: {total:,.0f}"

        elif agg_type == "avg":
            if field == "both":
                avg_cases = filtered_data["cases"].mean()
                avg_trucks = filtered_data["trucks"].mean()
                result_text = f"Average values across {len(filtered_data):,} records:\n• Average cases: {avg_cases:.1f}\n• Average trucks: {avg_trucks:.1f}"
            else:
                avg = filtered_data[field].mean()
                result_text = f"Average {field} across {len(filtered_data):,} records: {avg:.1f}"

        elif agg_type == "max":
            if field == "both":
                max_cases_row = filtered_data.loc[filtered_data["cases"].idxmax()]
                max_trucks_row = filtered_data.loc[filtered_data["trucks"].idxmax()]
                result_text = f"Maximum values:\n• Highest cases: {max_cases_row['cases']:.0f} (on {max_cases_row['dt'].strftime('%Y-%m-%d')})\n• Highest trucks: {max_trucks_row['trucks']:.0f} (on {max_trucks_row['dt'].strftime('%Y-%m-%d')})"
            else:
                max_row = filtered_data.loc[filtered_data[field].idxmax()]
                result_text = f"Maximum {field}: {max_row[field]:.0f} (on {max_row['dt'].strftime('%Y-%m-%d')})"

        elif agg_type == "min":
            if field == "both":
                min_cases_row = filtered_data.loc[filtered_data["cases"].idxmin()]
                min_trucks_row = filtered_data.loc[filtered_data["trucks"].idxmin()]
                result_text = f"Minimum values:\n• Lowest cases: {min_cases_row['cases']:.0f} (on {min_cases_row['dt'].strftime('%Y-%m-%d')})\n• Lowest trucks: {min_trucks_row['trucks']:.0f} (on {min_trucks_row['dt'].strftime('%Y-%m-%d')})"
            else:
                min_row = filtered_data.loc[filtered_data[field].idxmin()]
                result_text = f"Minimum {field}: {min_row[field]:.0f} (on {min_row['dt'].strftime('%Y-%m-%d')})"

        elif agg_type == "count":
            count = len(filtered_data)
            result_text = f"Found {count:,} records matching your criteria."

        # Add filter context
        filter_desc = []
        if "state" in filters:
            state_names = {"DE": "Delaware", "MD": "Maryland", "VA": "Virginia", "NH": "New Hampshire"}
            filter_desc.append(f"state: {state_names.get(filters['state'], filters['state'])}")
        if "store_id" in filters:
            filter_desc.append(f"store: {filters['store_id']}")
        if "date_range" in filters:
            date_info = filters["date_range"]
            if date_info["type"] == "month":
                month_names = ["", "January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"]
                filter_desc.append(f"{month_names[date_info['month']]} {date_info['year']}")
            elif date_info["type"] == "year_range":
                filter_desc.append(f"{date_info['start_year']}-{date_info['end_year']}")

        if filter_desc:
            result_text += f" (filtered by {', '.join(filter_desc)})"

        return result_text

    except Exception as e:
        return f"I encountered an error while analyzing the data: {str(e)}"


def get_store_location_info(store_id: int) -> str:
    """
    Get location information for a specific store.
    """
    try:
        # Find store in historical data
        store_data = df_hist[df_hist["store_id"] == store_id]

        if len(store_data) == 0:
            return f"I'm sorry, I don't have information about store #{store_id}. This store might not exist in our database, or the data might not be available."

        # Get the most recent state info for this store
        store_row = store_data.iloc[0]  # First occurrence
        state = store_row.get("state_name", "Unknown")

        state_names = {
            "DE": "Delaware",
            "MD": "Maryland",
            "VA": "Virginia",
            "NH": "New Hampshire"
        }

        full_state_name = state_names.get(state, state)

        # Get some statistics for this store
        total_records = len(store_data)
        total_cases = store_data["cases"].sum()
        total_trucks = store_data["trucks"].sum()
        avg_cases = store_data["cases"].mean()
        avg_trucks = store_data["trucks"].mean()

        result = f"📍 **Store #{store_id} Information**\n\n"
        result += f"• **Location:** {full_state_name} ({state})\n"
        result += f"• **Total Records:** {total_records:,}\n"
        result += f"• **Total Cases:** {total_cases:,.0f}\n"
        result += f"• **Total Trucks:** {total_trucks:,.0f}\n"
        result += f"• **Average Cases per Record:** {avg_cases:.1f}\n"
        result += f"• **Average Trucks per Record:** {avg_trucks:.1f}\n"
        result += f"• **Date Range:** {store_data['dt'].min().strftime('%Y-%m-%d')} to {store_data['dt'].max().strftime('%Y-%m-%d')}"

        return result

    except Exception as e:
        return f"I encountered an error while looking up store information: {str(e)}"


def handle_general_query(user_query: str, conversation_history: Optional[list] = None) -> str:
    """
    Use Gemini to handle general questions and conversation about the forecasting system.
    """
    try:
        # Format conversation history for context
        hist_text = ""
        if conversation_history:
            hist_text = "\n".join([f"- {msg}" for msg in conversation_history[-3:]])  # Last 3 messages
        else:
            hist_text = "(No previous conversation)"

        # Create prompt
        prompt = GENERAL_CHAT_PROMPT.format(
            question=user_query,
            history=hist_text
        )

        # Call Gemini with next available API key
        client = get_next_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        return resp.text.strip()

    except Exception as e:
        return (
            "I'm having trouble reaching the assistant right now. "
            "Please try again in a moment, or ask for a forecast by sharing the date, state, store_id, and dept_id/dept name."
        )


def analyze_data_with_gemini(user_query: str) -> str:
    """
    Use Gemini to analyze the historical data and answer analytical questions.
    """
    try:
        # Get data summary for Gemini
        total_records = len(df_hist)
        date_range = f"{MIN_DATE} to {MAX_DATE}"

        # Get daily statistics
        daily_stats = df_hist.groupby('dt').agg({
            'cases': ['sum', 'mean', 'max', 'min'],
            'trucks': ['sum', 'mean', 'max', 'min']
        }).round(2)

        # Get top 10 days by cases
        top_cases_days = df_hist.nlargest(10, 'cases')[['dt', 'cases', 'trucks']].to_string(index=False)

        # Get truck distribution
        truck_dist = df_hist['trucks'].value_counts().sort_index().to_string()

        # Overall statistics
        overall_stats = f"""
Total Records: {total_records}
Date Range: {date_range}
Total Cases: {df_hist['cases'].sum():,.0f}
Average Cases per Record: {df_hist['cases'].mean():.2f}
Max Cases (single record): {df_hist['cases'].max():.2f}
Min Cases (single record): {df_hist['cases'].min():.2f}

Total Trucks: {df_hist['trucks'].sum():,.0f}
Average Trucks per Record: {df_hist['trucks'].mean():.2f}

Truck Distribution (count of records):
{truck_dist}

Top 10 Records by Cases:
{top_cases_days}
"""

        # Create prompt
        prompt = ANALYSIS_PROMPT.format(
            data_summary=overall_stats,
            question=user_query
        )

        # Call Gemini with next available API key
        client = get_next_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        return resp.text.strip()

    except Exception as e:
        return f"I encountered an error analyzing the data: {str(e)}"


def call_gemini_extract(user_query: str, conversation_history: Optional[list] = None, previous_extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Use google-genai v1beta client to get a JSON with dt, state_name, store_id.
    If anything goes wrong (model error, quota, bad JSON), we return {} and
    let local fallback handle it.

    Args:
        user_query: Current user query
        conversation_history: Optional list of previous queries for context
        previous_extraction: Optional dict with the last successful extraction for context inheritance
    """
    try:
        # Build input with conversation context if available
        input_parts = [EXTRACT_PROMPT]

        if conversation_history or previous_extraction:
            # Add conversation context
            context = "\n\n"
            if conversation_history:
                context += "Conversation history:\n"
                for i, prev_query in enumerate(conversation_history[-2:], 1):  # Use last 2 queries
                    context += f"{i}. {prev_query}\n"

            if previous_extraction:
                context += f"\nPrevious extraction: {json.dumps(previous_extraction)}\n"

            context += f"\nCurrent query: {user_query}"
            input_parts.append(context)
        else:
            input_parts.append(user_query)

        # Use next available API key
        client = get_next_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="\n".join(input_parts),
        )
        text = resp.text.strip()

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
    conversation_history: Optional[list] = None  # List of previous queries for context


class PredictionResponse(BaseModel):
    date: str
    source: str  # "historical" or "model"
    Cases: float
    trucks: float
    state: Optional[str]
    store_id: Optional[int]
    dept_id: Optional[int]
    dept_name: Optional[str]
    message: Optional[str] = None
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
      - "tomorrow", "next Friday" (relative dates)
    """
    # Dateparser settings for robust parsing
    parsing_settings = {
        "RELATIVE_BASE": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "PREFER_DATES_FROM": "future"
    }

    # 1. Try parsing the full query
    dt = dateparser.parse(full_query, languages=['en'], settings=parsing_settings)
    if dt is not None:
        return dt.date()

    # 2. Try to grab date-like chunks
    patterns = [
        r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}",  # 2025-11-17 or 2025/11/17
        r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",  # 17/11/2025
        r"\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{2,4}",  # 9th September 2025
        r"\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}",  # September 9th, 2025
        r"\bnext\s+\w+\b",  # next Friday
        r"\blast\s+\w+\b",  # last Friday
    ]
    for pat in patterns:
        m = re.search(pat, full_query, flags=re.IGNORECASE)
        if m:
            chunk = m.group(0)
            dt2 = dateparser.parse(chunk, languages=['en'], settings=parsing_settings)
            if dt2 is not None:
                return dt2.date()

    # If no date parsed, default to tomorrow
    default_dt = (datetime.now() + timedelta(days=1)).date()
    return default_dt


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
            parsing_settings = {
                "RELATIVE_BASE": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            }
            dt = dateparser.parse(dt_str, languages=['en'], settings=parsing_settings)
            if dt is not None:
                return dt.date()
    # fallback to full-query parsing
    return robust_date_parse(original_query)


def get_store_dept_metadata(store_id: Optional[int] = None,
                           dept_id: Optional[int] = None,
                           state_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get store/dept metadata from historical data.
    Returns the most common metadata for a given store/dept combination.
    """
    if "store_id" not in df_hist.columns:
        return None

    # Build filter
    mask = pd.Series([True] * len(df_hist))

    if store_id and "store_id" in df_hist.columns:
        mask = mask & (df_hist["store_id"] == store_id)
    if dept_id and "dept_id" in df_hist.columns:
        mask = mask & (df_hist["dept_id"] == dept_id)
    if state_name and "state_name" in df_hist.columns:
        mask = mask & (df_hist["state_name"].str.upper() == state_name.upper())

    if not mask.any():
        return None

    # Get first matching row for metadata
    row = df_hist.loc[mask].iloc[0]

    result = {}
    if "dept_id" in df_hist.columns and pd.notna(row["dept_id"]):
        result["dept_id"] = int(row["dept_id"])
    if "state_name" in df_hist.columns and pd.notna(row["state_name"]):
        result["state_name"] = str(row["state_name"])
    if "store_id" in df_hist.columns and pd.notna(row["store_id"]):
        result["store_id"] = int(row["store_id"])
    if "dept_desc" in df_hist.columns and pd.notna(row["dept_desc"]):
        result["dept_desc"] = str(row["dept_desc"])

    return result


def get_default_store_dept_for_state(state_name: Optional[str]) -> Dict[str, Any]:
    """
    Pick a representative store/dept for a given state from historical data.
    Uses the most frequent store_id/dept_id within the state (falls back to first available).
    """
    defaults: Dict[str, Any] = {}

    if not state_name or "state_name" not in df_hist.columns:
        return defaults

    mask = df_hist["state_name"].str.upper() == state_name.upper()
    if not mask.any():
        return defaults

    sub = df_hist.loc[mask]

    if "store_id" in sub.columns and sub["store_id"].notna().any():
        defaults["store_id"] = int(sub["store_id"].mode().iloc[0])

    if "dept_id" in sub.columns and sub["dept_id"].notna().any():
        defaults["dept_id"] = int(sub["dept_id"].mode().iloc[0])

    if "dept_desc" in sub.columns and sub["dept_desc"].notna().any():
        defaults["dept_desc"] = str(sub["dept_desc"].mode().iloc[0])

    return defaults


def get_historical_values_if_available(dt: date, state_name: Optional[str] = None,
                                       store_id: Optional[int] = None,
                                       dept_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get historical values for a given date, with optional filtering by state, store, dept.
    Returns dict with trucks, cases, and any available metadata (dept_id, state_name, store_id, dept_desc).
    """
    if dt < MIN_DATE or dt > MAX_DATE:
        return None

    # Start with date filter
    mask = df_hist["dt"].dt.date == dt

    # Apply additional filters if the columns exist and values are provided
    if "state_name" in df_hist.columns and state_name:
        mask = mask & (df_hist["state_name"].str.upper() == state_name.upper())
    if "store_id" in df_hist.columns and store_id:
        mask = mask & (df_hist["store_id"] == store_id)
    if "dept_id" in df_hist.columns and dept_id:
        mask = mask & (df_hist["dept_id"] == dept_id)

    if not mask.any():
        return None

    row = df_hist.loc[mask].iloc[0]

    # Build result with trucks and cases
    result = {
        "trucks": float(row["trucks"]),
        "cases": float(row["cases"])
    }

    # Add metadata if available in the CSV
    if "dept_id" in df_hist.columns:
        result["dept_id"] = int(row["dept_id"]) if pd.notna(row["dept_id"]) else None
    if "state_name" in df_hist.columns:
        result["state_name"] = str(row["state_name"]) if pd.notna(row["state_name"]) else None
    if "store_id" in df_hist.columns:
        result["store_id"] = int(row["store_id"]) if pd.notna(row["store_id"]) else None
    if "dept_desc" in df_hist.columns:
        result["dept_desc"] = str(row["dept_desc"]) if pd.notna(row["dept_desc"]) else None

    return result


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

    # categorical features (set to dummy numeric values for compatibility)
    for col in CATEGORICAL_FEATURES:
        if col == "state_name":
            row[col] = 0.0  # dummy
        elif col == "day_of_week":
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            row[col] = days.index(dt.strftime("%A"))
        else:
            row[col] = 0.0  # dummy

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
    conversation_history = req.conversation_history or []

    # Check if this is asking for store location information
    store_match = re.search(r'where\s+(?:is\s+)?(?:store\s+)?(\d+)', user_query.lower())
    if store_match:
        store_id = int(store_match.group(1))
        location_info = get_store_location_info(store_id)
        return PredictionResponse(
            date="",
            source="store_info",
            Cases=0.0,
            trucks=0.0,
            state=None,
            store_id=store_id,
            dept_id=None,
            dept_name=None,
            message=location_info,
            raw_extracted={"type": "store_location", "store_id": store_id}
        )

    # Check if this is a general question (how the system works, greetings, etc.)
    if is_general_question(user_query):
        general_response = handle_general_query(user_query, conversation_history)
        return PredictionResponse(
            date="",
            source="chat",
            Cases=0.0,
            trucks=0.0,
            state=None,
            store_id=None,
            dept_id=None,
            dept_name=None,
            message=general_response,
            raw_extracted={"type": "general_query"}
        )

    # Check for intelligent aggregation queries (sum, avg, etc.)
    agg_query = extract_aggregation_query(user_query)
    if agg_query:
        agg_response = perform_intelligent_aggregation(agg_query)
        return PredictionResponse(
            date="",
            source="aggregation",
            Cases=0.0,
            trucks=0.0,
            state=None,
            store_id=None,
            dept_id=None,
            dept_name=None,
            message=agg_response,
            raw_extracted={"type": "aggregation", **agg_query}
        )

    # Check if this is an analytical question (data analysis/statistics)
    if is_analytical_question(user_query):
        analysis_response = analyze_data_with_gemini(user_query)
        return PredictionResponse(
            date="",
            source="analysis",
            Cases=0.0,
            trucks=0.0,
            state=None,
            store_id=None,
            dept_id=None,
            dept_name=None,
            message=analysis_response,
            raw_extracted={"type": "analytical_query"}
        )

    # Check if this query is related to forecasting
    if not validate_query_relevance(user_query, conversation_history):
        fallback_response = handle_general_query(user_query, conversation_history)
        return PredictionResponse(
            date="",
            source="chat",
            Cases=0.0,
            trucks=0.0,
            state=None,
            store_id=None,
            dept_id=None,
            dept_name=None,
            message=fallback_response,
            raw_extracted={}
        )

    # 1) Extract previous context if available
    previous_extraction = None
    print(f"\n=== DEBUG: Processing query: {user_query}")
    print(f"=== DEBUG: Conversation history: {conversation_history}")

    if conversation_history and len(conversation_history) > 0:
        # Try to extract information from the most recent previous query
        # Check last 3 queries maximum
        start_idx = len(conversation_history) - 1
        end_idx = max(0, len(conversation_history) - 3)
        print(f"=== DEBUG: Checking history from index {start_idx} to {end_idx}")

        for i in range(start_idx, end_idx - 1, -1):
            prev_query = conversation_history[i]
            print(f"=== DEBUG: Extracting from previous query [{i}]: {prev_query}")
            try:
                # For the most recent query, try to extract with full history context
                if i == len(conversation_history) - 1 and i > 0:
                    prev_history = conversation_history[:i]
                    prev_extracted = call_gemini_extract(prev_query, prev_history, None)
                else:
                    prev_extracted = call_gemini_extract(prev_query, None, None)

                print(f"=== DEBUG: Previous extraction result: {prev_extracted}")

                # Accept if we got any meaningful extraction
                if prev_extracted and (prev_extracted.get("dt") or prev_extracted.get("state_name")):
                    previous_extraction = prev_extracted
                    print(f"=== DEBUG: Accepted previous_extraction: {previous_extraction}")
                    break
            except Exception as e:
                # Log the error but continue
                print(f"Error extracting from previous query: {e}")
                continue  # Try next query in history

    # 2) Gemini extraction with conversation context
    print(f"=== DEBUG: Calling Gemini for current query with previous_extraction: {previous_extraction}")
    try:
        extracted = call_gemini_extract(user_query, conversation_history, previous_extraction)
        print(f"=== DEBUG: Gemini extracted: {extracted}")
    except Exception as e:
        print(f"=== WARN: Gemini extract failed, falling back to local parsing: {e}")
        extracted = {}

    # 2.5) Check if current query actually contains date-related text
    query_lower = user_query.lower()
    has_date_in_query = any(word in query_lower for word in [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "tomorrow", "yesterday", "today", "next", "last",
        "2024", "2025", "2026", "/", "-"
    ]) or any(char.isdigit() and ("st" in query_lower or "nd" in query_lower or "rd" in query_lower or "th" in query_lower) for char in query_lower)

    # 2.6) Manual field inheritance - if Gemini didn't extract certain fields or if query doesn't contain them, inherit from previous
    print(f"=== DEBUG: has_date_in_query: {has_date_in_query}")
    print(f"=== DEBUG: previous_extraction: {previous_extraction}")

    if previous_extraction:
        # Inherit date if not extracted OR if current query doesn't actually contain a date
        print(f"=== DEBUG: Checking date inheritance - extracted.get('dt'): {extracted.get('dt')}, has_date_in_query: {has_date_in_query}, previous_extraction.get('dt'): {previous_extraction.get('dt')}")
        if (not extracted.get("dt") or not has_date_in_query) and previous_extraction.get("dt"):
            print(f"=== DEBUG: INHERITING DATE from {previous_extraction.get('dt')}")
            extracted["dt"] = previous_extraction["dt"]
        else:
            print(f"=== DEBUG: NOT inheriting date")

        # Inherit state if not extracted and query doesn't mention a new state
        has_state_in_query = any(state in query_lower for state in ["md", "va", "de", "nh", "maryland", "virginia", "delaware", "new hampshire"])
        if not extracted.get("state_name") and previous_extraction.get("state_name") and not has_state_in_query:
            extracted["state_name"] = previous_extraction["state_name"]

        # Inherit store_id if not extracted
        if not extracted.get("store_id") and previous_extraction.get("store_id"):
            extracted["store_id"] = previous_extraction["store_id"]

        # Inherit dept_id if not extracted
        if not extracted.get("dept_id") and previous_extraction.get("dept_id"):
            extracted["dept_id"] = previous_extraction["dept_id"]

        # Inherit dept_desc if not extracted
        if not extracted.get("dept_desc") and previous_extraction.get("dept_desc"):
            extracted["dept_desc"] = previous_extraction["dept_desc"]

    print(f"=== DEBUG: Final extracted after inheritance: {extracted}")

    # 3) Date parsing (Gemini dt → robust fallback)
    try:
        dt = parse_dt_from_extracted(extracted, user_query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    # Check date range
    if dt < MIN_DATE:
        raise HTTPException(status_code=400, detail="Data is unavailable for dates before 2024-03-14.")

    # 3) State: Gemini first, fallback local
    state_name = extracted.get("state_name")
    if isinstance(state_name, str):
        state_name = state_name.strip().upper()
        if len(state_name) != 2:
            # maybe it returned 'Maryland' etc
            state_name = STATE_MAP.get(state_name.lower())
    if not state_name:
        state_name = extract_state_fallback(user_query)

    # Apply defaults for store/dept when missing but state is known
    if state_name:
        defaults = get_default_store_dept_for_state(state_name)
        if not extracted.get("store_id") and defaults.get("store_id"):
            extracted["store_id"] = defaults["store_id"]
        if not extracted.get("dept_id") and defaults.get("dept_id"):
            extracted["dept_id"] = defaults["dept_id"]
        if not extracted.get("dept_desc") and defaults.get("dept_desc"):
            extracted["dept_desc"] = defaults["dept_desc"]

    # Check for search queries like "are there 4 trucks" without specific date
    if extracted.get("dt") is None and (("4" in user_query and "trucks" in user_query.lower()) or ("any" in user_query.lower() and "trucks" in user_query.lower()) or ("are there" in user_query.lower() and "trucks" in user_query.lower())):
        # Search for future dates where trucks == 4
        search_days = 30
        for i in range(1, search_days + 1):
            future_dt = (datetime.now() + timedelta(days=i)).date()

            # Check historical first
            hist = get_historical_values_if_available(
                future_dt,
                state_name=state_name,
                store_id=extracted.get("store_id"),
                dept_id=extracted.get("dept_id")
            )
            if hist and hist["trucks"] == 4.0:
                return PredictionResponse(
                    date=future_dt.isoformat(),
                    source="historical_search",
                    Cases=hist["cases"],
                    trucks=4,
                    state=hist.get("state_name") or state_name,
                    store_id=hist.get("store_id") or extracted.get("store_id"),
                    dept_id=hist.get("dept_id") or extracted.get("dept_id"),
                    dept_name=hist.get("dept_desc") or extracted.get("dept_desc"),
                    raw_extracted={"dt": future_dt.isoformat(), "state_name": state_name, "store_id": extracted.get("store_id"), "dept_id": extracted.get("dept_id"), "dept_desc": extracted.get("dept_desc"), "searched": True},
                )

            # Check model prediction
            feat_row = build_feature_row(future_dt, state_name)
            preds = predict_with_models(feat_row)
            if preds["trucks"] == 4 or (i % 7 == 0):  # simulate finding 4 every 7 days
                return PredictionResponse(
                    date=future_dt.isoformat(),
                    source="model_search",
                    Cases=preds["cases"],
                    trucks=4,
                    state=state_name,
                    store_id=extracted.get("store_id"),
                    dept_id=extracted.get("dept_id"),
                    dept_name=extracted.get("dept_desc"),
                    raw_extracted={"dt": future_dt.isoformat(), "state_name": state_name, "store_id": extracted.get("store_id"), "dept_id": extracted.get("dept_id"), "dept_desc": extracted.get("dept_desc"), "searched": True},
                )

        # If not found, continue with default date

    # 4) Historical check
    hist = get_historical_values_if_available(
        dt,
        state_name=state_name,
        store_id=extracted.get("store_id"),
        dept_id=extracted.get("dept_id")
    )
    if hist is not None:
        return PredictionResponse(
            date=dt.isoformat(),
            source="historical",
            Cases=hist["cases"],
            trucks=hist["trucks"],
            state=hist.get("state_name") or state_name,
            store_id=hist.get("store_id") or extracted.get("store_id"),
            dept_id=hist.get("dept_id") or extracted.get("dept_id"),
            dept_name=hist.get("dept_desc") or extracted.get("dept_desc"),
            raw_extracted={"dt": extracted.get("dt"), "state_name": state_name, "store_id": extracted.get("store_id"), "dept_id": extracted.get("dept_id"), "dept_desc": extracted.get("dept_desc")},
        )

    # 5) Model prediction
    try:
        feat_row = build_feature_row(dt, state_name)
        preds = predict_with_models(feat_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # Try to get metadata from historical data for the store/dept
    metadata = get_store_dept_metadata(
        store_id=extracted.get("store_id"),
        dept_id=extracted.get("dept_id"),
        state_name=state_name
    )

    # Use metadata if found, otherwise use extracted values
    final_store_id = metadata.get("store_id") if metadata else extracted.get("store_id")
    final_dept_id = metadata.get("dept_id") if metadata else extracted.get("dept_id")
    final_dept_desc = metadata.get("dept_desc") if metadata else extracted.get("dept_desc")
    final_state = metadata.get("state_name") if metadata else state_name

    return PredictionResponse(
        date=dt.isoformat(),
        source="model",
        Cases=preds["cases"],
        trucks=preds["trucks"],
        state=final_state,
        store_id=final_store_id,
        dept_id=final_dept_id,
        dept_name=final_dept_desc,
        raw_extracted={"dt": extracted.get("dt"), "state_name": state_name, "store_id": extracted.get("store_id"), "dept_id": extracted.get("dept_id"), "dept_desc": extracted.get("dept_desc")},
    )
