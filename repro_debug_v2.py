import pandas as pd
from datetime import datetime, timedelta
import dateparser
import re

# Mock data mimicking the CSV structure
data = {
    "dt": ["2024-03-14", "2024-03-14"],
    "trucks": [3.0, 3.0],
    "cases": [81.0, 70.0],
    "dept_id": [10, 82],
    "state_name": ["MD", "FL"],
    "store_id": [10001, 10082],
    "dept_desc": ["AUTOMOTIVE", "IMPULSE MERCHANDISE"]
}
df_hist = pd.DataFrame(data)
# Ensure types match app.py loading (pandas defaults)
# store_id will be int64

def robust_date_parse(full_query: str) -> str:
    parsing_settings = {
        "RELATIVE_BASE": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "PREFER_DATES_FROM": "future"
    }
    
    # 1. Try parsing the full query
    dt = dateparser.parse(full_query, languages=['en'], settings=parsing_settings)
    if dt is not None:
        return f"Method 1 (Full): {dt.date()}"

    # 2. Try to grab date-like chunks
    patterns = [
        r"\bnext\s+\w+\b",  # next Friday
        r"\blast\s+\w+\b",  # last Friday
    ]
    for pat in patterns:
        m = re.search(pat, full_query, flags=re.IGNORECASE)
        if m:
            chunk = m.group(0)
            print(f"Debug: Found chunk '{chunk}'")
            
            # Special handling for "next [day]"
            if "next" in chunk.lower():
                parsing_settings["PREFER_DATES_FROM"] = "future"
                # Try parsing the chunk as is
                dt2 = dateparser.parse(chunk, languages=['en'], settings=parsing_settings)
                
                # If that fails, try parsing just the day name (e.g. "Saturday")
                if dt2 is None:
                    day_name = chunk.lower().replace("next", "").strip()
                    print(f"Debug: Retrying with day name '{day_name}'")
                    dt2 = dateparser.parse(day_name, languages=['en'], settings=parsing_settings)
                    if dt2 is not None:
                        # If we parsed "Saturday" and it's today or past, add 7 days to make it "next"
                        # But PREFER_DATES_FROM="future" should handle it? 
                        # Actually "Saturday" with future pref usually gives the *upcoming* Saturday.
                        # "Next Saturday" usually means the one *after* the upcoming one if today is close?
                        # Let's stick to: if it's <= today, add 7 days.
                        pass

            else:
                dt2 = dateparser.parse(chunk, languages=['en'], settings=parsing_settings)

            if dt2 is not None:
                # Logic from app.py
                if "next" in chunk.lower() and dt2.date() <= datetime.now().date():
                     dt2 = dt2 + timedelta(days=7)
                return f"Method 2 (Chunk '{chunk}'): {dt2.date()}"

    return "Default: Tomorrow"

def infer_state(store_id_input):
    print(f"Debug: df_hist['store_id'] dtype: {df_hist['store_id'].dtype}")
    print(f"Debug: Input store_id: {store_id_input} (type: {type(store_id_input)})")
    
    try:
        store_id_val = int(store_id_input)
        store_matches = df_hist[df_hist["store_id"] == store_id_val]
        if not store_matches.empty:
            return f"Found: {store_matches.iloc[0]['state_name']}"
        else:
            return "Not Found"
    except Exception as e:
        return f"Error: {e}"

print(f"Current Date: {datetime.now()}")
print("-" * 20)
query = "Forecast for store 10001 for next Saturday"
print(f"Query: '{query}'")
print(f"Parsed Date: {robust_date_parse(query)}")
print(f"Inferred State: {infer_state(10001)}")
print("-" * 20)
