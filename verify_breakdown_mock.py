
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np

# Mock Global Variables
MIN_DATE = date(2024, 3, 14)
MAX_DATE = date(2025, 3, 14)

# Mock DataFrame
data = {
    "dt": [pd.Timestamp("2024-03-14"), pd.Timestamp("2024-03-14"), pd.Timestamp("2024-03-14")],
    "state_name": ["MD", "MD", "VA"],
    "store_id": [10001, 10002, 20001],
    "dept_id": [1, 1, 1],
    "dept_desc": ["Dept A", "Dept A", "Dept A"],
    "cases": [100.0, 200.0, 150.0],
    "trucks": [2.0, 3.0, 2.5]
}
df_hist = pd.DataFrame(data)

# Mock Helper Functions
def build_feature_row(dt, state_name):
    return pd.DataFrame({"feature": [1]})

def predict_with_models(features):
    return {"cases": 123.0, "trucks": 4.0}

# The Function to Test (Copied from app.py)
def get_state_breakdown(state_name: str, target_date: date):
    """
    Get or predict cases/trucks for all stores/depts in a state for a specific date.
    """
    results = []
    
    # Get all unique store/dept combinations for this state
    state_mask = df_hist["state_name"].str.upper() == state_name.upper()
    state_data = df_hist[state_mask]
    
    if len(state_data) == 0:
        return []
        
    # Get unique combinations
    combinations = state_data[["store_id", "dept_id", "dept_desc"]].drop_duplicates()
    
    # Check if we have historical data for this date
    is_historical = MIN_DATE <= target_date <= MAX_DATE
    
    if is_historical:
        # Fetch actual values
        date_mask = state_data["dt"].dt.date == target_date
        daily_data = state_data[date_mask]
        
        for _, row in daily_data.iterrows():
            results.append({
                "store_id": int(row["store_id"]),
                "dept_id": int(row["dept_id"]),
                "dept_desc": row["dept_desc"],
                "cases": float(row["cases"]),
                "trucks": float(row["trucks"]),
                "source": "historical"
            })
    else:
        # Generate predictions
        for _, combo in combinations.iterrows():
            try:
                # Build feature row for this specific store/dept
                # We need to temporarily set the context for the feature builder
                # Note: build_feature_row uses some defaults, we might need to be more specific
                # For now, we'll use the existing helper but we need to inject the specific store params
                
                # 1. Build base features for date
                features = build_feature_row(target_date, state_name)
                
                # 2. Override with specific store/dept details
                features["store_id"] = combo["store_id"]
                features["dept_id"] = combo["dept_id"]
                # Ideally we would look up specific store/dept means here
                # For this implementation, we'll rely on the randomized variation in build_feature_row
                # but we should try to be as accurate as possible if we had the metadata map
                
                # 3. Predict
                preds = predict_with_models(features)
                
                results.append({
                    "store_id": int(combo["store_id"]),
                    "dept_id": int(combo["dept_id"]),
                    "dept_desc": combo["dept_desc"],
                    "cases": preds["cases"],
                    "trucks": preds["trucks"],
                    "source": "model"
                })
            except Exception as e:
                print(f"Error predicting for {combo['store_id']}/{combo['dept_id']}: {e}")
                continue
                
    return sorted(results, key=lambda x: (x["store_id"], x["dept_id"]))

def test_state_breakdown():
    print("\n=== Testing State Breakdown Logic (Mocked) ===")
    
    # Test Case 1: Historical Data (MD)
    test_date = date(2024, 3, 14)
    state = "MD"
    print(f"\nTesting Historical Data for {state} on {test_date}")
    
    results = get_state_breakdown(state, test_date)
    
    if len(results) == 2:
        print(f"✅ Success! Found {len(results)} records.")
        print("Sample record:", results[0])
        if results[0]["source"] == "historical":
             print("✅ Source is correctly marked as 'historical'")
        else:
             print(f"❌ Error: Source should be 'historical', got '{results[0]['source']}'")
    else:
        print(f"❌ Error: Expected 2 records, got {len(results)}")

    # Test Case 2: Future Prediction (MD)
    future_date = date(2026, 1, 1)
    state = "MD"
    print(f"\nTesting Future Prediction for {state} on {future_date}")
    
    results = get_state_breakdown(state, future_date)
    
    if len(results) == 2:
        print(f"✅ Success! Generated {len(results)} predictions.")
        print("Sample record:", results[0])
        if results[0]["source"] == "model":
             print("✅ Source is correctly marked as 'model'")
        else:
             print(f"❌ Error: Source should be 'model', got '{results[0]['source']}'")
    else:
        print(f"❌ Error: Expected 2 records, got {len(results)}")

if __name__ == "__main__":
    test_state_breakdown()
