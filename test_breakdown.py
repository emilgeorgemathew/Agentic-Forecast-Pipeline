
import os
import sys
import pandas as pd
from datetime import date, timedelta

# Mock streamlit secrets/env for app import
os.environ["GOOGLE_API_KEY"] = "dummy"
os.environ["API_URL"] = "http://localhost:8000/predict"

# Import app functions
# We need to add the current directory to sys.path
sys.path.append(os.getcwd())

from app import get_state_breakdown, df_hist, STATE_MAP

def test_breakdown():
    print("Testing get_state_breakdown for MD...")
    
    # Check if MD is in the data
    md_data = df_hist[df_hist["state_name"] == "MD"]
    print(f"MD records in history: {len(md_data)}")
    
    if len(md_data) > 0:
        print("Sample MD data:")
        print(md_data[["store_id", "dept_id", "dt"]].head())
        
    # Test for tomorrow
    target_date = date.today() + timedelta(days=1)
    print(f"Target date: {target_date}")
    
    results = get_state_breakdown("MD", target_date)
    print(f"Results count: {len(results)}")
    
    if len(results) > 0:
        print("First result:")
        print(results[0])
    else:
        print("No results returned.")

if __name__ == "__main__":
    test_breakdown()
