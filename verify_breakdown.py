
import sys
import os
import pandas as pd
from datetime import date, datetime

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

# Mock the necessary parts or import app
try:
    from app import get_state_breakdown, df_hist, MIN_DATE, MAX_DATE
    print("Successfully imported app modules")
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

def test_state_breakdown():
    print("\n=== Testing State Breakdown Logic ===")
    
    # Test Case 1: Historical Data (MD)
    # Find a date that exists in the data
    test_date = df_hist["dt"].min().date()
    state = "MD"
    print(f"\nTesting Historical Data for {state} on {test_date}")
    
    results = get_state_breakdown(state, test_date)
    
    if results:
        print(f"✅ Success! Found {len(results)} records.")
        print("Sample record:")
        print(results[0])
        
        # Verify structure
        required_keys = ["store_id", "dept_id", "dept_desc", "cases", "trucks", "source"]
        missing_keys = [k for k in required_keys if k not in results[0]]
        if missing_keys:
            print(f"❌ Error: Missing keys in result: {missing_keys}")
        else:
            print("✅ Record structure looks correct.")
            
        # Verify source
        if results[0]["source"] == "historical":
             print("✅ Source is correctly marked as 'historical'")
        else:
             print(f"❌ Error: Source should be 'historical', got '{results[0]['source']}'")
    else:
        print(f"❌ Error: No results found for {state} on {test_date}")

    # Test Case 2: Future Prediction (VA)
    future_date = date(2026, 1, 1)
    state = "VA"
    print(f"\nTesting Future Prediction for {state} on {future_date}")
    
    results = get_state_breakdown(state, future_date)
    
    if results:
        print(f"✅ Success! Generated {len(results)} predictions.")
        print("Sample record:")
        print(results[0])
        
        # Verify source
        if results[0]["source"] == "model":
             print("✅ Source is correctly marked as 'model'")
        else:
             print(f"❌ Error: Source should be 'model', got '{results[0]['source']}'")
    else:
        print(f"❌ Error: No results generated for {state} on {future_date}")

if __name__ == "__main__":
    test_state_breakdown()
