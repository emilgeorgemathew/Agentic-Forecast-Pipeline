"""
Add export cell to the LightGBM notebook
"""

import json
import sys

# Path to notebook
NOTEBOOK_PATH = "/Users/emilgeorgemathew/Downloads/wat/LightGBM_Final_Forecasting_V2.ipynb"
OUTPUT_PATH = "/Users/emilgeorgemathew/Downloads/wat/LightGBM_Final_Forecasting_V2_with_export.ipynb"

# Export code to add
EXPORT_CODE = '''# ============================================================
# EXPORT MODEL AND DATA FOR PRODUCTION DEPLOYMENT
# ============================================================

import pandas as pd
import pickle
import json
import os

print("=" * 70)
print("Exporting LightGBM Model for Production")
print("=" * 70)

# Output directory
OUTPUT_DIR = "/Users/emilgeorgemathew/Agentic-Forecast-Pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: EXPORT TRAINED MODEL
# ============================================================

print("\\n📦 Step 1: Exporting trained model...")

# The final trained model from this notebook
MODEL = final_lgbm

model_path = os.path.join(OUTPUT_DIR, "best_trucks_lgbm.pkl")
print(f"   Output: {model_path}")

with open(model_path, 'wb') as f:
    pickle.dump(MODEL, f)

model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"✅ Model exported successfully! Size: {model_size:.2f} MB")
print(f"   Model type: {type(MODEL)}")

# ============================================================
# STEP 2: EXPORT HISTORICAL DATA WITH METADATA
# ============================================================

print(f"\\n📊 Step 2: Exporting historical data with metadata...")

# Use the full merged dataset that has all features and metadata
# This should be your merged_data3 or similar variable that has:
# dt, trucks, cases, dept_id, state_name, store_id, dept_desc

# Try to find the data variable - update this if your variable name is different
try:
    DATA = merged_data3
    print("   Using: merged_data3")
except NameError:
    try:
        DATA = full_data
        print("   Using: full_data")
    except NameError:
        try:
            DATA = final_df
            print("   Using: final_df")
        except NameError:
            print("❌ ERROR: Could not find data variable!")
            print("   Please set DATA = your_data_variable_name above this cell")
            sys.exit(1)

# Required columns for the API
required_cols = ['dt', 'trucks', 'cases', 'dept_id', 'state_name', 'store_id', 'dept_desc']

# Check if all required columns exist
missing = [col for col in required_cols if col not in DATA.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"\\nAvailable columns in your data:")
    print(DATA.columns.tolist())
    print("\\n⚠️  You may need to merge additional data to get these columns.")
else:
    # Create historical data export
    hist_df = DATA[required_cols].copy()

    # Clean and format data
    print("   Cleaning and formatting data...")
    hist_df['dt'] = pd.to_datetime(hist_df['dt'])
    hist_df['trucks'] = hist_df['trucks'].astype(float)
    hist_df['cases'] = hist_df['cases'].astype(float)
    hist_df['dept_id'] = hist_df['dept_id'].astype(int)
    hist_df['store_id'] = hist_df['store_id'].astype(int)
    hist_df['state_name'] = hist_df['state_name'].astype(str).str.upper()
    hist_df['dept_desc'] = hist_df['dept_desc'].astype(str)

    # Remove rows with missing critical values
    before_count = len(hist_df)
    hist_df = hist_df.dropna(subset=['dt', 'trucks', 'cases'])
    after_count = len(hist_df)

    if before_count > after_count:
        print(f"   Removed {before_count - after_count:,} rows with missing values")

    # Sort by date
    hist_df = hist_df.sort_values('dt').reset_index(drop=True)

    # Display summary
    print(f"\\n📊 Historical Data Summary:")
    print(f"   Total rows: {len(hist_df):,}")
    print(f"   Date range: {hist_df['dt'].min()} to {hist_df['dt'].max()}")
    print(f"   Unique departments: {hist_df['dept_id'].nunique()}")
    print(f"   Unique states: {hist_df['state_name'].nunique()}")
    print(f"   States: {sorted(hist_df['state_name'].unique())}")
    print(f"   Unique stores: {hist_df['store_id'].nunique()}")

    # Display sample
    print(f"\\n📝 Sample data (first 10 rows):")
    display(hist_df.head(10))

    # Export to CSV
    hist_path = os.path.join(OUTPUT_DIR, "date_trucks_cases.csv")
    print(f"\\n   Exporting to: {hist_path}")
    hist_df.to_csv(hist_path, index=False)

    hist_size = os.path.getsize(hist_path) / (1024 * 1024)
    print(f"✅ Historical data exported successfully! Size: {hist_size:.2f} MB")

# ============================================================
# STEP 3: EXPORT FEATURE MEANS
# ============================================================

print(f"\\n📐 Step 3: Exporting feature means...")

# All numeric features used by the model
# These are used as fallback values when making predictions
numeric_features = [
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

# Calculate means from the data
feature_means = {}
missing_features = []

for feat in numeric_features:
    if feat in DATA.columns:
        feature_means[feat] = float(DATA[feat].mean())
    else:
        feature_means[feat] = 0.0
        missing_features.append(feat)

if missing_features:
    print(f"   ⚠️  Warning: {len(missing_features)} features not found in data (using 0.0 as default)")
    print(f"   Missing: {missing_features[:5]}...")

# Create JSON output
means_data = {
    "numeric_feature_means": feature_means,
    "feature_count": len(feature_means),
    "missing_features": missing_features,
    "created_from": "LightGBM notebook export"
}

means_path = os.path.join(OUTPUT_DIR, "feature_means.json")
with open(means_path, 'w') as f:
    json.dump(means_data, f, indent=2)

print(f"✅ Feature means exported to: {means_path}")
print(f"   Features exported: {len(feature_means)}")

# ============================================================
# STEP 4: SUMMARY AND NEXT STEPS
# ============================================================

print("\\n" + "=" * 70)
print("🎉 EXPORT COMPLETE!")
print("=" * 70)

print(f"\\n📁 Files created in: {OUTPUT_DIR}")
print(f"   1. best_trucks_lgbm.pkl     - Trained LightGBM model")
print(f"   2. date_trucks_cases.csv    - Historical data with metadata")
print(f"   3. feature_means.json       - Feature default values")

print(f"\\n📝 Next steps to deploy:")
print(f"   1. Backup your current files:")
print(f"      cd {OUTPUT_DIR}")
print(f"      cp best_trucks_ts_cv.pkl best_trucks_ts_cv.pkl.backup")
print(f"      cp date_trucks_cases.csv date_trucks_cases.csv.backup")

print(f"\\n   2. Update app.py to use the new model:")
print(f"      Edit line ~95: TRUCKS_MODEL_PATH = 'best_trucks_lgbm.pkl'")

print(f"\\n   3. Restart your servers:")
print(f"      lsof -ti:8000 -ti:8501 | xargs kill -9")
print(f"      python3 -m uvicorn app:app --reload --port 8000 &")
print(f"      streamlit run Interface.py --server.port 8501 &")

print(f"\\n   4. Test with queries like:")
print(f"      - 'What's the forecast for Maryland tomorrow?'")
print(f"      - 'Show me forecast for store 10001 on Jan 1 2025'")
print(f"      - 'What's department 5 forecast in Virginia?'")

print("\\n✅ Your bot will now show dept_id, state_name, store_id, and dept_desc!")
print("\\nSee EXPORT_LGBM_README.md for detailed deployment instructions.")
'''

def add_export_cell_to_notebook():
    """Add the export code as a new cell at the end of the notebook"""

    print(f"Reading notebook: {NOTEBOOK_PATH}")

    # Load notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    print(f"Current cells: {len(notebook['cells'])}")

    # Create new code cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": EXPORT_CODE.split('\n')
    }

    # Add to notebook
    notebook['cells'].append(new_cell)

    print(f"Added export cell. New total: {len(notebook['cells'])}")

    # Save updated notebook
    print(f"Saving to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"\n✅ Success! Created: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"1. Open the new notebook in Jupyter/Colab")
    print(f"2. Run all cells (Runtime > Run all)")
    print(f"3. The last cell will export your model and data")
    print(f"4. Follow the instructions printed by the export cell")

if __name__ == "__main__":
    add_export_cell_to_notebook()
