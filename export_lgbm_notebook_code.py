"""
PASTE THIS CODE INTO YOUR LIGHTGBM NOTEBOOK

Copy and paste this code into a new cell at the end of your
LightGBM_Final_Forecasting_V2.ipynb notebook to export everything.
"""

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

import pandas as pd
import pickle
import json
import os

# UPDATE THESE VARIABLE NAMES TO MATCH YOUR NOTEBOOK
# Look for your final trained model - common names:
# final_lgbm, lgb_model_clean, final_model_reduced, reduced_model, etc.
MODEL = final_lgbm  # <-- UPDATE THIS

# Look for your full dataset with all features - common names:
# merged_data3, full_data, final_df, df, etc.
DATA = merged_data3  # <-- UPDATE THIS

# Output directory (change if needed)
OUTPUT_DIR = "/Users/emilgeorgemathew/Agentic-Forecast-Pipeline"

# ============================================================
# STEP 2: EXPORT MODEL
# ============================================================

print("=" * 70)
print("Exporting LightGBM Model for Production")
print("=" * 70)

model_path = os.path.join(OUTPUT_DIR, "best_trucks_lgbm.pkl")
print(f"\n📦 Exporting model to: {model_path}")

with open(model_path, 'wb') as f:
    pickle.dump(MODEL, f)

model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"✅ Model exported! Size: {model_size:.2f} MB")

# ============================================================
# STEP 3: EXPORT HISTORICAL DATA WITH METADATA
# ============================================================

print(f"\n📊 Exporting historical data with metadata...")

# Required columns
required_cols = ['dt', 'trucks', 'cases', 'dept_id', 'state_name', 'store_id', 'dept_desc']

# Check columns exist
missing = [col for col in required_cols if col not in DATA.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"Available columns: {DATA.columns.tolist()}")
else:
    # Create historical data export
    hist_df = DATA[required_cols].copy()

    # Clean and format
    hist_df['dt'] = pd.to_datetime(hist_df['dt'])
    hist_df['trucks'] = hist_df['trucks'].astype(float)
    hist_df['cases'] = hist_df['cases'].astype(float)
    hist_df['dept_id'] = hist_df['dept_id'].astype(int)
    hist_df['store_id'] = hist_df['store_id'].astype(int)
    hist_df['state_name'] = hist_df['state_name'].astype(str).str.upper()
    hist_df['dept_desc'] = hist_df['dept_desc'].astype(str)

    # Remove missing values
    hist_df = hist_df.dropna(subset=['dt', 'trucks', 'cases'])

    # Sort by date
    hist_df = hist_df.sort_values('dt').reset_index(drop=True)

    # Summary
    print(f"\n📊 Data Summary:")
    print(f"   Total rows: {len(hist_df):,}")
    print(f"   Date range: {hist_df['dt'].min()} to {hist_df['dt'].max()}")
    print(f"   Departments: {hist_df['dept_id'].nunique()}")
    print(f"   States: {sorted(hist_df['state_name'].unique())}")
    print(f"   Stores: {hist_df['store_id'].nunique()}")

    # Sample
    print(f"\n📝 Sample (first 10 rows):")
    print(hist_df.head(10))

    # Export
    hist_path = os.path.join(OUTPUT_DIR, "date_trucks_cases.csv")
    hist_df.to_csv(hist_path, index=False)

    hist_size = os.path.getsize(hist_path) / (1024 * 1024)
    print(f"\n✅ Historical data exported to: {hist_path}")
    print(f"   Size: {hist_size:.2f} MB")

# ============================================================
# STEP 4: EXPORT FEATURE MEANS
# ============================================================

print(f"\n📐 Exporting feature means...")

# List of numeric features (update if your model uses different features)
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

feature_means = {}
for feat in numeric_features:
    if feat in DATA.columns:
        feature_means[feat] = float(DATA[feat].mean())
    else:
        feature_means[feat] = 0.0

# Create JSON
means_data = {
    "numeric_feature_means": feature_means,
    "feature_count": len(feature_means),
    "created_from": "LightGBM model export"
}

means_path = os.path.join(OUTPUT_DIR, "feature_means.json")
with open(means_path, 'w') as f:
    json.dump(means_data, f, indent=2)

print(f"✅ Feature means exported to: {means_path}")
print(f"   Features: {len(feature_means)}")

# ============================================================
# STEP 5: SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("🎉 Export Complete!")
print("=" * 70)
print(f"\nFiles created:")
print(f"  1. {model_path}")
print(f"  2. {hist_path}")
print(f"  3. {means_path}")

print(f"\n📝 Next steps:")
print(f"  1. Update app.py to use LightGBM model:")
print(f"     TRUCKS_MODEL_PATH = 'best_trucks_lgbm.pkl'")
print(f"  2. Restart your servers")
print(f"  3. Test queries with metadata:")
print(f"     - 'What's the forecast for Maryland store 10001 tomorrow?'")
print(f"     - 'Show me department 5 forecast for Jan 1 2025'")

print("\n✅ Ready for deployment!")
