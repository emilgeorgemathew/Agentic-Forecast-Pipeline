"""
Export LightGBM Model and Historical Data with Metadata

This script exports:
1. The LightGBM model as a pickle file for trucks prediction
2. Historical data CSV with all required metadata (dept_id, state_name, store_id, dept_desc)
3. Updated feature_means.json for the new model

INSTRUCTIONS:
1. Run this in the same environment where your LightGBM notebook runs
2. Make sure you have the trained model and data loaded
3. Update the paths and variable names below to match your notebook
4. Run: python3 export_lgbm_model.py
"""

import pandas as pd
import pickle
import json
import os
from pathlib import Path

# ============================================================
# CONFIGURATION - UPDATE THESE TO MATCH YOUR NOTEBOOK
# ============================================================

# The trained LightGBM model variable name from your notebook
# Common names: final_lgbm, lgb_model_clean, final_model_reduced, etc.
# Update this to match your notebook's final model variable
MODEL_VARIABLE_NAME = "final_lgbm"  # Change this to your model variable name

# The full dataset with all features and metadata
# Common names: merged_data3, full_data, final_df, etc.
DATA_VARIABLE_NAME = "merged_data3"  # Change this to your data variable name

# Output paths (relative to current directory)
OUTPUT_DIR = "/Users/emilgeorgemathew/Agentic-Forecast-Pipeline"
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "best_trucks_lgbm.pkl")
HISTORICAL_DATA_PATH = os.path.join(OUTPUT_DIR, "date_trucks_cases.csv")
FEATURE_MEANS_PATH = os.path.join(OUTPUT_DIR, "feature_means.json")

# Required columns for historical data
REQUIRED_METADATA = ['dt', 'trucks', 'cases', 'dept_id', 'state_name', 'store_id', 'dept_desc']

# Features used by the model (update based on your notebook)
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

# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_model(model, output_path):
    """Export the trained model to a pickle file."""
    print(f"\n📦 Exporting model to: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✅ Model exported successfully! Size: {file_size:.2f} MB")
    return True


def export_historical_data(data, output_path, required_cols=REQUIRED_METADATA):
    """Export historical data with metadata."""
    print(f"\n📊 Preparing historical data...")

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        print(f"\nAvailable columns in your data:")
        print(data.columns.tolist())
        return False

    # Select required columns
    historical_df = data[required_cols].copy()

    # Ensure proper data types
    historical_df['dt'] = pd.to_datetime(historical_df['dt'])
    historical_df['trucks'] = historical_df['trucks'].astype(float)
    historical_df['cases'] = historical_df['cases'].astype(float)
    historical_df['dept_id'] = historical_df['dept_id'].astype(int)
    historical_df['store_id'] = historical_df['store_id'].astype(int)
    historical_df['state_name'] = historical_df['state_name'].astype(str).str.upper()
    historical_df['dept_desc'] = historical_df['dept_desc'].astype(str)

    # Remove any rows with missing values in critical columns
    before_count = len(historical_df)
    historical_df = historical_df.dropna(subset=['dt', 'trucks', 'cases'])
    after_count = len(historical_df)

    if before_count > after_count:
        print(f"⚠️  Removed {before_count - after_count} rows with missing critical values")

    # Sort by date
    historical_df = historical_df.sort_values('dt').reset_index(drop=True)

    # Display summary
    print(f"\n📊 Historical Data Summary:")
    print(f"   Total rows: {len(historical_df):,}")
    print(f"   Date range: {historical_df['dt'].min()} to {historical_df['dt'].max()}")
    print(f"   Unique departments: {historical_df['dept_id'].nunique()}")
    print(f"   Unique states: {historical_df['state_name'].nunique()} ({sorted(historical_df['state_name'].unique())})")
    print(f"   Unique stores: {historical_df['store_id'].nunique()}")

    print(f"\n📝 Sample data:")
    print(historical_df.head(10).to_string(index=False))

    # Export to CSV
    print(f"\n💾 Exporting to: {output_path}")
    historical_df.to_csv(output_path, index=False)

    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✅ Historical data exported successfully! Size: {file_size:.2f} MB")
    return True


def export_feature_means(data, output_path, numeric_features=NUMERIC_FEATURES):
    """Export feature means for prediction fallback."""
    print(f"\n📐 Calculating feature means...")

    feature_means = {}

    for feature in numeric_features:
        if feature in data.columns:
            mean_val = float(data[feature].mean())
            feature_means[feature] = mean_val
        else:
            print(f"⚠️  Warning: Feature '{feature}' not found in data, using 0.0")
            feature_means[feature] = 0.0

    # Create output structure
    output_data = {
        "numeric_feature_means": feature_means,
        "feature_count": len(feature_means),
        "created_from": "LightGBM model export"
    }

    print(f"\n💾 Exporting feature means to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Feature means exported successfully! ({len(feature_means)} features)")
    return True


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("LightGBM Model & Data Export Tool")
    print("=" * 70)

    # This script should be run in the notebook environment
    # or you need to manually set the model and data variables

    print("\n⚠️  IMPORTANT: This script expects to be run in an environment where:")
    print(f"   1. The trained model is available as: {MODEL_VARIABLE_NAME}")
    print(f"   2. The full dataset is available as: {DATA_VARIABLE_NAME}")
    print("\n   If running standalone, you need to load these variables first!")

    # Check if running in notebook context
    try:
        # Try to access the global namespace (works in notebooks and some interactive environments)
        model = globals().get(MODEL_VARIABLE_NAME)
        data = globals().get(DATA_VARIABLE_NAME)

        if model is None:
            print(f"\n❌ ERROR: Model variable '{MODEL_VARIABLE_NAME}' not found!")
            print("   Please update MODEL_VARIABLE_NAME in this script to match your model variable.")
            return

        if data is None:
            print(f"\n❌ ERROR: Data variable '{DATA_VARIABLE_NAME}' not found!")
            print("   Please update DATA_VARIABLE_NAME in this script to match your data variable.")
            return

    except Exception as e:
        print(f"\n❌ ERROR: Could not access variables: {e}")
        print("\n   This script should be run inside your notebook or you need to load the data manually.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export model
    success_model = export_model(model, MODEL_OUTPUT_PATH)

    # Export historical data with metadata
    success_data = export_historical_data(data, HISTORICAL_DATA_PATH)

    # Export feature means
    success_means = export_feature_means(data, FEATURE_MEANS_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    print(f"✅ Model export: {'SUCCESS' if success_model else 'FAILED'}")
    print(f"✅ Historical data: {'SUCCESS' if success_data else 'FAILED'}")
    print(f"✅ Feature means: {'SUCCESS' if success_means else 'FAILED'}")

    if success_model and success_data and success_means:
        print("\n🎉 All exports completed successfully!")
        print("\n📝 Next steps:")
        print(f"   1. Copy {MODEL_OUTPUT_PATH} to your deployment directory")
        print(f"   2. Backup and replace date_trucks_cases.csv with the new version")
        print(f"   3. Update your app.py to use the LightGBM model:")
        print(f"      TRUCKS_MODEL_PATH = 'best_trucks_lgbm.pkl'")
        print(f"   4. Restart your servers")
    else:
        print("\n⚠️  Some exports failed. Please check the errors above.")


if __name__ == "__main__":
    main()
