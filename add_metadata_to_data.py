"""
Script to add dept_id and state_id columns to your historical data files.

This script helps you update your date_trucks_cases.csv to include metadata
like dept_id, state_name, store_id, and dept_desc so the bot can display
this information along with forecasts.

INSTRUCTIONS:
1. You need your original data source that has dept_id, state_id/state_name,
   store_id, and dept_desc columns
2. Update the SOURCE_FILE path below to point to your original data
3. Run this script: python3 add_metadata_to_data.py
4. It will create a new file: date_trucks_cases_with_metadata.csv
5. Update your .env or code to use the new file

IMPORTANT: The models were trained with specific features. Adding metadata
to the historical data allows the API to return this info in responses,
but doesn't change the predictions themselves.
"""

import pandas as pd
import os

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================

# Path to your original data source that has all the metadata columns
# This could be your training data CSV or the original dataset
SOURCE_FILE = "REPLACE_WITH_YOUR_ORIGINAL_DATA.csv"  # e.g., "train.csv"

# Path to your current minimal historical data
CURRENT_FILE = "date_trucks_cases.csv"

# Output file with metadata
OUTPUT_FILE = "date_trucks_cases_with_metadata.csv"

# Required columns in the source file
REQUIRED_COLUMNS = ['dt', 'trucks', 'cases', 'dept_id', 'state_name', 'store_id', 'dept_desc']

# ============================================================
# SCRIPT
# ============================================================

def main():
    print("=" * 60)
    print("Adding metadata to historical data")
    print("=" * 60)

    # Check if source file exists
    if not os.path.exists(SOURCE_FILE):
        print(f"\n❌ ERROR: Source file not found: {SOURCE_FILE}")
        print("\nPlease update SOURCE_FILE in this script to point to your original data.")
        print("Your original data should have these columns:")
        print("  - dt (date)")
        print("  - trucks")
        print("  - cases")
        print("  - dept_id (department ID)")
        print("  - state_name (state code like 'MD', 'VA', etc.)")
        print("  - store_id (store ID)")
        print("  - dept_desc (department description/name)")
        return

    print(f"\n📁 Reading source data from: {SOURCE_FILE}")
    df_source = pd.read_csv(SOURCE_FILE)

    print(f"   Columns in source: {df_source.columns.tolist()}")
    print(f"   Rows: {len(df_source):,}")

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_source.columns]
    if missing_cols:
        print(f"\n❌ ERROR: Missing required columns in source file: {missing_cols}")
        print(f"\nAvailable columns: {df_source.columns.tolist()}")
        print("\nIf your columns have different names, you can either:")
        print("  1. Rename them in your source file, or")
        print("  2. Modify this script to map your column names")
        return

    # Parse date column
    if 'dt' in df_source.columns:
        df_source['dt'] = pd.to_datetime(df_source['dt'])

    # Select and order the required columns
    print(f"\n✅ Extracting required columns...")
    df_final = df_source[REQUIRED_COLUMNS].copy()

    # Sort by date
    df_final = df_final.sort_values('dt').reset_index(drop=True)

    # Display sample
    print(f"\n📊 Sample of processed data:")
    print(df_final.head(10).to_string(index=False))

    print(f"\n📊 Data summary:")
    print(f"   Total rows: {len(df_final):,}")
    print(f"   Date range: {df_final['dt'].min()} to {df_final['dt'].max()}")
    print(f"   Unique departments: {df_final['dept_id'].nunique()}")
    print(f"   Unique states: {df_final['state_name'].nunique()}")
    print(f"   Unique stores: {df_final['store_id'].nunique()}")

    # Save to new file
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ SUCCESS! Created {OUTPUT_FILE}")
    print(f"\n📝 Next steps:")
    print(f"   1. Backup your current file:")
    print(f"      cp {CURRENT_FILE} {CURRENT_FILE}.backup")
    print(f"   2. Replace the current file with the new one:")
    print(f"      mv {OUTPUT_FILE} {CURRENT_FILE}")
    print(f"   3. Restart your server to use the new data")
    print(f"\n   OR update your .env file:")
    print(f"      DATE_TRUCKS_CASES_CSV={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
