import pandas as pd
import gdown

# Google Drive file ID
file_id = "1hlt8iiILix5PVGoQJjXdtqdULXwRYkKQ"

print("Downloading CSV from Google Drive using gdown...")
try:
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "gdrive_data.csv", quiet=False)
    df = pd.read_csv("gdrive_data.csv")
except Exception as e:
    print(f"Error downloading: {e}")
    exit(1)

# Display basic info
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Find date column
date_col = None
for col in ["dt", "date", "Date", "DT", "timestamp", "Timestamp"]:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    print("\nWARNING: No date column found (checked: dt, date, Date, DT, timestamp, Timestamp)")
    print("First few rows:")
    print(df.head())
else:
    print(f"\nUsing date column: '{date_col}'")
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract min and max dates
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    print(f"\n=== Date Range ===")
    print(f"  Start: {min_date.date()}")
    print(f"  End:   {max_date.date()}")
    print(f"  Total days: {(max_date - min_date).days + 1}")
    
    # Show date distribution
    print(f"\n=== Date Distribution ===")
    print(f"  Year range: {min_date.year} - {max_date.year}")
    print(f"  Unique dates in dataset: {df[date_col].nunique()}")
    
    # Check for key columns
    key_cols = ["store_id", "dept_id", "cases", "trucks"]
    available_cols = [c for c in key_cols if c in df.columns]
    print(f"\n=== Available Columns ===")
    print(f"  {available_cols}")
    
    # Show sample data
    print(f"\n=== Sample Data ===")
    print(df.head(10))
