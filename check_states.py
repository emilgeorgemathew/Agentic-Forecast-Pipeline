import pandas as pd
import os

csv_path = "date_trucks_cases.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if "state_name" in df.columns:
        states = sorted(df["state_name"].dropna().unique())
        print(f"Unique states in CSV ({len(states)}): {states}")
        if "DE" in states:
            print("DE is present in the CSV.")
        else:
            print("DE is NOT present in the CSV.")
    else:
        print("Column 'state_name' not found in CSV.")
else:
    print(f"File {csv_path} not found.")
