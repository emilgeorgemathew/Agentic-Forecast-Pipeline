# Guide: Adding Metadata (dept_id, state_id) to Your Forecast Bot

This guide explains how to update your data files so the forecasting bot can display department ID, state name, store ID, and department description along with the forecasts.

## Current Situation

Your current `date_trucks_cases.csv` file has only 3 columns:
- `dt` (date)
- `trucks` (number of trucks)
- `cases` (number of cases)

The API already supports returning metadata (dept_id, state_name, store_id, dept_desc), but this metadata isn't in your historical data file, so it can't be displayed for historical lookups.

## What I've Done

I've made two key updates:

### 1. Updated `app.py`
Modified the `get_historical_values_if_available()` function ([app.py:430-472](app.py#L430-L472)) to:
- Accept optional filtering parameters: `state_name`, `store_id`, `dept_id`
- Return metadata from the CSV if available
- Filter historical data based on these parameters when provided

Updated all calls to this function to pass filtering parameters and use returned metadata.

### 2. Created `add_metadata_to_data.py`
A helper script to add metadata columns to your historical data file.

## How to Add Metadata to Your Data

### Option 1: You Have the Original Data Source

If you have your original training data or source dataset that includes all the metadata columns:

1. **Update the script**: Edit [add_metadata_to_data.py](add_metadata_to_data.py)
   ```python
   SOURCE_FILE = "your_original_data.csv"  # Update this line
   ```

2. **Run the script**:
   ```bash
   python3 add_metadata_to_data.py
   ```

3. **Backup and replace**:
   ```bash
   cp date_trucks_cases.csv date_trucks_cases.csv.backup
   mv date_trucks_cases_with_metadata.csv date_trucks_cases.csv
   ```

4. **Restart servers** to use the new data

### Option 2: You Don't Have the Original Data

If you don't have the original data with metadata, you have two options:

#### A. Add Placeholder/Default Metadata

Create a simple script to add default values:

```python
import pandas as pd

# Read current data
df = pd.read_csv('date_trucks_cases.csv', parse_dates=['dt'])

# Add metadata columns with default values
# You can customize these based on your needs
df['dept_id'] = 1  # or any default department
df['state_name'] = 'MD'  # or any default state
df['store_id'] = 10001  # or any default store
df['dept_desc'] = 'General'  # or any default description

# Save back
df.to_csv('date_trucks_cases.csv', index=False)
```

#### B. Generate Metadata Based on Patterns

If your data has patterns (e.g., different combinations of cases/trucks correspond to different departments), you could create logic to infer the metadata:

```python
import pandas as pd

df = pd.read_csv('date_trucks_cases.csv', parse_dates=['dt'])

# Example: Infer dept_id based on case ranges
def infer_dept(cases):
    if cases < 50:
        return 1
    elif cases < 100:
        return 2
    else:
        return 3

df['dept_id'] = df['cases'].apply(infer_dept)
df['state_name'] = 'MD'  # Default state
df['store_id'] = 10001  # Default store
df['dept_desc'] = df['dept_id'].map({1: 'Small', 2: 'Medium', 3: 'Large'})

df.to_csv('date_trucks_cases.csv', index=False)
```

## Expected CSV Format After Update

Your `date_trucks_cases.csv` should have these columns:

```csv
dt,trucks,cases,dept_id,state_name,store_id,dept_desc
2024-03-14,3.0,81.0,5,MD,10001,GROCERY
2024-03-14,2.0,52.0,2,VA,10002,PRODUCE
2024-03-14,2.0,52.0,2,VA,10002,PRODUCE
...
```

## How It Works

### Historical Data Queries

When a user asks for a historical date:
1. The API extracts date, state, store_id, and dept_id from the query
2. It filters the historical data based on these parameters
3. It returns the trucks/cases values along with the metadata from the matching row

**Example**:
- User: "What's the forecast for Maryland store 10001 on March 14, 2024?"
- API filters: `dt='2024-03-14'`, `state_name='MD'`, `store_id=10001`
- Returns: trucks, cases, AND the dept_id and dept_desc from that row

### Model Predictions

For future dates (no historical data):
- The API uses the extracted metadata from the query
- Displays it alongside the model's predictions
- The metadata helps provide context about WHAT is being forecasted

## Benefits

Once metadata is in your data:

1. **Better Context**: Users see what department/store the forecast is for
2. **Filtering**: Historical lookups can filter by state, store, or department
3. **Tracking**: Users can track forecasts across different dimensions
4. **Debugging**: Easier to understand which data point is being returned

## Frontend Display

The Streamlit interface in [Interface.py](Interface.py) already displays this information when available:
- State name
- Store ID
- Department ID
- Department name/description

Example response display:
```
📊 Forecast for 2024-03-14

Cases: 81.0
Trucks: 3

📍 State: MD
🏪 Store: 10001
🏷️  Department: 5 (GROCERY)
```

## Testing

After updating your data, test with queries like:

1. **Simple query**: "What's the forecast for Maryland tomorrow?"
   - Should return metadata if available

2. **Specific query**: "Show me cases for store 10001 on March 14, 2024"
   - Should filter and return specific metadata

3. **Department query**: "Forecast for department 5 in Virginia on Jan 1, 2025"
   - Should extract and display department info

## Troubleshooting

### Metadata not showing up
- Check that your CSV has the metadata columns
- Verify column names match exactly: `dept_id`, `state_name`, `store_id`, `dept_desc`
- Restart both servers after updating the CSV

### Filtering not working
- Ensure data types are correct (dept_id and store_id should be integers)
- Check for case sensitivity in state names (use uppercase: MD, VA, DE, NH)
- Verify no extra spaces in the data

### Script errors
- Make sure your source file path is correct
- Check that all required columns exist in your source data
- Look at the error message for specific missing columns

## Files Modified

- [app.py:430-472](app.py#L430-L472) - Updated `get_historical_values_if_available()` function
- [app.py:716-733](app.py#L716-L733) - Updated historical check to use metadata
- [app.py:678-695](app.py#L678-L695) - Updated search queries to use metadata
- [add_metadata_to_data.py](add_metadata_to_data.py) - New helper script

## Next Steps

1. Locate your original data source with metadata
2. Run the `add_metadata_to_data.py` script
3. Backup and replace your current CSV
4. Restart servers
5. Test queries with metadata
6. Enjoy enhanced forecasts with full context!

## Questions?

If you run into issues or need clarification, check:
- Column names in your source data
- Data types (integers for IDs, strings for names)
- That you've restarted servers after updating the CSV
