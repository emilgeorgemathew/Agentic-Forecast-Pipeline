# Exporting LightGBM Model with Metadata

This guide shows you how to export your LightGBM model from the notebook and deploy it with full metadata support (dept_id, state_name, store_id, dept_desc).

## Quick Start (Easiest Method)

### Option 1: Copy-Paste into Notebook (Recommended)

1. **Open your notebook**: `LightGBM_Final_Forecasting_V2.ipynb`

2. **Find your final trained model variable**. Look for lines like:
   ```python
   final_lgbm = LGBMRegressor(**best_params_red)
   final_lgbm.fit(...)
   ```
   The variable name is `final_lgbm` in this example.

3. **Find your full dataset variable**. Look for your merged data, usually called:
   ```python
   merged_data3 = ...  # or full_data, final_df, etc.
   ```

4. **Open the export code**: [export_lgbm_notebook_code.py](export_lgbm_notebook_code.py)

5. **Copy the entire contents** of that file

6. **Paste into a new cell** at the end of your notebook

7. **Update the variable names** (lines 14-18):
   ```python
   MODEL = final_lgbm  # <-- Change to your model variable
   DATA = merged_data3  # <-- Change to your data variable
   ```

8. **Run the cell** - it will export:
   - ✅ `best_trucks_lgbm.pkl` - Your trained model
   - ✅ `date_trucks_cases.csv` - Historical data with metadata
   - ✅ `feature_means.json` - Feature defaults

### Option 2: Run as Standalone Script

If you prefer to run outside the notebook:

1. **Save your model and data** from the notebook:
   ```python
   # In your notebook, run this:
   import pickle
   with open('my_model.pkl', 'wb') as f:
       pickle.dump(final_lgbm, f)

   merged_data3.to_pickle('my_data.pkl')
   ```

2. **Modify [export_lgbm_model.py](export_lgbm_model.py)** to load these files

3. **Run the script**:
   ```bash
   python3 export_lgbm_model.py
   ```

## What Gets Exported

### 1. Model File: `best_trucks_lgbm.pkl`
- Your trained LightGBM model
- Can be loaded with `pickle.load()`
- Used for truck predictions

### 2. Historical Data: `date_trucks_cases.csv`
Contains 7 columns:
- `dt` - Date (datetime)
- `trucks` - Number of trucks (float)
- `cases` - Number of cases (float)
- `dept_id` - Department ID (int)
- `state_name` - State code: MD, VA, DE, NH (string)
- `store_id` - Store ID (int)
- `dept_desc` - Department description (string)

Example:
```csv
dt,trucks,cases,dept_id,state_name,store_id,dept_desc
2024-03-14,3.0,81.0,5,MD,10001,GROCERY
2024-03-14,2.0,52.0,2,VA,10002,PRODUCE
```

### 3. Feature Means: `feature_means.json`
Default values for all numeric features used when making predictions:
```json
{
  "numeric_feature_means": {
    "is_weekend": 0.285,
    "lag_35": 52.3,
    ...
  }
}
```

## Deploying to Production

### Step 1: Backup Current Files
```bash
cd /Users/emilgeorgemathew/Agentic-Forecast-Pipeline

# Backup current files
cp best_trucks_ts_cv.pkl best_trucks_ts_cv.pkl.backup
cp date_trucks_cases.csv date_trucks_cases.csv.backup
cp feature_means.json feature_means.json.backup
```

### Step 2: Update app.py to Use LightGBM Model

Edit [app.py](app.py) and find the line (around line 95):
```python
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_ts_cv.pkl")
```

Change it to:
```python
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_lgbm.pkl")
```

**OR** set environment variable in `.env`:
```bash
TRUCKS_MODEL_PATH=best_trucks_lgbm.pkl
```

### Step 3: Restart Servers

The servers should auto-reload with `--reload` flag:
```bash
# If they don't auto-reload, manually restart:
lsof -ti:8000 -ti:8501 | xargs kill -9
python3 -m uvicorn app:app --reload --port 8000 &
streamlit run Interface.py --server.port 8501 &
```

### Step 4: Test the Deployment

Test with queries that use metadata:

1. **Simple forecast**:
   ```
   What's the forecast for tomorrow?
   ```

2. **With state**:
   ```
   What's the forecast for Maryland on Jan 1 2025?
   ```

3. **With store**:
   ```
   Show me forecast for store 10001 tomorrow
   ```

4. **With department**:
   ```
   What's department 5 forecast for Virginia next week?
   ```

The bot should now display:
- 📊 Cases and Trucks predictions
- 📍 State name
- 🏪 Store ID
- 🏷️ Department ID and description

## Troubleshooting

### Error: Model variable not found

**Problem**: The export script can't find your model variable.

**Solution**:
1. Look for your final trained model in the notebook
2. Common variable names: `final_lgbm`, `lgb_model_clean`, `reduced_model`
3. Update the `MODEL` variable in the export code

### Error: Missing columns in data

**Problem**: Your dataset doesn't have all required metadata columns.

**Solution**:
Check which columns you have:
```python
print(merged_data3.columns.tolist())
```

Required columns:
- `dt`, `trucks`, `cases` (must have)
- `dept_id`, `state_name`, `store_id`, `dept_desc` (metadata)

If missing metadata, you may need to:
1. Go back to your data merging steps
2. Ensure you merged with stores data (for state_name, store_id)
3. Ensure you have department mappings (for dept_id, dept_desc)

### Model predictions look wrong

**Problem**: The LightGBM model gives unexpected results.

**Solution**:
1. Verify you exported the correct model (not an intermediate one)
2. Check that feature names match between training and prediction
3. Verify the model was trained on the same features as in `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` in app.py

### Metadata not showing in responses

**Problem**: The API returns predictions but no dept_id, state_name, etc.

**Solution**:
1. Verify the CSV has the metadata columns: `head date_trucks_cases.csv`
2. Check the CSV was loaded correctly by the API
3. Restart the servers to reload the new data
4. Test with a specific query that mentions state/store/dept

## File Structure After Export

```
Agentic-Forecast-Pipeline/
├── app.py                          # Backend API (updated to use LGBM)
├── Interface.py                    # Frontend UI
├── best_trucks_lgbm.pkl           # ✨ NEW: Your LightGBM model
├── date_trucks_cases.csv          # ✨ UPDATED: Now has metadata
├── feature_means.json             # ✨ UPDATED: New feature means
├── best_cases_catboost_ts_cv.cbm  # Cases model (unchanged)
├── export_lgbm_model.py           # Export script (standalone)
├── export_lgbm_notebook_code.py   # Export code (for notebook)
└── EXPORT_LGBM_README.md          # This file
```

## Verification Checklist

After deployment, verify:

- [ ] Model file exists and is ~10-50MB
- [ ] CSV has 7 columns including metadata
- [ ] CSV has multiple unique dept_id values
- [ ] CSV has state codes (MD, VA, DE, NH)
- [ ] CSV has multiple unique store_id values
- [ ] feature_means.json has 40+ features
- [ ] app.py points to correct model file
- [ ] Servers restart successfully
- [ ] API returns predictions with metadata
- [ ] Frontend displays state, store, dept info

## Model Comparison

| Feature | Old (RandomForest) | New (LightGBM) |
|---------|-------------------|----------------|
| Model Type | RandomForestRegressor | LGBMRegressor |
| File | best_trucks_ts_cv.pkl | best_trucks_lgbm.pkl |
| Metadata | ❌ No | ✅ Yes |
| dept_id | ❌ No | ✅ Yes |
| state_name | ❌ No | ✅ Yes |
| store_id | ❌ No | ✅ Yes |
| dept_desc | ❌ No | ✅ Yes |

## Next Steps

Once deployed successfully:

1. **Test thoroughly** with various queries
2. **Monitor predictions** for accuracy
3. **Compare** with old model if needed
4. **Document** any special department/state patterns
5. **Share** with stakeholders

## Need Help?

Check these files:
- [METADATA_UPDATE_GUIDE.md](METADATA_UPDATE_GUIDE.md) - General metadata guide
- [add_metadata_to_data.py](add_metadata_to_data.py) - Helper for adding metadata
- [export_lgbm_notebook_code.py](export_lgbm_notebook_code.py) - Notebook export code

## Success!

Once everything is working, you should see responses like:

```
📊 Forecast for 2025-01-01

Cases: 85.3
Trucks: 3

📍 State: MD
🏪 Store: 10001
🏷️ Department: 5 (GROCERY)

Source: Model prediction
```

🎉 Congratulations! Your LightGBM model is now deployed with full metadata support!
