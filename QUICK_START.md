# Quick Start: Export LightGBM Model with Metadata

## ✅ What I've Done

I've created everything you need to export your LightGBM model with full metadata support (dept_id, state_name, store_id, dept_desc).

## 📁 Files Created

1. **`LightGBM_Final_Forecasting_V2_with_export.ipynb`** (in Downloads/wat/)
   - Your original notebook PLUS a new export cell at the end
   - **This is the file you should use!**

2. **Helper Scripts** (in Agentic-Forecast-Pipeline/):
   - `export_lgbm_notebook_code.py` - Standalone export code
   - `export_lgbm_model.py` - Alternative export script
   - `add_export_cell.py` - Script that added the cell (already run)

3. **Documentation**:
   - `EXPORT_LGBM_README.md` - Complete deployment guide
   - `METADATA_UPDATE_GUIDE.md` - General metadata guide
   - `QUICK_START.md` - This file

## 🚀 How to Use (3 Simple Steps)

### Step 1: Run the Notebook

```bash
# Open the NEW notebook (the one WITH export cell)
cd /Users/emilgeorgemathew/Downloads/wat/
jupyter notebook LightGBM_Final_Forecasting_V2_with_export.ipynb

# OR if using Google Colab:
# Upload LightGBM_Final_Forecasting_V2_with_export.ipynb to Colab
```

### Step 2: Execute All Cells

In Jupyter/Colab:
- Click: **Cell > Run All** (or Runtime > Run all in Colab)
- Wait for all cells to complete
- The **LAST cell** will export 3 files:
  - ✅ `best_trucks_lgbm.pkl` - Your model
  - ✅ `date_trucks_cases.csv` - Historical data with metadata
  - ✅ `feature_means.json` - Feature defaults

### Step 3: Deploy

The export cell will print instructions, but here's the summary:

```bash
cd /Users/emilgeorgemathew/Agentic-Forecast-Pipeline

# Backup current files (optional but recommended)
cp best_trucks_ts_cv.pkl best_trucks_ts_cv.pkl.backup
cp date_trucks_cases.csv date_trucks_cases.csv.backup

# The new files are already in the right place!
# Just update app.py to use the new model:
```

Edit `app.py` line ~95:
```python
# Change from:
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_ts_cv.pkl")

# To:
TRUCKS_MODEL_PATH = os.environ.get("TRUCKS_MODEL_PATH", "best_trucks_lgbm.pkl")
```

Restart servers:
```bash
lsof -ti:8000 -ti:8501 | xargs kill -9
python3 -m uvicorn app:app --reload --port 8000 &
streamlit run Interface.py --server.port 8501 &
```

## ✨ What You'll Get

After deployment, your bot will display full metadata:

```
📊 Forecast for 2025-01-01

Cases: 85.3
Trucks: 3

📍 State: MD
🏪 Store: 10001
🏷️ Department: 5 (GROCERY)

Source: Model prediction
```

Test with queries like:
- "What's the forecast for Maryland tomorrow?"
- "Show me forecast for store 10001 on Jan 1 2025"
- "What's department 5 forecast in Virginia?"

## 🔍 What the Export Cell Does

The new cell at the end of your notebook:

1. **Finds your trained model**: `final_lgbm`
2. **Finds your data**: `merged_data3` (or similar)
3. **Exports 3 files** with all required metadata
4. **Validates** everything is correct
5. **Prints** deployment instructions

## ⚠️ Troubleshooting

### "Model variable not found"
The cell tries to find `final_lgbm`. If your variable has a different name, edit the cell:
```python
MODEL = your_model_variable_name  # Update this line
```

### "Data variable not found"
The cell tries `merged_data3`, `full_data`, `final_df`. If you use a different name, edit:
```python
DATA = your_data_variable_name  # Update this line
```

### "Missing columns"
Your data needs these columns:
- `dt`, `trucks`, `cases` (basic)
- `dept_id`, `state_name`, `store_id`, `dept_desc` (metadata)

If missing, check your data merging steps in the notebook.

## 📚 More Help

- Full guide: [EXPORT_LGBM_README.md](EXPORT_LGBM_README.md)
- Metadata details: [METADATA_UPDATE_GUIDE.md](METADATA_UPDATE_GUIDE.md)

## ✅ Checklist

- [ ] Open `LightGBM_Final_Forecasting_V2_with_export.ipynb`
- [ ] Run all cells (Cell > Run All)
- [ ] Last cell exports 3 files successfully
- [ ] Update `app.py` to use `best_trucks_lgbm.pkl`
- [ ] Restart servers
- [ ] Test with metadata queries
- [ ] Verify bot shows dept_id, state_name, etc.

## 🎉 Success!

Once complete, you'll have:
- ✅ LightGBM model deployed
- ✅ Full metadata support (dept_id, state_name, store_id, dept_desc)
- ✅ Better forecasting with context

Questions? Check the troubleshooting section in [EXPORT_LGBM_README.md](EXPORT_LGBM_README.md)
