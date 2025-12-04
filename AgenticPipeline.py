# ==========================================
# LOAD SAVED MODELS AND PREDICT
# ==========================================
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor

# -----------------------------
# 1) Load best saved models
# -----------------------------
cases_model_path = "saved_models/best_cases_catboost_ts_cv.cbm"
trucks_model_path = "saved_models/best_trucks_ts_cv.pkl"

best_cases_model = CatBoostRegressor()
best_cases_model.load_model(cases_model_path)

with open(trucks_model_path, "rb") as f:
    best_trucks_model = pickle.load(f)

print("Loaded models:")
print(f"  CASES  -> {cases_model_path}")
print(f"  TRUCKS -> {trucks_model_path}")

# ------------------------------------------------
# 2) Example: predict on existing TEST feature set
#    (X_test_np and test_df already exist)
# ------------------------------------------------
cases_pred_test  = best_cases_model.predict(X_test_np)
trucks_pred_test = best_trucks_model.predict(X_test_np)

# Rounded trucks prediction (to see discrete truck counts)
trucks_pred_test_round = np.clip(
    np.round(trucks_pred_test),
    np.min(y_train_trucks),   # lower bound from training labels
    np.max(y_train_trucks)    # upper bound from training labels
).astype(int)

# Build a small prediction frame to inspect
pred_test_df = test_df[["dt", "store_id", "dept_id"]].copy().reset_index(drop=True)
pred_test_df["cases_pred"]            = cases_pred_test
pred_test_df["trucks_pred"]           = trucks_pred_test
pred_test_df["trucks_pred_rounded"]   = trucks_pred_test_round

print("\nSample predictions on TEST:")
print(pred_test_df.head(10))

# ------------------------------------------------
# 3) Helper: predict on NEW data with same features
# ------------------------------------------------
def predict_cases_trucks_for_features(X_new):
    """
    X_new: 
      - either a numpy array with shape (n_samples, n_features) matching training
      - or a pandas DataFrame containing all `feature_cols` used in training
    
    Returns:
      cases_pred, trucks_pred, trucks_pred_rounded
    """
    if isinstance(X_new, pd.DataFrame):
        # ensure correct column order
        X_arr = X_new[feature_cols].values.astype(np.float32)
    else:
        X_arr = np.asarray(X_new, dtype=np.float32)
    
    cases_hat  = best_cases_model.predict(X_arr)
    trucks_hat = best_trucks_model.predict(X_arr)
    
    trucks_hat_round = np.clip(
        np.round(trucks_hat),
        np.min(y_train_trucks),
        np.max(y_train_trucks)
    ).astype(int)
    
    return cases_hat, trucks_hat, trucks_hat_round

# ------------------------------------------------
# 4) (Optional) Example usage on some subset of TEST
# ------------------------------------------------
X_example = X_test.iloc[:5].copy()
cases_hat_ex, trucks_hat_ex, trucks_hat_round_ex = predict_cases_trucks_for_features(X_example)

print("\nExample prediction call on 5 rows:")
example_out = X_example.copy()
example_out["cases_pred"]          = cases_hat_ex
example_out["trucks_pred"]         = trucks_hat_ex
example_out["trucks_pred_rounded"] = trucks_hat_round_ex
print(example_out[["cases_pred", "trucks_pred", "trucks_pred_rounded"]])
