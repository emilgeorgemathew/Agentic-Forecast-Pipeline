# Prediction API (FastAPI)

This project provides a FastAPI service that:
- Accepts a natural-language query from the user.
- Sends the query to the Google Generative API (Gemini / text-bison by default) to extract structured features.
- Loads local models (CatBoost for `cases`, RandomForest for `trucks`) and returns predictions.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Confirm `.env` contains your `GOOGLE_API_KEY` and correct model paths.

3. Run the API:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Example `curl` request (POST):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"query":"Predict cases and trucks for store 123 department 45 on 2025-06-01; typical volume, 2 trucks recent", "overrides": {"store_id":"123","dept_id":"45","month":6}}'
```

Notes
- The Gemini parsing step attempts to transform free text into a JSON of features listed in the model summary. If parsing fails, you can provide a full `overrides` dict with feature values.
