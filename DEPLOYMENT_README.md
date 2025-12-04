# 🚀 Deployment Instructions for Agentic Forecast API

## Overview
This project consists of two components:
- **FastAPI Backend** - ML model predictions and intelligent query handling
- **Streamlit Frontend** - Chat interface for users

## 🔧 FastAPI Backend Deployment

### Option 1: Render (Recommended)

1. **Sign up for render.com**

2. **Create New Web Service:**
   - Choose "Web Service" from dashboard
   - Connect your GitHub repository
   - Select branch and folder

3. **Configure Service:**
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables:**
   - `GOOGLE_API_KEY` = your Gemini API key
   - `MODEL_NAME` = `gemini-1.5-flash-latest`

5. **Deploy!** Render will build and deploy automatically

### Option 2: Railway

1. **Sign up for Railway.app**
2. **Deploy the API:**
```bash
railway init
railway up
```
3. **Set Environment Variables:**
```bash
railway variables set GOOGLE_API_KEY="your-api-key-here"
railway variables set MODEL_NAME="gemini-1.5-flash-latest"
```
4. **Get your URL:**
```bash
railway domain
# Example: https://your-api.railway.app
```

## 🎨 Streamlit Frontend Deployment

### Streamlit Cloud

1. **Go to share.streamlit.io**
2. **Connect GitHub repository**
3. **Select Interface.py as the main file**
4. **Set environment variable:**
   ```
   API_URL=https://your-deployed-api-url/predict
   ```

### Alternative: Railway/Render for Frontend

```bash
# Add packages.txt
streamlit

# Deploy Interface.py on Railway/Render
# Set API_URL environment variable to your FastAPI backend URL
```

## 🔐 Environment Variables

### Required:
```bash
GOOGLE_API_KEY=your-gemini-api-key
MODEL_NAME=gemini-1.5-flash-latest

# For frontend:
API_URL=https://your-fastapi-backend-url/predict
```

### Multiple API Keys (for rotation):
```bash
GOOGLE_API_KEY=key1,key2,key3
```

## 📁 Project Structure

```
Agentic-Forecast-Pipeline/
├── app.py                    # FastAPI backend with ML models
├── Interface.py             # Streamlit frontend
├── requirements.txt         # Python dependencies
├── railway.toml            # Railway deployment config
├── .gitignore              # Git ignore rules
├── best_trucks_lgbm.pkl    # LightGBM model
├── best_cases_catboost_ts_cv.cbm  # CatBoost model
├── date_trucks_cases.csv   # Historical data
└── feature_means.json      # Feature normalization
```

## 🧪 Testing Deployment

### Health Check:
```bash
curl https://your-api-url/health
# Expected: {"status": "healthy"}
```

### API Test:
```bash
curl -X POST "https://your-api-url/predict" \
     -H "Content-Type: application/json" \
     -d '{"query": "what is the forecast for tomorrow in DE"}'
```

## 🚨 Security Notes

- `.env` files are ignored by git for security
- Never commit API keys to repositories
- Use Railway/Render environment variables for secrets

## 💡 Features

- **Intelligent Query Processing**: Conversations, aggregations, store lookups
- **Multi-turn Context**: Remembers previous queries for better responses
- **Fallback Handling**: Graceful degradation when services are unavailable
- **Comprehensive Validation**: Date parsing, state mapping, and data validation

---

**Deployed API** 🎯 will be available at: `https://your-domain.com/predict`

**Frontend App** 🎨 will be available at: `https://your-streamlit-app.streamlit.app`
