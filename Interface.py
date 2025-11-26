import streamlit as st
import requests
import json
from datetime import datetime
import dateparser

API_URL = "https://agentic-forecast-pipeline-1.onrender.com/predict"

# -----------------------------------------------------------------------------
# IMAGE URLS (Google Drive - export as download)
# -----------------------------------------------------------------------------
IMAGE_URL_1 = "https://drive.google.com/uc?export=download&id=1HTIDnSe32w3Eh7NSYHVbrTN_ZItn_r6r"
IMAGE_URL_2 = "https://drive.google.com/uc?export=download&id=1K0KiUule39xWMhHb3Cw1qhPGKBQ8c_Ac"


def load_image_bytes(url: str):
    """
    Download image bytes from a URL.
    Returns bytes on success, None on failure.
    """
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
        else:
            return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Forecast App",
    layout="centered",
    page_icon="🧠"
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🧠 Agentic Forecast")
st.markdown(
    """
    ### Trucks & Cases Prediction  
    Ask in natural language or select a date manually.
    """
)

# -----------------------------------------------------------------------------
# IMAGE SECTION
# -----------------------------------------------------------------------------
st.subheader("🔎 App Overview")

img1_bytes = load_image_bytes(IMAGE_URL_1)
if img1_bytes:
    st.image(
        img1_bytes,
        caption="Truck Map",
        use_container_width=True
    )
else:
    st.warning("Could not load Forecast Flow Diagram image.")

img2_bytes = load_image_bytes(IMAGE_URL_2)
if img2_bytes:
    st.image(
        img2_bytes,
        caption="Cases Map",
        use_container_width=True
    )
else:
    st.warning("Could not load Prediction Output Example image.")

st.divider()

# -----------------------------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------------------------
st.subheader("💬 Enter your query")

user_query = st.text_area(
    "",
    placeholder='e.g., "What is the prediction for Maryland store 10001 next Friday?"',
    height=120
)

# DATE PICKER
st.subheader("📅 Optional Date")
picked_date = st.date_input("Select a date (optional)", value=None)

# Toggle
force_date = st.checkbox("Auto-detect & correct dates from text", value=True)

# -----------------------------------------------------------------------------
# DATE NORMALIZATION
# -----------------------------------------------------------------------------
def normalize_date(text: str) -> str:
    dt = dateparser.parse(text)
    if dt is None:
        return None
    return dt.date().isoformat()

# -----------------------------------------------------------------------------
# SUBMIT BUTTON
# -----------------------------------------------------------------------------
if st.button("🚀 Get Prediction"):
    
    if not user_query:
        st.error("Please enter a query first.")
        st.stop()

    final_query = user_query.strip()

    # If user picked date → override
    if picked_date:
        iso_date = picked_date.isoformat()
        final_query += f" on {iso_date}"

    # If no manual date → auto-extract using NLP
    elif force_date:
        detected = normalize_date(final_query)
        if detected:
            final_query += f" on {detected}"

    # Show clean status without showing payload
    with st.status("📡 Contacting prediction engine...", expanded=False):
        try:
            r = requests.post(API_URL, json={"query": final_query}, timeout=20)
        except Exception as e:
            st.error(f" ❌ API unreachable: {e}")
            st.stop()

        if r.status_code != 200:
            st.error(f"❌ API returned {r.status_code}")
            st.write(r.text)
            st.stop()

        data = r.json()

    # -----------------------------------------------------------------------------
    # DISPLAY PREDICTIONS
    # -----------------------------------------------------------------------------
    st.success("✅ Prediction Ready")

    col1, col2 = st.columns(2)
    col1.metric("🗳 Estimated Cases", round(data["Cases"], 2))
    col2.metric("🚚 Required Trucks", int(data["trucks"]))

    st.divider()

    st.subheader("📅 Forecast Date")
    st.markdown(f"**{data['date']}**")

    st.subheader("📘 Data Source Used")
    source_label = "📈 Model Forecast" if data["source"] == "model" else "📜 Historical Value"
    st.markdown(f"**{source_label}**")

    st.caption("Prediction generated using advanced time-series ensemble models.")
