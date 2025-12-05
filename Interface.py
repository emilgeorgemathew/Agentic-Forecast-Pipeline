import streamlit as st
import requests
import json
from datetime import datetime
import dateparser
import os
import pandas as pd

# -----------------------------------------------------------------------------
# PAGE CONFIG - Must be called FIRST before any other Streamlit commands
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Forecast App",
    layout="centered",
    page_icon="🧠"
)

# Use environment variable for API URL (supports deployment to different backends)
# For Streamlit Cloud: Set API_URL in Secrets section
# For Render: Set API_URL environment variable
# Defaults to Render URL for production stability if secrets fail
PRODUCTION_URL = "https://agentic-forecast-pipeline-1.onrender.com/predict"
LOCAL_URL = "http://127.0.0.1:8000/predict"

try:
    # Try secrets, then env, then fallback to PRODUCTION URL (not localhost)
    API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", PRODUCTION_URL))
except:
    API_URL = os.environ.get("API_URL", PRODUCTION_URL)

USER_AVATAR = """data:image/svg+xml;utf8,<svg width='64' height='64' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><g fill='%23111826'><circle cx='32' cy='22' r='12'/><path d='M12 54c0-11 8.5-18 20-18s20 7 20 18v4H12z' /></g></svg>"""
ASSISTANT_AVATAR = """data:image/svg+xml;utf8,<svg width='64' height='64' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><g fill='%23111826'><rect x='12' y='14' width='40' height='36' rx='8' ry='8'/><rect x='24' y='10' width='16' height='8' rx='2'/><circle cx='24' cy='32' r='4' fill='%23f8fafc'/><circle cx='40' cy='32' r='4' fill='%23f8fafc'/><rect x='24' y='40' width='16' height='4' rx='2' fill='%23f8fafc'/></g></svg>"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Single light theme
bg_gradient = "linear-gradient(135deg, #f5f5f7, #e8e8ed, #d1d1d6, #f5f5f7)"
text_primary = "#000000"
text_secondary = "#6e6e73"
title_gradient = "linear-gradient(135deg, #1d1d1f 0%, #0066cc 100%)"
glass_bg = "rgba(255, 255, 255, 0.7)"
glass_border = "rgba(0, 0, 0, 0.1)"
glass_hover_bg = "rgba(255, 255, 255, 0.9)"
glass_hover_border = "rgba(0, 0, 0, 0.15)"
user_msg_bg = "rgba(0, 122, 255, 0.12)"
user_msg_border = "#b7c6d8"
assistant_msg_bg = "rgba(255, 255, 255, 0.9)"
assistant_msg_border = "#c2c9d4"
btn_bg = "rgba(0, 0, 0, 0.05)"
btn_border = "rgba(0, 0, 0, 0.1)"
btn_hover = "rgba(0, 0, 0, 0.08)"
particle_1 = "rgba(0, 122, 255, 0.08)"
particle_2 = "rgba(150, 100, 255, 0.06)"
particle_3 = "rgba(100, 150, 255, 0.05)"
info_bg = "rgba(0, 122, 255, 0.25)"
info_border = "rgba(0, 122, 255, 0.5)"
error_bg = "rgba(255, 59, 48, 0.15)"
error_border = "rgba(255, 59, 48, 0.35)"
bottom_bar_bg = "#ffffff"
input_container_bg = "#ffffff"
input_border = "#8c95a3"
avatar_bg = "#f3f4f6"
avatar_border = "#cbd5e1"
avatar_icon = "#0f172a"

# Custom CSS for iOS-inspired glass design
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app background with animated gradient */
    body, .stApp {{
        background: {bg_gradient};
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        font-family: 'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Title styling with gradient */
    h1 {{
        background: {title_gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }}

    h3 {{
        color: {text_secondary};
        font-weight: 400;
        font-size: 1rem;
        margin-top: 0;
        opacity: 0.9;
    }}

    /* iOS-style glass button */
    .stButton > button {{
        background: {btn_bg} !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid {btn_border} !important;
        border-radius: 50% !important;
        color: {text_primary} !important;
        font-size: 1.2rem !important;
        width: 48px !important;
        height: 48px !important;
        padding: 0 !important;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        background: {btn_hover} !important;
        border: 1px solid {glass_hover_border} !important;
        transform: scale(1.05);
        box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.3);
    }}

    .stButton > button:active {{
        transform: scale(0.95);
    }}

    /* Glass morphism for chat messages */
    .stChatMessage {{
        background-color: transparent !important;
        margin: 12px 0;
        display: flex;
        justify-content: center;
    }}

    /* iOS-style glass effect for message bubbles */
    [data-testid="stChatMessageContent"] {{
        background: {assistant_msg_bg};
        border: 1px solid {assistant_msg_border};
        border-radius: 18px;
        padding: 16px 20px;
        color: {text_primary};
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.12);
        width: 90%;
        max-width: 1100px;
    }}

    /* Strip any inner borders/boxes inside chat bubbles */
    [data-testid="stChatMessageContent"] div,
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] pre,
    [data-testid="stChatMessageContent"] code {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* User message - iOS blue */
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
        background: {user_msg_bg};
        border: 1px solid {user_msg_border};
    }}

    /* Assistant message */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
        background: linear-gradient(135deg, rgba(30,64,175,0.32), rgba(59,130,246,0.28), rgba(99,102,241,0.24));
        border: 1px solid {assistant_msg_border};
    }}

    /* Bottom bar container */
    [data-testid="stBottom"],
    [data-testid="stChatInputRoot"],
    [data-testid="stChatInput"] > div:first-child {{
        background: {bottom_bar_bg} !important;
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border-top: 1px solid {glass_border};
    }}

    /* iOS-style glass chat input */
    .stChatInputContainer {{
        background: {input_container_bg} !important;
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border: 2px solid {input_border} !important;
        border-radius: 18px;
        padding: 6px;
        box-shadow: 0 6px 18px 0 rgba(0, 0, 0, 0.28);
    }}

    /* Fix for weird labels and error states */
    .stChatInput label {{
        display: none !important;
    }}
    
    .stChatInputContainer[data-testid="stChatInput"] {{
        border-color: {input_border} !important;
    }}

    .stChatInput textarea {{
        background-color: transparent !important;
        color: {text_primary} !important;
        border: none !important;
        font-size: 0.95rem;
    }}

    .stChatInput textarea::placeholder {{
        color: {text_secondary} !important;
        opacity: 0.7;
    }}

    /* Additional override for text input visibility - CRITICAL */
    .stChatInput input[type="text"],
    .stChatInput textarea,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    .stChatInputContainer textarea,
    .stChatInputContainer input {{
        color: {text_primary} !important;
        background: transparent !important;
        -webkit-text-fill-color: {text_primary} !important;
    }}

    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {text_secondary} !important;
        opacity: 0.7 !important;
    }}

    /* Divider with gradient */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {glass_border}, transparent);
        margin: 1.5rem 0;
    }}

    /* Glass metrics containers */
    [data-testid="metric-container"] {{
        background: {glass_bg};
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid {glass_border};
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }}

    [data-testid="stMetricValue"] {{
        color: {text_primary};
        font-size: 2rem;
        font-weight: 700;
    }}

    [data-testid="stMetricLabel"] {{
        color: {text_secondary};
        font-size: 0.9rem;
        font-weight: 500;
    }}

    /* Glass alert boxes */
    .stAlert {{
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 16px !important;
        font-weight: 500 !important;
    }}

    /* Error alert styling */
    [data-testid="stNotificationContentError"] {{
        background: {error_bg} !important;
        border: 1px solid {error_border} !important;
        color: {text_primary} !important;
        box-shadow: 0 8px 32px 0 rgba(255, 0, 0, 0.15) !important;
    }}

    /* Info alert styling with better contrast */
    [data-testid="stNotificationContentInfo"] {{
        background: {info_bg} !important;
        border-top-color: #0066cc !important;
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {glass_border};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {glass_hover_border};
    }}

    /* Column spacing */
    [data-testid="column"] {{
        padding: 0 8px;
    }}

    /* Avatar styling - simple, no boxes */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 40px;
        height: 40px;
        padding: 0;
    }}

    [data-testid="chatAvatarIcon-user"] *,
    [data-testid="chatAvatarIcon-assistant"] * {{
        color: {avatar_icon} !important;
        fill: {avatar_icon} !important;
        stroke: {avatar_icon} !important;
    }}

    [data-testid="chatAvatarIcon-user"] img,
    [data-testid="chatAvatarIcon-assistant"] img {{
        width: 36px;
        height: 36px;
        object-fit: contain;
    }}

    /* Animated background particles */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            radial-gradient(circle at 20% 30%, {particle_1} 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, {particle_2} 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, {particle_3} 0%, transparent 50%);
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }}

    @keyframes particleFloat {{
        0%, 100% {{
            transform: translate(0, 0) scale(1);
            opacity: 0.3;
        }}
        33% {{
            transform: translate(30px, -30px) scale(1.1);
            opacity: 0.5;
        }}
        66% {{
            transform: translate(-20px, 20px) scale(0.9);
            opacity: 0.4;
        }}
    }}

    /* Ensure content is above particles */
    [data-testid="stAppViewContainer"] > section {{
        position: relative;
        z-index: 1;
    }}

    /* Hide Streamlit header anchor links */
    .stApp a[href^="#"] {{
        display: none !important;
    }}
    
    /* Specific selector for newer Streamlit versions */
    [data-testid="stHeaderActionElements"] {{
        display: none !important;
    }}
    
    /* Hide anchors inside headers specifically */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {{
        display: none !important;
    }}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("<h1>Agentic AI Powered Forecast Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6e6e73; margin-top: -10px; font-size: 1.1rem;'>Retail Cases and Trucks Forecasting</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS - MUST BE DEFINED BEFORE USE
# -----------------------------------------------------------------------------
def normalize_date(text: str) -> str:
    # Dateparser settings for robust parsing
    parsing_settings = {
        "RELATIVE_BASE": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "PREFER_DATES_FROM": "future"
    }

    # 1. Try parsing the full query
    dt = dateparser.parse(text, languages=['en'], settings=parsing_settings)
    if dt is not None:
        return dt.date().isoformat()

    # 2. Try to grab date-like chunks
    import re
    patterns = [
        r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}",  # 2025-11-17 or 2025/11/17
        r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",  # 17/11/2025
        r"\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{2,4}",  # 9th September 2025
        r"\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}",  # September 9th, 2025
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            chunk = m.group(0)
            dt2 = dateparser.parse(chunk, languages=['en'], settings=parsing_settings)
            if dt2 is not None:
                return dt2.date().isoformat()

    # 3. Try relative terms
    relative_phrases = [
        "next month", "last month", "next week", "last week",
        "next friday", "last friday", "tomorrow", "yesterday",
        "next year", "last year", "today", "now"
    ]
    for phrase in relative_phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text, flags=re.IGNORECASE):
            dt_rel = dateparser.parse(phrase, languages=['en'], settings=parsing_settings)
            if dt_rel is not None:
                return dt_rel.date().isoformat()

    return None

def display_prediction_response(data, key_suffix=None):
    """Display prediction data using Streamlit components"""
    
    # Generate a unique key for widgets if not provided
    if key_suffix is None:
        import uuid
        key_suffix = str(uuid.uuid4())[:8]
    
    # Check if this is a state breakdown response
    if data.get("breakdown_data"):
        st.markdown(f"### 📋 State Breakdown: {data.get('state')}")
        st.markdown(f"**📅 Date:** {data.get('date')}")
        
        # Metrics for the whole state
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Estimated Cases", f"{data.get('Cases'):,.0f}")
        with col2:
            st.metric("Total Required Trucks", f"{data.get('trucks'):,.0f}")
            
        st.markdown("---")
        
        # Prepare DataFrame
        breakdown_df = pd.DataFrame(data["breakdown_data"])
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📄 Detailed Data", "📥 Export"])
        
        with tab1:
            st.markdown("#### Store Performance")
            
            # Aggregate by store for clearer charts
            store_agg = breakdown_df.groupby("store_id")[["cases", "trucks"]].sum().reset_index()
            store_agg["store_id"] = store_agg["store_id"].astype(str)  # Convert to string for categorical plotting
            
            # Chart 1: Cases by Store
            st.caption("Total Cases by Store")
            st.bar_chart(store_agg.set_index("store_id")["cases"], color="#0066cc")
            
            # Chart 2: Trucks by Store
            st.caption("Required Trucks by Store")
            st.bar_chart(store_agg.set_index("store_id")["trucks"], color="#FF4B4B")
            
        with tab2:
            st.markdown("#### Store & Department Breakdown")
            
            # Rename columns for better display
            display_df = breakdown_df.rename(columns={
                "store_id": "Store ID",
                "dept_id": "Dept ID",
                "dept_desc": "Department",
                "cases": "Cases",
                "trucks": "Trucks",
                "source": "Source"
            })
            
            # Reorder columns
            cols = ["Store ID", "Dept ID", "Department", "Cases", "Trucks", "Source"]
            display_df = display_df[cols]
            
            # Group by Store ID for clearer display
            unique_stores = sorted(display_df["Store ID"].unique())
            
            for store_id in unique_stores:
                st.markdown(f"#### 🏪 Store {store_id}")
                
                # Filter for this store
                store_data = display_df[display_df["Store ID"] == store_id].drop(columns=["Store ID"])
                
                # Display interactive table for this store
                st.dataframe(
                    store_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Dept ID": st.column_config.NumberColumn(format="%d"),
                        "Cases": st.column_config.NumberColumn(
                            format="%.0f",
                            help="Estimated number of cases"
                        ),
                        "Trucks": st.column_config.ProgressColumn(
                            "Trucks",
                            help="Number of trucks required (0-4 scale)",
                            format="%.1f",
                            min_value=0,
                            max_value=4,
                        ),
                        "Source": st.column_config.TextColumn(
                            "Data Source",
                            help="Historical data or Model prediction"
                        )
                    }
                )
                st.divider()
            
        with tab3:
            st.markdown("#### Download Data")
            st.markdown("Download the full breakdown data as a CSV file for further analysis.")
            
            csv = breakdown_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"forecast_{data.get('state')}_{data.get('date')}.csv",
                mime="text/csv",
                key=f"download_csv_{key_suffix}"
            )
        
        if data.get("message"):
            st.info(data.get("message"))
            
        return

    st.markdown("### ✅ Prediction Ready")

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🗳 Estimated Cases", round(data['Cases'], 2))
    with col2:
        st.metric("🚚 Required Trucks", int(data['trucks']))

    st.markdown("---")

    def fmt(val, fallback="N/A"):
        return val if val not in (None, "", []) else fallback
    
    # 2x3 grid (2 columns, 3 rows)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**📅 Forecast Date:** {fmt(data.get('date'))}")
        st.markdown(f"**🏠 State:** {fmt(data.get('state'))}")
        source_label = "📈 Model Forecast" if data.get("source") in ["model", "model_search"] else "📜 Historical Value"
        st.markdown(f"**📘 Data Source:** {fmt(source_label)}")

    with c2:
        # Always show these fields, even if N/A to maintain consistent display
        st.markdown(f"**🏪 Store ID:** {fmt(data.get('store_id'))}")
        st.markdown(f"**🏢 Dept ID:** {fmt(data.get('dept_id'))}")
        st.markdown(f"**🏢 Dept Name:** {fmt(data.get('dept_name'))}")

    # Additional context if available
    if data.get("message"):
        st.info(data.get("message"))

    st.caption("Prediction generated using advanced time-series models.")

# -----------------------------------------------------------------------------
# CHAT DISPLAY
# -----------------------------------------------------------------------------
# Display chat messages from history
for i, message in enumerate(st.session_state.messages):
    avatar_choice = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar_choice):
        if message.get("type") == "prediction":
            display_prediction_response(message["data"], key_suffix=f"hist_{i}")
        elif message.get("type") == "info":
            st.info(message["content"])
        else:
            st.markdown(message["content"])

# -----------------------------------------------------------------------------
# CHAT INPUT & PROCESSING
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask about future cases and trucks (e.g., 'Forecast for Store 10001 next Friday')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Display assistant response with spinner
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Analyzing data..."):
            try:
                # Prepare payload
                payload = {
                    "query": prompt,
                    "conversation_history": [
                        {"role": m["role"], "content": str(m.get("content", "")) or str(m.get("data", ""))}
                        for m in st.session_state.messages[-5:]  # Send last 5 messages for context
                    ]
                }
                
                # Call API
                r = requests.post(API_URL, json=payload)
                
                if r.status_code == 200:
                    # Success - process response
                    data = r.json()

                    if "message" in data and data["message"] and not (data.get("breakdown_data") or data.get("source") in ["model", "model_search", "breakdown"]):
                        # Info message from API (only if NOT a prediction/breakdown)
                        st.info(data["message"])
                        st.session_state.messages.append({"role": "assistant", "type": "info", "content": data["message"]})
                    else:
                        # Prediction data (or breakdown with message)
                        # Use a new key for the current prediction
                        current_key = f"curr_{len(st.session_state.messages)}"
                        display_prediction_response(data, key_suffix=current_key)
                        st.session_state.messages.append({"role": "assistant", "type": "prediction", "data": data})

                else:
                    # API Error
                    error_msg = f"Error: {r.status_code} - {r.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except Exception as e:
                # Connection/Code Error
                error_msg = f"Connection Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
