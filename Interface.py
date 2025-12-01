import streamlit as st
import requests
import json
from datetime import datetime
import dateparser
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")
USER_AVATAR = """data:image/svg+xml;utf8,<svg width='64' height='64' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><g fill='%23111826'><circle cx='32' cy='22' r='12'/><path d='M12 54c0-11 8.5-18 20-18s20 7 20 18v4H12z' /></g></svg>"""
ASSISTANT_AVATAR = """data:image/svg+xml;utf8,<svg width='64' height='64' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><g fill='%23111826'><rect x='12' y='14' width='40' height='36' rx='8' ry='8'/><rect x='24' y='10' width='16' height='8' rx='2'/><circle cx='24' cy='32' r='4' fill='%23f8fafc'/><circle cx='40' cy='32' r='4' fill='%23f8fafc'/><rect x='24' y='40' width='16' height='4' rx='2' fill='%23f8fafc'/></g></svg>"""

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Forecast App",
    layout="centered",
    page_icon="🧠"
)

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
        border: 1px solid {info_border} !important;
        color: {text_primary} !important;
        box-shadow: 0 8px 32px 0 rgba(0, 122, 255, 0.2) !important;
    }}

    /* Alert text */
    .stAlert > div {{
        color: {text_primary} !important;
    }}

    /* Markdown styling */
    .stMarkdown {{
        color: {text_primary};
    }}

    .stMarkdown strong {{
        color: {text_primary};
        font-weight: 600;
    }}

    /* Caption styling */
    .stCaptionContainer {{
        color: {text_secondary} !important;
        font-style: italic;
        font-size: 0.85rem;
    }}

    /* Spinner */
    .stSpinner > div {{
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

</style>
""", unsafe_allow_html=True)

st.title("Agentic AI Powered Forecast Engine")
st.markdown("### Retail Cases and Trucks Forecasting")
st.divider()

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

def display_prediction_response(data):
    """Display prediction data using Streamlit components"""
    st.markdown("### ✅ Prediction Ready")

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🗳 Estimated Cases", round(data['Cases'], 2))
    with col2:
        st.metric("🚚 Required Trucks", int(data['trucks']))

    st.markdown("---")

    # Details in two columns (2x4-style grid)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**📅 Forecast Date:** {data['date']}")
        if data.get("state"):
            st.markdown(f"**🏠 State:** {data['state']}")
        source_label = "📈 Model Forecast" if data["source"] in ["model", "model_search"] else "📜 Historical Value"
        st.markdown(f"**📘 Data Source:** {source_label}")

    with right:
        if data.get("dept_name"):
            st.markdown(f"**🏢 Dept Name:** {data['dept_name']}")
        if data.get("dept_id"):
            st.markdown(f"**🏢 Dept ID:** {data['dept_id']}")
        if data.get("store_id"):
            st.markdown(f"**🏪 Store ID:** {data['store_id']}")

    st.caption("Prediction generated using advanced time-series models.")

# -----------------------------------------------------------------------------
# CHAT DISPLAY
# -----------------------------------------------------------------------------
# Display chat messages from history
for message in st.session_state.messages:
    avatar_choice = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar_choice):
        if message.get("type") == "prediction":
            display_prediction_response(message["data"])
        elif message.get("type") == "error":
            st.error(message["content"])
        elif message.get("type") == "info":
            st.info(message["content"])
        else:
            st.markdown(message["content"])

# -----------------------------------------------------------------------------
# CHAT INPUT
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask anything", key="chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                # Extract previous user queries for conversation context
                conversation_history = [
                    msg["content"]
                    for msg in st.session_state.messages
                    if msg["role"] == "user"
                ]

                # Make API request with conversation history
                r = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                        "conversation_history": conversation_history
                    },
                    timeout=20
                )

                if r.status_code != 200:
                    # Handle API errors in chat
                    try:
                        error_data = r.json()
                        error_detail = error_data.get("detail", r.text)
                        error_msg = f"❌ **Error ({r.status_code})**\n\n{error_detail}"
                    except:
                        error_msg = f"❌ **Error ({r.status_code})**\n\n{r.text}"

                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "type": "error", "content": error_msg})
                else:
                    # Success - process response
                    data = r.json()

                    if "message" in data and data["message"]:
                        # Info message from API
                        st.info(data["message"])
                        st.session_state.messages.append({"role": "assistant", "type": "info", "content": data["message"]})
                    else:
                        # Prediction data
                        display_prediction_response(data)
                        st.session_state.messages.append({"role": "assistant", "type": "prediction", "data": data})

            except requests.exceptions.RequestException as e:
                error_msg = f"❌ **API Unreachable**\n\n{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "type": "error", "content": error_msg})
            except Exception as e:
                error_msg = f"❌ **Error**\n\n{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "type": "error", "content": error_msg})
