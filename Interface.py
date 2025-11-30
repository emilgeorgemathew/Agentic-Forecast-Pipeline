import streamlit as st
import requests
import json
from datetime import datetime
import dateparser

API_URL = "http://localhost:8000/predict"

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
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Theme colors based on mode
dark_mode = st.session_state.dark_mode

if dark_mode:
    bg_gradient = "linear-gradient(135deg, #000000, #0f0f23, #1a1a2e, #000000)"
    text_primary = "#ffffff"
    text_secondary = "#b0b0c0"
    title_gradient = "linear-gradient(135deg, #ffffff 0%, #a0a0ff 100%)"
    glass_bg = "rgba(255, 255, 255, 0.05)"
    glass_border = "rgba(255, 255, 255, 0.1)"
    glass_hover_bg = "rgba(255, 255, 255, 0.08)"
    glass_hover_border = "rgba(255, 255, 255, 0.15)"
    user_msg_bg = "rgba(0, 122, 255, 0.15)"
    user_msg_border = "rgba(0, 122, 255, 0.3)"
    assistant_msg_bg = "rgba(150, 100, 255, 0.1)"
    assistant_msg_border = "rgba(180, 150, 255, 0.2)"
    btn_bg = "rgba(255, 255, 255, 0.08)"
    btn_border = "rgba(255, 255, 255, 0.15)"
    btn_hover = "rgba(255, 255, 255, 0.12)"
    particle_1 = "rgba(0, 122, 255, 0.15)"
    particle_2 = "rgba(150, 100, 255, 0.12)"
    particle_3 = "rgba(100, 150, 255, 0.1)"
    info_bg = "rgba(0, 122, 255, 0.2)"
    info_border = "rgba(0, 122, 255, 0.4)"
    error_bg = "rgba(255, 100, 100, 0.15)"
    error_border = "rgba(255, 100, 100, 0.3)"
else:
    bg_gradient = "linear-gradient(135deg, #f5f5f7, #e8e8ed, #d1d1d6, #f5f5f7)"
    text_primary = "#1d1d1f"
    text_secondary = "#6e6e73"
    title_gradient = "linear-gradient(135deg, #1d1d1f 0%, #0066cc 100%)"
    glass_bg = "rgba(255, 255, 255, 0.7)"
    glass_border = "rgba(0, 0, 0, 0.1)"
    glass_hover_bg = "rgba(255, 255, 255, 0.9)"
    glass_hover_border = "rgba(0, 0, 0, 0.15)"
    user_msg_bg = "rgba(0, 122, 255, 0.12)"
    user_msg_border = "rgba(0, 122, 255, 0.25)"
    assistant_msg_bg = "rgba(255, 255, 255, 0.8)"
    assistant_msg_border = "rgba(0, 0, 0, 0.12)"
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

# Custom CSS for iOS-inspired glass design
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app background with animated gradient */
    .stApp {{
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
    }}

    /* iOS-style glass effect for message bubbles */
    [data-testid="stChatMessageContent"] {{
        background: {glass_bg};
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border: 1px solid {glass_border};
        border-radius: 20px;
        padding: 16px 20px;
        color: {text_primary};
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }}

    [data-testid="stChatMessageContent"]:hover {{
        background: {glass_hover_bg};
        border: 1px solid {glass_hover_border};
        transform: translateY(-1px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.25);
    }}

    /* User message - iOS blue */
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
        background: {user_msg_bg};
        border: 1px solid {user_msg_border};
    }}

    /* Assistant message */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
        background: {assistant_msg_bg};
        border: 1px solid {assistant_msg_border};
    }}

    /* Bottom bar container */
    [data-testid="stBottom"] {{
        background: {glass_bg} !important;
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border-top: 1px solid {glass_border};
    }}

    /* iOS-style glass chat input */
    .stChatInputContainer {{
        background: {glass_bg} !important;
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border: 1px solid {glass_border} !important;
        border-radius: 28px;
        padding: 4px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
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

    /* Avatar styling */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {{
        background: {glass_bg} !important;
        border: 1px solid {glass_border} !important;
        backdrop-filter: blur(10px);
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

    /* iOS-style Toggle Switch */
    .theme-switch {{
        width: 90px;
        height: 45px;
        cursor: pointer;
    }}

    .slider-track {{
        position: relative;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border-radius: 45px;
        box-shadow:
            inset 0 2px 8px rgba(0, 0, 0, 0.5),
            0 4px 20px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .theme-switch.active .slider-track {{
        background: linear-gradient(135deg, #ff9500 0%, #ff7a00 100%);
        border: 2px solid rgba(255, 200, 100, 0.3);
        box-shadow:
            inset 0 2px 8px rgba(0, 0, 0, 0.2),
            0 4px 20px rgba(255, 149, 0, 0.3);
    }}

    .slider-thumb {{
        position: absolute;
        top: 3px;
        left: 3px;
        width: 37px;
        height: 37px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        border-radius: 50%;
        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.3),
            0 1px 2px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }}

    .theme-switch.active .slider-thumb {{
        transform: translateX(45px);
    }}

    .slider-thumb::before {{
        content: "🌙";
    }}

    .theme-switch.active .slider-thumb::before {{
        content: "☀️";
    }}

    .slider-icons {{
        position: absolute;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 12px;
        pointer-events: none;
    }}

    .icon-left, .icon-right {{
        font-size: 16px;
        opacity: 0.4;
        transition: opacity 0.3s;
    }}

    .theme-switch .icon-left {{
        opacity: 0.6;
    }}

    .theme-switch.active .icon-left {{
        opacity: 0.3;
    }}

    .theme-switch .icon-right {{
        opacity: 0.3;
    }}

    .theme-switch.active .icon-right {{
        opacity: 0.7;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HEADER WITH THEME TOGGLE
# -----------------------------------------------------------------------------
# Display toggle switch visual with clickable overlay
toggle_active = "active" if st.session_state.dark_mode else ""

# Create invisible button overlay on toggle position
col_toggle, col_spacer = st.columns([1, 10])
with col_toggle:
    st.markdown(f"""
    <div style="position: fixed; top: 20px; right: 20px; z-index: 999;">
        <div class="theme-switch {toggle_active}">
            <div class="slider-track">
                <div class="slider-thumb"></div>
                <div class="slider-icons">
                    <span class="icon-left">🌙</span>
                    <span class="icon-right">☀️</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Invisible button overlay
    st.markdown("""
    <style>
    div[data-testid="column"]:first-child button {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        width: 90px !important;
        height: 45px !important;
        z-index: 1000 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 45px !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="column"]:first-child button:hover {
        background: transparent !important;
        border: none !important;
        transform: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("", key="theme_toggle", help="Toggle theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

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

    # Details
    st.markdown(f"**📅 Forecast Date:** {data['date']}")

    source_label = "📈 Model Forecast" if data["source"] in ["model", "model_search"] else "📜 Historical Value"
    st.markdown(f"**📘 Data Source:** {source_label}")

    if data.get("state"):
        st.markdown(f"**🏠 State:** {data['state']}")

    if data.get("store_id"):
        st.markdown(f"**🏪 Store ID:** {data['store_id']}")

    if data.get("dept_id"):
        st.markdown(f"**🏢 Dept ID:** {data['dept_id']}")

    if data.get("dept_name"):
        st.markdown(f"**🏢 Dept Name:** {data['dept_name']}")

    st.caption("Prediction generated using advanced time-series ensemble models.")

# -----------------------------------------------------------------------------
# CHAT DISPLAY
# -----------------------------------------------------------------------------
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
if prompt := st.chat_input("Ask anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
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
