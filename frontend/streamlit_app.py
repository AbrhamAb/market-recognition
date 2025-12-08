import os
import streamlit as st
import requests


st.set_page_config(page_title="Market Recognition",
                   page_icon="MR", layout="wide")


def resolve_backend_url():
    env_val = os.getenv("BACKEND_URL")
    if env_val:
        return env_val
    try:
        return st.secrets["backend_url"]
    except Exception:
        return "http://localhost:8000"


def resolve_logo_path():
    env_val = os.getenv("APP_LOGO")
    if env_val and os.path.exists(env_val):
        return env_val
    default_path = os.path.join(
        os.path.dirname(__file__), "assets", "logo.png")
    return default_path if os.path.exists(default_path) else None


BACKEND_URL = resolve_backend_url()
LOGO_PATH = resolve_logo_path()

# Light/Dark responsive styling with glassy cards
st.markdown(
    """
    <style>
    :root {
        /* Light mode */
        --bg: radial-gradient(circle at 10% 20%, #e0f2fe 0%, #f8fafc 40%, #eef2ff 100%);
        --card: rgba(255,255,255,0.85);
        --card-border: rgba(15,23,42,0.06);
        --shadow: 0 18px 45px rgba(15,23,42,0.12);
        --accent: #0ea5e9;
        --text: #0f172a;
        --muted: #475569;
        --pill-bg: rgba(14,165,233,0.12);
        --warn-bg: rgba(247,201,72,0.18);
        --warn-border: #d4a017;
        --warn-text: #92400e;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: radial-gradient(circle at 15% 20%, #0b2535 0%, #0e2f46 40%, #0a1b2b 100%);
            --card: rgba(255,255,255,0.08);
            --card-border: rgba(255,255,255,0.08);
            --shadow: 0 18px 45px rgba(0,0,0,0.35);
            --accent: #39d2c0;
            --text: #e8f1f5;
            --muted: #9fb3c8;
            --pill-bg: rgba(57,210,192,0.15);
            --warn-bg: rgba(247,201,72,0.12);
            --warn-border: #f7c948;
            --warn-text: #fbe6a2;
        }
    }
    * { font-family: "Inter", "Segoe UI", sans-serif; }
    html, body { background: var(--bg); color: var(--text); }
    .main { background: var(--bg); color: var(--text); }
    .block-container { background: transparent; }
    section[data-testid="stSidebar"] { background: var(--card); color: var(--text); box-shadow: var(--shadow); border-right: 1px solid var(--card-border); }
    .stMarkdown, .stText, .stCaption, .stRadio, .stSelectbox { color: var(--text); }
    h1, h2, h3, h4, h5, h6, .stCaption, .stText { color: var(--text); }
    .hero-title { color: var(--text); }
    .header-caption { color: var(--muted); } 
    .metric-card { 
        background: var(--card);
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 14px;
        background: var(--pill-bg);
        color: var(--text);
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .warning-box {
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid var(--warn-border);
        background: var(--warn-bg);
        color: var(--warn-text);
        margin-bottom: 8px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid transparent;
        background: linear-gradient(120deg, var(--accent), #5eead4);
        color: #0b1224;
        font-weight: 600;
        padding: 10px 14px;
        box-shadow: var(--shadow);
        transition: transform 0.08s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(14,165,233,0.25);
    }
    .stButton > button:focus { outline: 2px solid var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=96)

with header_col2:
    st.markdown("<h2 class='hero-title' style='margin-bottom:4px;'>Market Recognition</h2>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p class='header-caption'>Backend: {BACKEND_URL}</p>", unsafe_allow_html=True)
    st.markdown("<p class='header-caption'>Upload a product photo, get a label, confidence, and pricing hints.</p>", unsafe_allow_html=True)

controls_col, preview_col = st.columns([1.2, 1])

with controls_col:
    st.subheader("Input")
    uploaded = st.file_uploader(
        "Choose a product photo", type=["jpg", "jpeg", "png"])
    vendor_id = st.text_input("Vendor ID", value="V0001")
    qty = st.number_input("Quantity (kg)", min_value=0.01, value=1.0, step=0.1)
    buy_price = st.number_input(
        "Buy price per unit (optional)", min_value=0.0, value=0.0, step=0.1)

    analyze = st.button("Analyze", use_container_width=True)

with preview_col:
    st.subheader("Preview")
    if uploaded is not None:
        st.image(uploaded.getvalue(), caption=uploaded.name,
                 width=600)
    else:
        st.info("Upload an image to see a preview here.")

result_placeholder = st.container()

if analyze and uploaded is not None:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    data = {"vendor_id": vendor_id, "qty": qty,
            "buy_price_per_unit": (buy_price if buy_price > 0 else None)}
    with st.spinner("Contacting backend..."):
        resp = requests.post(BACKEND_URL + "/predict", files=files, data={
                             "vendor_id": vendor_id, "qty": qty, "buy_price_per_unit": buy_price if buy_price > 0 else ""})
    if resp.status_code == 200:
        j = resp.json()
        with result_placeholder:
            metrics_col, detail_col = st.columns([1.2, 1])
            with metrics_col:
                st.markdown("<div class='metric-card'>",
                            unsafe_allow_html=True)
                st.metric("Item", j.get("item"))
                st.metric("Confidence", f"{j.get('confidence'):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

                price_per_unit = j.get("price_per_unit")
                unit = j.get("unit")
                if j.get("price_available") and price_per_unit is not None:
                    st.metric("Price per unit", f"{price_per_unit} {unit}")
                else:
                    st.markdown(
                        "<div class='warning-box'>No price set for this item.</div>", unsafe_allow_html=True)
                st.metric("Quantity", j.get("qty"))
                st.metric("Total", j.get("total"))

            with detail_col:
                if j.get("low_confidence"):
                    st.markdown(
                        "<div class='warning-box'>Low confidence prediction — double check or retake the photo.</div>", unsafe_allow_html=True)
                top_k = j.get("top_k", [])
                if top_k:
                    st.write("Top suggestions:")
                    for entry in top_k:
                        st.markdown(
                            f"<span class='pill'>{entry.get('label')} ({entry.get('confidence'):.2f})</span>", unsafe_allow_html=True)
                st.write("Payment options:")
                for p in j.get("payment_options", []):
                    st.button(f"Pay with {p}", use_container_width=True)
    else:
        st.error(f"Server error: {resp.status_code} - {resp.text}")
elif analyze and uploaded is None:
    st.warning("Please upload an image first.")
