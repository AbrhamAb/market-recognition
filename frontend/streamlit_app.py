import streamlit as st
import requests

BACKEND_URL = st.secrets.get("backend_url", "http://localhost:8000")

st.title("Market Product Recognition — Demo")

st.write("Upload an image of a market product (spice, vegetable, grain) and get a suggested price.")

uploaded = st.file_uploader(
    "Choose a product photo", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)
with col1:
    vendor_id = st.text_input("Vendor ID", value="V0001")
with col2:
    qty = st.number_input("Quantity (kg)", min_value=0.01, value=1.0, step=0.1)

buy_price = st.number_input(
    "Buy price per unit (optional)", min_value=0.0, value=0.0, step=0.1)

if st.button("Analyze") and uploaded is not None:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    data = {"vendor_id": vendor_id, "qty": qty,
            "buy_price_per_unit": (buy_price if buy_price > 0 else None)}
    with st.spinner("Contacting backend..."):
        resp = requests.post(BACKEND_URL + "/predict", files=files, data={
                             "vendor_id": vendor_id, "qty": qty, "buy_price_per_unit": buy_price if buy_price > 0 else ""})
    if resp.status_code == 200:
        j = resp.json()
        st.metric("Item", j.get("item"))
        st.metric("Confidence", f"{j.get('confidence'):.2f}")
        st.write(f"Price per unit: {j.get('price_per_unit')} {j.get('unit')}")
        st.write(f"Quantity: {j.get('qty')}")
        st.write(f"Total: {j.get('total')}")
        st.write("Payment options:")
        for p in j.get("payment_options", []):
            st.button(f"Pay with {p}")
    else:
        st.error(f"Server error: {resp.status_code} - {resp.text}")
