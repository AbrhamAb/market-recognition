from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import datetime
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from . import database
from .price_engine import fetch_price_for_item, update_price
from .profit_module import compute_vendor_summary

app = FastAPI(title="Market Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "model"))
MODEL_PATH = os.path.join(MODEL_DIR, "product_classifier")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
IMG_SIZE = (224, 224)

_model = None
_labels = []


def dummy_predict(image: Image.Image):
    # Very simple stub — replace with real model inference later.
    # For now, return a fixed label based on image size or other simple heuristic.
    w, h = image.size
    if w > h:
        return {"label": "banana", "confidence": 0.87}
    return {"label": "garlic", "confidence": 0.92}


def load_model_and_labels():
    """Load SavedModel and labels if available. Returns True if loaded."""
    global _model, _labels
    if _model is not None and _labels:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    _model = load_model(MODEL_PATH)
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = [line.strip() for line in f.readlines() if line.strip()]
    return _model is not None


def predict_image(image: Image.Image):
    """Run model inference; fall back to dummy if model unavailable."""
    global _model, _labels
    if _model is None:
        loaded = load_model_and_labels()
        if not loaded:
            return dummy_predict(image)
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    preds = _model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = _labels[idx] if idx < len(_labels) else str(idx)
    conf = float(preds[idx])
    return {"label": label, "confidence": conf}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}


@app.on_event("startup")
async def startup_event():
    load_model_and_labels()


@app.post("/predict")
async def predict(file: UploadFile = File(...), vendor_id: str = Form("unknown"), qty: float = Form(1.0), buy_price_per_unit: float = Form(None)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    res = predict_image(image)
    label = res["label"]
    conf = res["confidence"]
    price_info = fetch_price_for_item(label)
    if price_info:
        price = price_info["price_per_unit"]
        unit = price_info["unit"]
    else:
        price = None
        unit = None
    total = (price * qty) if price is not None else None
    txn_id = database.insert_transaction(
        label, qty, price, buy_price_per_unit, total, vendor_id, None)
    return JSONResponse({
        "transaction_id": txn_id,
        "item": label,
        "confidence": conf,
        "qty": qty,
        "price_per_unit": price,
        "unit": unit,
        "total": total,
        "payment_options": ["Telebirr", "CBE Birr", "Amole", "M-Pesa Ethiopia"]
    })


@app.get("/prices/{item_key}")
async def get_price(item_key: str):
    p = fetch_price_for_item(item_key)
    if not p:
        return JSONResponse({"error": "not found"}, status_code=404)
    return p


@app.post("/prices/{item_key}")
async def set_price(item_key: str, min_price: float = Form(...), max_price: float = Form(...), unit: str = Form("birr/kg")):
    update_price(item_key, min_price, max_price, unit)
    return {"status": "ok", "item": item_key}


@app.get("/vendor/{vendor_id}/summary")
async def vendor_summary(vendor_id: str, since: str = None):
    summary = compute_vendor_summary(vendor_id, since)
    return summary
