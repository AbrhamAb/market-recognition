from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
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
CONF_THRESHOLD = 0.5

_model = None
_labels = []


def dummy_predict(image: Image.Image, top_k: int = 3):
    # Simple stub used when a trained model is unavailable.
    w, h = image.size
    best = ("banana", 0.87) if w > h else ("garlic", 0.92)
    fallback = [
        {"label": best[0], "confidence": best[1]},
        {"label": "tomato", "confidence": 0.22},
        {"label": "potato", "confidence": 0.18},
    ]
    return {"label": best[0], "confidence": best[1], "top_k": fallback[:top_k]}


def load_model_and_labels():
    """Load model (keras .keras/.h5 or SavedModel via TFSMLayer) and labels."""
    global _model, _labels
    if _model is not None and _labels:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        _model = load_model(MODEL_PATH)  # handles .keras / .h5
    except Exception:
        # Fall back to SavedModel directory using TFSMLayer (Keras 3 path)
        _model = tf.keras.layers.TFSMLayer(
            MODEL_PATH, call_endpoint="serving_default")
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = [line.strip() for line in f.readlines() if line.strip()]
    return _model is not None


def predict_image(image: Image.Image, top_k: int = 3):
    """Run model inference; fall back to dummy if model unavailable."""
    global _model, _labels
    if _model is None:
        loaded = load_model_and_labels()
        if not loaded:
            return dummy_predict(image, top_k=top_k)
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    if isinstance(_model, tf.keras.layers.TFSMLayer):
        outputs = _model(arr)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())[0]
        preds = outputs.numpy()[0]
    else:
        preds = _model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]
    ranked = []
    for idx in order[:top_k]:
        label = _labels[idx] if idx < len(_labels) else str(idx)
        ranked.append({"label": label, "confidence": float(preds[idx])})
    best = ranked[0] if ranked else {"label": "unknown", "confidence": 0.0}
    return {"label": best["label"], "confidence": best["confidence"], "top_k": ranked}


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
        "low_confidence": conf < CONF_THRESHOLD,
        "top_k": res.get("top_k", []),
        "qty": qty,
        "price_per_unit": price,
        "unit": unit,
        "total": total,
        "price_available": price_info is not None,
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
