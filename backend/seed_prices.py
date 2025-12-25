"""Seed prices for all model labels.

This script reads the latest `labels.txt` under `model/` and seeds a price
range for each label. The special label `unknown` is skipped so unknown
predictions return no price. If a default mapping exists for a label it is
used (case-insensitive); otherwise a sensible fallback range is inserted.

Usage:
    conda activate marketrec
    python -m backend.seed_prices
"""
import os
import glob
from . import database

# Per-label defaults (canonical forms). Keys are informal and matched
# case-insensitively against model labels.
PRICES = {
    "bean": (30, 45, "birr/kg"),
    "bitter_gourd": (35, 50, "birr/kg"),
    "bottle_gourd": (25, 40, "birr/kg"),
    "brinjal": (28, 42, "birr/kg"),
    "broccoli": (55, 75, "birr/kg"),
    "cabbage": (18, 28, "birr/kg"),
    "capsicum": (45, 65, "birr/kg"),
    "carrot": (26, 38, "birr/kg"),
    "cauliflower": (35, 55, "birr/kg"),
    "cucumber": (20, 32, "birr/kg"),
    "papaya": (22, 34, "birr/kg"),
    "potato": (24, 34, "birr/kg"),
    "pumpkin": (20, 32, "birr/kg"),
    "radish": (18, 28, "birr/kg"),
    "tomato": (22, 30, "birr/kg"),
}


def find_latest_labels(model_root=None):
    root = model_root or os.path.join(os.path.dirname(__file__), "..", "model")
    # look for model/*/labels.txt
    candidates = glob.glob(os.path.join(root, "*", "labels.txt"))
    if not candidates:
        fallback = os.path.join(root, "labels.txt")
        if os.path.exists(fallback):
            with open(fallback, encoding="utf-8") as f:
                return [l.strip() for l in f.readlines() if l.strip()]
        return []
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def normalize_key(s: str) -> str:
    # simple normalization for matching (lowercase, replace spaces with underscore)
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch == ' ').strip().replace(' ', '_')


def main():
    labels = find_latest_labels()
    if not labels:
        print("No labels.txt found under model/. Populate a model first or seed manually.")
        return

    # build normalized mapping for PRICES
    mapping = {normalize_key(k): v for k, v in PRICES.items()}
    fallback = (20.0, 40.0, "birr/kg")

    for label in labels:
        if label is None:
            continue
        if label.lower() == "unknown":
            print(
                f"Skipping '{label}' (unknown class) — no price will be seeded")
            continue
        key = label
        nk = normalize_key(label)
        if nk in mapping:
            mn, mx, unit = mapping[nk]
        else:
            mn, mx, unit = fallback
        database.insert_price(key, mn, mx, unit)
        print(f"Seeded {key}: {mn}-{mx} {unit}")


if __name__ == "__main__":
    main()
