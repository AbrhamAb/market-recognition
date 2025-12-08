"""Seed example price ranges into the SQLite DB.

Usage (from repo root):
    conda activate marketrec
    python -m backend.seed_prices

Edit the PRICES dict below to match your classes.
"""
from . import database

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


def main():
    for item, (mn, mx, unit) in PRICES.items():
        database.insert_price(item, mn, mx, unit)
        print(f"Seeded {item}: {mn}-{mx} {unit}")


if __name__ == "__main__":
    main()
