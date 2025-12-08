"""Seed example price ranges into the SQLite DB.

Usage (from repo root):
    conda activate marketrec
    python -m backend.seed_prices

Edit the PRICES dict below to match your classes.
"""
from . import database

PRICES = {
    "banana": (15, 20, "birr/kg"),
    "garlic": (50, 70, "birr/kg"),
    "red_onion": (28, 35, "birr/kg"),
    "berbere": (250, 300, "birr/kg"),
    "coffee_beans": (90, 120, "birr/kg"),
    "injera": (18, 25, "birr/pc"),
    "lentils": (60, 80, "birr/kg"),
    "tomato": (20, 30, "birr/kg"),
    "shiro": (120, 160, "birr/kg"),
    "teff": (70, 95, "birr/kg"),
}


def main():
    for item, (mn, mx, unit) in PRICES.items():
        database.insert_price(item, mn, mx, unit)
        print(f"Seeded {item}: {mn}-{mx} {unit}")


if __name__ == "__main__":
    main()
