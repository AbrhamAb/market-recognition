from .database import get_price, insert_price


def fetch_price_for_item(item_key):
    """Return average price and unit or None"""
    p = get_price(item_key)
    if not p:
        return None
    avg = (p["min"] + p["max"]) / 2.0
    return {"price_per_unit": avg, "unit": p["unit"], "min": p["min"], "max": p["max"]}


def update_price(item_key, min_price, max_price, unit="birr/kg"):
    insert_price(item_key, min_price, max_price, unit)
