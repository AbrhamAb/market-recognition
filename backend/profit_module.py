from .database import get_transactions_for_vendor


def compute_vendor_summary(vendor_id, since=None):
    rows = get_transactions_for_vendor(vendor_id, since)
    total_sales = 0.0
    total_cost = 0.0
    total_profit = 0.0
    count = 0
    for item, qty, price_per_unit, buy_price_per_unit, total, created_at in rows:
        if total is None:
            continue
        total_sales += total
        if buy_price_per_unit is not None:
            total_cost += (buy_price_per_unit * qty)
            total_profit += (price_per_unit - buy_price_per_unit) * qty
        count += 1
    return {
        "vendor_id": vendor_id,
        "transactions_count": count,
        "total_sales": total_sales,
        "total_cost": total_cost,
        "total_profit": total_profit
    }
