import sqlite3
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "transactions.db")


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prices (
                 item_key TEXT PRIMARY KEY,
                 min_price REAL,
                 max_price REAL,
                 unit TEXT,
                 updated_at TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 item TEXT,
                 qty REAL,
                 price_per_unit REAL,
                 buy_price_per_unit REAL,
                 total REAL,
                 vendor_id TEXT,
                 payment_method TEXT,
                 created_at TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS vendors (
                 id TEXT PRIMARY KEY,
                 name TEXT,
                 created_at TEXT
                 )''')
    conn.commit()
    conn.close()


def insert_price(item_key, min_price, max_price, unit="birr/kg"):
    conn = get_conn()
    c = conn.cursor()
    c.execute("REPLACE INTO prices (item_key, min_price, max_price, unit, updated_at) VALUES (?, ?, ?, ?, ?)",
              (item_key, min_price, max_price, unit, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_price(item_key):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT min_price, max_price, unit FROM prices WHERE item_key = ?", (item_key,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"min": row[0], "max": row[1], "unit": row[2]}
    return None


def insert_transaction(item, qty, price_per_unit, buy_price_per_unit, total, vendor_id, payment_method):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO transactions (item, qty, price_per_unit, buy_price_per_unit, total, vendor_id, payment_method, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (item, qty, price_per_unit, buy_price_per_unit, total, vendor_id, payment_method, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    tid = c.lastrowid
    conn.close()
    return tid


def get_transactions_for_vendor(vendor_id, since=None):
    conn = get_conn()
    c = conn.cursor()
    if since:
        c.execute("SELECT item, qty, price_per_unit, buy_price_per_unit, total, created_at FROM transactions WHERE vendor_id = ? AND created_at >= ? ORDER BY created_at DESC", (vendor_id, since))
    else:
        c.execute("SELECT item, qty, price_per_unit, buy_price_per_unit, total, created_at FROM transactions WHERE vendor_id = ? ORDER BY created_at DESC", (vendor_id,))
    rows = c.fetchall()
    conn.close()
    return rows


init_db()
