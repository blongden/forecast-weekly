"""
Elexon BMRS system price (cash-out / imbalance) client.
Fetches half-hourly system buy/sell prices — captures real-time supply stress.
Free API, no key required.
"""
from datetime import date, timedelta
import requests
import pandas as pd
from app import db

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"

def fetch_system_prices(date_from: date, date_to: date) -> int:
    """Fetch half-hourly system prices and store in DB. Returns rows upserted."""
    all_rows = []
    current = date_from
    while current <= date_to:
        try:
            resp = requests.get(f"{BASE_URL}/{current}", timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            for item in data:
                all_rows.append({
                    "datetime_utc": item["startTime"].replace("Z", ""),
                    "system_buy_price": item.get("systemBuyPrice"),
                    "system_sell_price": item.get("systemSellPrice"),
                    "net_imbalance_mw": item.get("netImbalanceVolume"),
                })
        except Exception as e:
            print(f"  [SysPrice] Warning: {current} failed: {e}")
        current += timedelta(days=1)

    if not all_rows:
        return 0
    n = db.upsert_system_prices(all_rows)
    db.log_fetch("system_prices", date_from, date_to, n)
    return n

def missing_sysprice_range(date_from: date, date_to: date):
    """Return (start, end) of missing data, or None if up to date."""
    existing_min, existing_max = db.get_sysprice_date_range()
    if existing_max is None:
        return (date_from, date_to)
    last = date.fromisoformat(existing_max[:10])
    if last >= date_to - timedelta(days=1):
        return None
    return (last + timedelta(days=1), date_to)
