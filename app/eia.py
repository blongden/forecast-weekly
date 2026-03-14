"""
US petroleum inventory data from the EIA (Energy Information Administration).

Fetches weekly US commercial crude oil stocks (million barrels) from the EIA API v2.
Requires a free API key — register at https://www.eia.gov/opendata/register.php
Set the environment variable EIA_API_KEY to enable fetching.

Data stored:
  - us_crude_stocks_mb:  US commercial crude oil ending stocks (million barrels)
"""
import os
from datetime import date, timedelta

import pandas as pd
import requests

from app import db


EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

# EIA API v2 endpoint for weekly US petroleum stocks
EIA_BASE = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"


def fetch_oil_inventory(date_from: date, date_to: date) -> int:
    """
    Fetch weekly US crude oil stocks from EIA and upsert into the DB.
    Returns number of rows stored, or 0 if API key is not set.
    """
    if not EIA_API_KEY:
        print("  [EIA] Skipped — EIA_API_KEY not set "
              "(register free at https://www.eia.gov/opendata/register.php)")
        return 0

    params = {
        "api_key": EIA_API_KEY,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPC0",   # crude oil
        "facets[process][]": "SAE",    # ending stocks
        "start": str(date_from),
        "end": str(date_to),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    try:
        resp = requests.get(EIA_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [EIA] Warning: API request failed: {e}")
        return 0

    records = data.get("response", {}).get("data", [])
    if not records:
        return 0

    rows = []
    for rec in records:
        period = rec.get("period")
        value = rec.get("value")
        if period and value is not None:
            try:
                rows.append({
                    "date": str(period),
                    "us_crude_stocks_mb": float(value),
                })
            except (ValueError, TypeError):
                continue

    if not rows:
        return 0

    n = db.upsert_oil_inventory(rows)
    db.log_fetch("oil_inventory", date_from, date_to, n)
    return n


def missing_oil_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet in the DB."""
    min_dt, max_dt = db.get_oil_inventory_date_range()

    if min_dt is None:
        return [(date_from, date_to)]

    stored_min = date.fromisoformat(min_dt)
    stored_max = date.fromisoformat(max_dt)

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        gaps.append((stored_max + timedelta(days=1), date_to))
    return gaps
