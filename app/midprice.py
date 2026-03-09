"""
Elexon BMRS — EPEX SPOT GB Day-Ahead Market Index Prices (APXMIDP).

Half-hourly day-ahead auction clearing prices in £/MWh.
These are the wholesale prices that Octopus uses to calculate the Agile tariff.

Endpoint: /balancing/pricing/market-index
  - dataProviders=APXMIDP  (EPEX SPOT, formerly APX — carries real data)
  - from / to date parameters
  - startTime = UTC period-start

Used in the model as a 1-day lag feature (yesterday's day-ahead price is
always known when forecasting today and tomorrow).
"""
from datetime import date, timedelta

import requests

from app import db


BASE_URL    = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
_CHUNK_DAYS = 7  # API maximum


def fetch_midprice(date_from: date, date_to: date) -> int:
    """Fetch half-hourly EPEX day-ahead prices from Elexon BMRS and upsert into DB."""
    total = 0
    chunk_start = date_from
    while chunk_start <= date_to:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), date_to)
        resp = requests.get(
            BASE_URL,
            params={
                "from":           str(chunk_start),
                "to":             str(chunk_end),
                "dataProviders":  "APXMIDP",
            },
            timeout=60,
        )
        resp.raise_for_status()
        rows = _parse_response(resp.json())
        n = db.upsert_midprice(rows)
        db.log_fetch("midprice_apx", chunk_start, chunk_end, n)
        total += n
        chunk_start = chunk_end + timedelta(days=1)
    return total


def _parse_response(payload: dict) -> list[dict]:
    """
    Parse market-index response.  Returns per-slot rows with UTC period-start key.
    price is in £/MWh.
    """
    rows = []
    for rec in payload.get("data", []):
        if rec.get("dataProvider") != "APXMIDP":
            continue
        start_time = rec.get("startTime")
        price      = rec.get("price")
        volume     = rec.get("volume")
        if start_time is None or price is None:
            continue
        dt_str = start_time.replace("+00:00", "").replace("Z", "")[:19]
        rows.append({
            "datetime_utc":    dt_str,
            "price_gbp_mwh":   float(price),
            "volume_mwh":      float(volume) if volume is not None else None,
        })
    return rows


def missing_midprice_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored in market_index_halfhourly."""
    min_dt, max_dt = db.get_midprice_date_range()

    if min_dt is None:
        return [(date_from, date_to)]

    stored_min = date.fromisoformat(min_dt[:10])
    stored_max = date.fromisoformat(max_dt[:10])

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        safe_to = min(date_to, date.today() - timedelta(days=1))
        if stored_max < safe_to:
            gaps.append((stored_max + timedelta(days=1), safe_to))
    return gaps
