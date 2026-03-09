"""
Sheffield Solar PV_Live API client — GB national solar generation.

Historical data is fetched from api.pvlive.uk (free, no authentication).
Provides half-hourly GB solar generation estimates in MW.

Forecast: PV_Live does not provide forecasts (the PV_Forecast API requires
a paid subscription).  Instead, solar_gw for the forecast horizon is estimated
from Open-Meteo shortwave_radiation using a linear model fitted on the last
90 days of historical PVLIVE data.  See analysis.estimate_solar_from_radiation().
"""
from datetime import date, timedelta

import requests

from app import db


BASE_URL = "https://api.pvlive.uk/pvlive/api/v4/gsp/0"
# Chunk size for historical fetches — avoids very large responses
_CHUNK_DAYS = 90


def fetch_historical(date_from: date, date_to: date) -> int:
    """
    Fetch 30-min GB national solar generation from PV_Live and upsert into DB.
    Data is chunked into 90-day windows to avoid oversized responses.
    Returns total rows stored.
    """
    total = 0
    chunk_start = date_from
    while chunk_start <= date_to:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), date_to)
        resp = requests.get(
            BASE_URL,
            params={
                "start": f"{chunk_start}T00:00:00",
                "end":   f"{chunk_end}T23:30:00",
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = _parse_response(payload)
        n = db.upsert_solar(rows)
        db.log_fetch("pvlive", chunk_start, chunk_end, n)
        total += n
        chunk_start = chunk_end + timedelta(days=1)
    return total


def _parse_response(payload: dict) -> list[dict]:
    """
    Parse PV_Live API v4 response.
    Accepts both array-of-arrays and array-of-dicts formats.
    """
    data = payload.get("data", [])
    if not data:
        return []

    # Determine field positions from meta if present
    meta = payload.get("meta", ["gsp_id", "datetime_gmt", "generation_mw"])
    if isinstance(meta, list) and len(meta) >= 3:
        dt_idx  = meta.index("datetime_gmt")  if "datetime_gmt"  in meta else 1
        mw_idx  = meta.index("generation_mw") if "generation_mw" in meta else 2
    else:
        dt_idx, mw_idx = 1, 2

    rows = []
    for record in data:
        if isinstance(record, (list, tuple)):
            dt_raw = record[dt_idx]
            mw     = record[mw_idx]
        else:
            dt_raw = record.get("datetime_gmt")
            mw     = record.get("generation_mw")
        if dt_raw is None:
            continue
        # Normalise to bare ISO-8601 without timezone suffix for consistent storage
        dt_str = str(dt_raw).replace("+00:00", "").replace("Z", "")
        rows.append({"datetime_gmt": dt_str, "generation_mw": mw if mw is not None else 0.0})
    return rows


def missing_solar_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored in the solar_generation table."""
    min_dt, max_dt = db.get_solar_date_range()

    if min_dt is None:
        return [(date_from, date_to)]

    stored_min = date.fromisoformat(min_dt[:10])
    stored_max = date.fromisoformat(max_dt[:10])

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        # PV_Live data has ~6 min latency; don't request beyond yesterday
        safe_to = min(date_to, date.today() - timedelta(days=1))
        if stored_max < safe_to:
            gaps.append((stored_max + timedelta(days=1), safe_to))
    return gaps
