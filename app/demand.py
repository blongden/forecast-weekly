"""
Elexon BMRS — GB Initial National Demand Outturn (INDO).

Half-hourly GB total demand in MW (includes distributed/embedded generation).
Uses the /demand/outturn endpoint with settlementDate filtering.
The 'startTime' field is UTC period-start, matching Octopus period-start convention,
so datetime alignment with the prices table is direct.
"""
from datetime import date, timedelta

import requests

from app import db


BASE_URL    = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
_CHUNK_DAYS = 14


def fetch_demand(date_from: date, date_to: date) -> int:
    """Fetch half-hourly GB demand from Elexon BMRS and upsert into DB."""
    total = 0
    chunk_start = date_from
    while chunk_start <= date_to:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), date_to)
        resp = requests.get(
            BASE_URL,
            params={
                "settlementDateFrom": str(chunk_start),
                "settlementDateTo":   str(chunk_end),
            },
            timeout=60,
        )
        resp.raise_for_status()
        rows = _parse_response(resp.json())
        n = db.upsert_demand(rows)
        db.log_fetch("demand_indo", chunk_start, chunk_end, n)
        total += n
        chunk_start = chunk_end + timedelta(days=1)
    return total


def _parse_response(payload: dict) -> list[dict]:
    """
    Parse /demand/outturn response.
    'startTime' is the UTC period-start, matching Octopus period-start convention.
    'initialDemandOutturn' is GB total demand including embedded generation (INDO).
    """
    rows = []
    for rec in payload.get("data", []):
        start_time = rec.get("startTime")
        demand_mw  = rec.get("initialDemandOutturn")
        if start_time is None or demand_mw is None:
            continue
        # Normalise to bare ISO-8601 without timezone suffix
        dt_str = start_time.replace("+00:00", "").replace("Z", "")
        # Keep only YYYY-MM-DDTHH:MM:SS portion
        if "T" in dt_str:
            dt_str = dt_str[:19]
        rows.append({"datetime_utc": dt_str, "demand_mw": float(demand_mw)})
    return rows


def missing_demand_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored in demand_halfhourly."""
    min_dt, max_dt = db.get_demand_date_range()

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
