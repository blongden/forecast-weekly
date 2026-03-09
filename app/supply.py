"""
Elexon BMRS FUELHH — half-hourly GB generation by fuel type.

Aggregated per slot into:
  wind_mw    — GB wind generation (onshore + offshore)
  gas_mw     — gas (CCGT + OCGT)
  nuclear_mw — nuclear (relatively constant; ~6–8 GW)
  imports_mw — net interconnector flows (positive = UK importing)

Max 7 days per FUELHH request.
startTime is UTC period-start, matching Octopus and demand_halfhourly conventions.
"""
from datetime import date, datetime, timedelta
from collections import defaultdict

import requests

from app import db


BASE_URL    = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"
_CHUNK_DAYS = 7

# Interconnector fuel type prefixes — positive = UK importing
_INT_FUELS = frozenset({
    "INTELEC", "INTEW", "INTFR", "INTGRNL", "INTIFA2",
    "INTIRL", "INTNED", "INTNEM", "INTNSL", "INTVKL",
})


def fetch_supply(date_from: date, date_to: date) -> int:
    """Fetch half-hourly GB generation mix from Elexon BMRS and upsert into DB."""
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
        n = db.upsert_generation(rows)
        db.log_fetch("supply_fuelhh", chunk_start, chunk_end, n)
        total += n
        chunk_start = chunk_end + timedelta(days=1)
    return total


def _parse_response(payload: dict) -> list[dict]:
    """
    Parse FUELHH response.  Aggregates all fuel types into per-slot summary rows.
    startTime is UTC period-start, stored as-is (matches prices table convention).
    """
    # Group records by UTC period-start datetime
    slots: dict[str, dict] = defaultdict(lambda: {
        "wind_mw": 0.0, "gas_mw": 0.0, "nuclear_mw": 0.0, "imports_mw": 0.0
    })

    for rec in payload.get("data", []):
        start_time = rec.get("startTime")
        fuel       = rec.get("fuelType", "")
        generation = rec.get("generation")
        if start_time is None or generation is None:
            continue

        # Normalise to bare ISO-8601 without timezone suffix
        dt_str = start_time.replace("+00:00", "").replace("Z", "")[:19]

        mw = float(generation)
        if fuel == "WIND":
            slots[dt_str]["wind_mw"] += mw
        elif fuel in ("CCGT", "OCGT"):
            slots[dt_str]["gas_mw"] += mw
        elif fuel == "NUCLEAR":
            slots[dt_str]["nuclear_mw"] += mw
        elif fuel in _INT_FUELS:
            slots[dt_str]["imports_mw"] += mw

    return [{"datetime_utc": dt, **vals} for dt, vals in slots.items()]


def missing_supply_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored in generation_halfhourly."""
    min_dt, max_dt = db.get_generation_date_range()

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
