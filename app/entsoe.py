"""
ENTSO-E Transparency Platform data client.

Fetches two data streams for the GB electricity price model:
  1. Day-ahead scheduled interconnector exchanges (hourly)
  2. Generation unit unavailability / outages (daily summary)

Requires a free API key from https://transparency.entsoe.eu/
Set ENTSOE_API_KEY in .env or environment.
"""
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app import db

load_dotenv()

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")

# GB bidding zone EIC code
GB_AREA = "GB"

# Interconnector neighbours — keys match entsoe-py country code convention
# GB has ~10 GW of interconnector capacity across 6 links:
#   IFA + IFA2 + ElecLink (FR ~4 GW), Nemo (BE 1 GW), BritNed (NL 1 GW),
#   NSL (NO 1.4 GW), Moyle + EWIC (IE ~1 GW), Viking Link (DK 1.4 GW)
INTERCONNECTOR_NEIGHBOURS = ["FR", "BE", "NL", "NO_2", "IE_SEM", "DK_1"]

# Map ENTSO-E production type codes to simplified fuel categories
_FUEL_MAP = {
    "Biomass":                            "other",
    "Fossil Brown coal/Lignite":          "coal",
    "Fossil Coal-derived gas":            "gas",
    "Fossil Gas":                         "gas",
    "Fossil Hard coal":                   "coal",
    "Fossil Oil":                         "other",
    "Fossil Oil shale":                   "other",
    "Fossil Peat":                        "other",
    "Geothermal":                         "other",
    "Hydro Pumped Storage":               "other",
    "Hydro Run-of-river and poundage":    "other",
    "Hydro Water Reservoir":              "other",
    "Marine":                             "other",
    "Nuclear":                            "nuclear",
    "Other":                              "other",
    "Other renewable":                    "other",
    "Solar":                              "other",
    "Waste":                              "other",
    "Wind Offshore":                      "wind",
    "Wind Onshore":                       "wind",
}


def _get_client():
    """Return an EntsoePandasClient, or None if no API key is configured."""
    if not ENTSOE_API_KEY:
        return None
    try:
        from entsoe import EntsoePandasClient
        return EntsoePandasClient(api_key=ENTSOE_API_KEY)
    except ImportError:
        print("  [ENTSO-E] entsoe-py not installed — skipping.")
        return None


def fetch_scheduled_exchanges(date_from: date, date_to: date) -> int:
    """
    Fetch day-ahead scheduled exchanges between GB and each interconnector neighbour.
    Stores hourly rows in entsoe_scheduled_exchanges.
    Returns total number of rows upserted.
    """
    client = _get_client()
    if client is None:
        return 0

    start = pd.Timestamp(str(date_from), tz="UTC")
    end = pd.Timestamp(str(date_to + timedelta(days=1)), tz="UTC")

    all_rows = []
    for neighbour in INTERCONNECTOR_NEIGHBOURS:
        # Imports: neighbour -> GB
        try:
            imports = client.query_scheduled_exchanges(
                country_code_from=neighbour,
                country_code_to=GB_AREA,
                start=start,
                end=end,
            )
            if imports is not None and not imports.empty:
                for dt, mw in imports.items():
                    if pd.notna(mw):
                        all_rows.append({
                            "datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                            "country_from": neighbour,
                            "country_to": "GB",
                            "scheduled_mw": float(mw),
                        })
        except Exception as e:
            print(f"  [ENTSO-E] Warning: scheduled imports {neighbour}->GB failed: {e}")

        # Exports: GB -> neighbour
        try:
            exports = client.query_scheduled_exchanges(
                country_code_from=GB_AREA,
                country_code_to=neighbour,
                start=start,
                end=end,
            )
            if exports is not None and not exports.empty:
                for dt, mw in exports.items():
                    if pd.notna(mw):
                        all_rows.append({
                            "datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                            "country_from": "GB",
                            "country_to": neighbour,
                            "scheduled_mw": float(mw),
                        })
        except Exception as e:
            print(f"  [ENTSO-E] Warning: scheduled exports GB->{neighbour} failed: {e}")

    n = db.upsert_entsoe_exchanges(all_rows)
    db.log_fetch("entsoe_exchanges", date_from, date_to, n)
    return n


def fetch_unavailability(date_from: date, date_to: date) -> int:
    """
    Fetch generation unit unavailability (planned + forced outages) for GB.
    Aggregates to daily totals by fuel type.
    Returns number of rows upserted.
    """
    client = _get_client()
    if client is None:
        return 0

    start = pd.Timestamp(str(date_from), tz="UTC")
    end = pd.Timestamp(str(date_to + timedelta(days=1)), tz="UTC")

    try:
        unavail = client.query_unavailability_of_generation_units(
            country_code=GB_AREA,
            start=start,
            end=end,
        )
    except Exception as e:
        print(f"  [ENTSO-E] Warning: unavailability query failed: {e}")
        return 0

    if unavail is None or unavail.empty:
        return 0

    # unavail is a DataFrame with columns including:
    #   'start', 'end', 'Nominal_MW' (or similar), 'Production_Type'
    # Column names vary by entsoe-py version; normalise
    df = unavail.reset_index()

    # Find the capacity/power column
    mw_col = None
    for candidate in ["Nominal_MW", "NominalPower", "nominal_mw", "Available Capacity"]:
        if candidate in df.columns:
            mw_col = candidate
            break
    if mw_col is None:
        # Try to find any column with 'mw' or 'power' or 'capacity' in name
        for c in df.columns:
            if any(k in c.lower() for k in ("mw", "power", "capacity", "nominal")):
                mw_col = c
                break
    if mw_col is None:
        print(f"  [ENTSO-E] Warning: could not find MW column in unavailability data. Columns: {list(df.columns)}")
        return 0

    # Find production type column
    ptype_col = None
    for candidate in ["Production_Type", "production_type", "ProductionType"]:
        if candidate in df.columns:
            ptype_col = candidate
            break
    if ptype_col is None:
        for c in df.columns:
            if "production" in c.lower() or "type" in c.lower():
                ptype_col = c
                break

    # Find start/end datetime columns
    start_col = end_col = None
    for candidate in ["start", "Start_DateTime", "start_date"]:
        if candidate in df.columns:
            start_col = candidate
            break
    for candidate in ["end", "End_DateTime", "end_date"]:
        if candidate in df.columns:
            end_col = candidate
            break

    if start_col is None or end_col is None:
        # If no explicit start/end, the index might be a DatetimeIndex
        print(f"  [ENTSO-E] Warning: could not find start/end columns. Columns: {list(df.columns)}")
        return 0

    # Expand each outage event into per-date rows
    daily_unavail: dict[tuple[str, str], float] = {}  # (date_str, fuel_type) -> MW
    for _, row in df.iterrows():
        try:
            mw = float(row[mw_col])
        except (ValueError, TypeError):
            continue
        if np.isnan(mw) or mw <= 0:
            continue

        fuel = "other"
        if ptype_col and pd.notna(row.get(ptype_col)):
            fuel = _FUEL_MAP.get(str(row[ptype_col]), "other")

        try:
            ev_start = pd.Timestamp(row[start_col])
            ev_end = pd.Timestamp(row[end_col])
        except Exception:
            continue

        # Clip to our query range
        ev_start = max(ev_start, pd.Timestamp(str(date_from), tz=ev_start.tz))
        ev_end = min(ev_end, pd.Timestamp(str(date_to + timedelta(days=1)), tz=ev_end.tz))

        # Expand to each date the outage overlaps
        current = ev_start.normalize()
        while current < ev_end:
            d_str = current.strftime("%Y-%m-%d")
            key = (d_str, fuel)
            # Take max per unit per day (events may overlap for same unit)
            daily_unavail[key] = daily_unavail.get(key, 0) + mw
            current += pd.Timedelta(days=1)

    rows = [
        {"date": d, "fuel_type": ft, "unavailable_mw": round(mw, 1)}
        for (d, ft), mw in sorted(daily_unavail.items())
    ]

    n = db.upsert_entsoe_unavailability(rows)
    db.log_fetch("entsoe_unavailability", date_from, date_to, n)
    return n


def missing_exchanges_range(date_from: date, date_to: date) -> tuple[date, date] | None:
    """Return the gap (start, end) of missing scheduled exchange data, or None if up to date."""
    existing_min, existing_max = db.get_entsoe_exchanges_date_range()
    if existing_max is None:
        return (date_from, date_to)
    last = date.fromisoformat(existing_max)
    if last >= date_to - timedelta(days=1):
        return None
    return (last + timedelta(days=1), date_to)


def missing_unavailability_range(date_from: date, date_to: date) -> tuple[date, date] | None:
    """Return the gap (start, end) of missing unavailability data, or None if up to date."""
    existing_min, existing_max = db.get_entsoe_unavailability_date_range()
    if existing_max is None:
        return (date_from, date_to)
    last = date.fromisoformat(existing_max)
    if last >= date_to - timedelta(days=1):
        return None
    return (last + timedelta(days=1), date_to)
