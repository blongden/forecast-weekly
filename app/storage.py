"""
Gas storage data from GIE AGSI+ (Gas Infrastructure Europe).

Fetches daily EU aggregate and GB-specific gas storage fill percentages.
Requires a free API key — register at https://agsi.gie.eu/account
Set the environment variable GIE_API_KEY to enable fetching.

Data stored:
  - eu_gas_pct:  EU aggregate gas storage fill level (%)
  - gb_gas_pct:  GB gas storage fill level (%)
  - eu_gas_twh:  EU working gas in storage (TWh)
  - gb_gas_twh:  GB working gas in storage (TWh)
"""
import os
from datetime import date, timedelta

import pandas as pd

from app import db


GIE_API_KEY = os.environ.get("GIE_API_KEY", "")


def _get_client():
    """Return a GiePandasClient or None if no API key is set."""
    if not GIE_API_KEY:
        return None
    from gie import GiePandasClient
    return GiePandasClient(api_key=GIE_API_KEY)


def fetch_gas_storage(date_from: date, date_to: date) -> int:
    """
    Fetch EU and GB gas storage data from AGSI+ and upsert into the DB.
    Returns number of rows stored, or 0 if API key is not set.
    """
    client = _get_client()
    if client is None:
        print("  [Storage] Skipped — GIE_API_KEY not set "
              "(register free at https://agsi.gie.eu/account)")
        return 0

    start_str = str(date_from)
    end_str = str(date_to)

    eu_df = None
    gb_df = None

    try:
        eu_df = client.query_gas_country("EU", start=start_str, end=end_str)
    except Exception as e:
        print(f"  [Storage] Warning: EU query failed: {e}")

    try:
        gb_df = client.query_gas_country("GB", start=start_str, end=end_str)
    except Exception as e:
        print(f"  [Storage] Warning: GB query failed: {e}")

    if eu_df is None and gb_df is None:
        return 0

    # Build a combined daily DataFrame keyed by date
    rows: list[dict] = []
    all_dates: set[str] = set()

    eu_by_date: dict[str, dict] = {}
    if eu_df is not None and not eu_df.empty:
        for _, row in eu_df.iterrows():
            d = str(row.get("gasDayStart", row.name))[:10] if "gasDayStart" in eu_df.columns \
                else str(row.name)[:10]
            eu_by_date[d] = {
                "eu_gas_pct": _safe_float(row.get("full")),
                "eu_gas_twh": _safe_float(row.get("gasInStorage")),
            }
            all_dates.add(d)

    gb_by_date: dict[str, dict] = {}
    if gb_df is not None and not gb_df.empty:
        for _, row in gb_df.iterrows():
            d = str(row.get("gasDayStart", row.name))[:10] if "gasDayStart" in gb_df.columns \
                else str(row.name)[:10]
            gb_by_date[d] = {
                "gb_gas_pct": _safe_float(row.get("full")),
                "gb_gas_twh": _safe_float(row.get("gasInStorage")),
            }
            all_dates.add(d)

    for d in sorted(all_dates):
        eu = eu_by_date.get(d, {})
        gb = gb_by_date.get(d, {})
        rows.append({
            "date":       d,
            "eu_gas_pct": eu.get("eu_gas_pct"),
            "eu_gas_twh": eu.get("eu_gas_twh"),
            "gb_gas_pct": gb.get("gb_gas_pct"),
            "gb_gas_twh": gb.get("gb_gas_twh"),
        })

    n = db.upsert_gas_storage(rows)
    db.log_fetch("gas_storage", date_from, date_to, n)
    return n


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if it can't be parsed."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def missing_storage_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet in the DB."""
    min_dt, max_dt = db.get_gas_storage_date_range()

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
