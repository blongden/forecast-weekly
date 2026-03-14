"""
Octopus Agile price fetching and wholesale price derivation.

Formula (AGILE-24-10-01, region N — Southern Scotland):
    price_inc_vat  = API value_inc_vat
    price_ex_vat   = price_inc_vat / 1.05
    wholesale      = (price_ex_vat - P) / D   if 16:00 ≤ local_time < 19:00
                   =  price_ex_vat / D         otherwise

Where D = 2.1, P = 13 p/kWh  (from https://octopus.energy/blog/agile-pricing-explained/)
"""
from datetime import date, timedelta

import requests
from zoneinfo import ZoneInfo

from app import config
from app import db

LOCAL_TZ = ZoneInfo(config.TIMEZONE)


def _is_peak(dt_utc) -> bool:
    """Return True if the UTC datetime falls in the 16:00–19:00 local peak window."""
    local_hour = dt_utc.astimezone(LOCAL_TZ).hour
    return 16 <= local_hour < 19


def _wholesale(price_ex_vat: float, peak: bool) -> float:
    """Reverse-engineer wholesale cost from ex-VAT Agile price."""
    if peak:
        return (price_ex_vat - config.AGILE_P) / config.AGILE_D
    return price_ex_vat / config.AGILE_D


def fetch_prices(date_from: date, date_to: date) -> int:
    """
    Fetch half-hourly Agile prices for [date_from, date_to], derive wholesale
    costs, and upsert into the database.  Returns number of records stored.
    """
    url = (f"{config.OCTOPUS_BASE}/products/{config.OCTOPUS_PRODUCT}"
           f"/electricity-tariffs/{config.OCTOPUS_TARIFF}/standard-unit-rates/")

    params = {
        "period_from": f"{date_from}T00:00:00Z",
        "period_to":   f"{date_to}T23:59:59Z",
        "page_size":   25000,
    }

    all_records: list[dict] = []
    while url:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        all_records.extend(data.get("results", []))
        url    = data.get("next")
        params = {}

    rows = []
    for rec in all_records:
        from datetime import datetime, timezone
        dt_utc = datetime.fromisoformat(
            rec["valid_from"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        peak          = _is_peak(dt_utc)
        price_inc_vat = float(rec["value_inc_vat"])
        price_ex_vat  = price_inc_vat / config.AGILE_VAT
        wholesale     = _wholesale(price_ex_vat, peak)

        rows.append({
            "datetime":       dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price_inc_vat":  price_inc_vat,
            "price_ex_vat":   round(price_ex_vat, 6),
            "wholesale_price": round(wholesale, 6),
            "is_peak":        int(peak),
        })

    n = db.upsert_prices(rows)
    db.log_fetch("octopus", date_from, date_to, n)
    return n


def missing_price_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """
    Compare requested range with what is already in the DB and return
    a list of (start, end) date pairs that need to be fetched.

    Currently uses a simple check: if the DB has no data before date_from or
    after the latest stored date, return the gap.  This handles the common
    case of daily top-ups cleanly.
    """
    min_dt, max_dt = db.get_price_date_range()

    if min_dt is None:
        # Nothing stored at all
        return [(date_from, date_to)]

    from datetime import datetime, timezone
    stored_min = datetime.fromisoformat(min_dt.replace("Z", "+00:00")).date()
    stored_max = datetime.fromisoformat(max_dt.replace("Z", "+00:00")).date()

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        gaps.append((stored_max + timedelta(days=1), date_to))
    return gaps
