"""
Commodity price fetching via Yahoo Finance (yfinance).

  BZ=F  — Brent Crude futures (USD/barrel)
  TTF=F — Dutch TTF Natural Gas futures (EUR/MWh)
          TTF is the European gas benchmark and the primary driver of
          UK gas-fired generation costs, and therefore wholesale electricity prices.

Prices are stored daily.  Weekends/holidays are forward-filled at analysis time.
"""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app import db


SYMBOLS = {
    "brent_crude_usd": "BZ=F",      # USD per barrel
    "gas_ttf_eur":     "TTF=F",     # EUR per MWh
    "gbpusd":          "GBPUSD=X",  # GBP/USD exchange rate
    "usd_index":       "DX-Y.NYB",  # US Dollar Index (DXY)
    "carbon_ets_gbp":  "CO2.L",     # EU ETS carbon price (GBP/tonne, proxy for UK ETS)
}


def _download(symbol: str, date_from: date, date_to: date) -> pd.Series:
    """Download closing prices for a single Yahoo Finance symbol."""
    df = yf.download(
        symbol,
        start=str(date_from),
        end=str(date_to + timedelta(days=1)),  # yfinance end is exclusive
        progress=False,
        auto_adjust=True,
    )
    # yfinance >= 0.2 returns multi-level columns for some calls
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"].dropna()


def fetch_commodity(date_from: date, date_to: date) -> int:
    """
    Fetch Brent and TTF closing prices and upsert into the commodity_prices table.
    Returns number of rows stored.
    """
    series: dict[str, pd.Series] = {}
    for col, sym in SYMBOLS.items():
        try:
            s = _download(sym, date_from, date_to)
            series[col] = s
        except Exception as e:
            print(f"  [gas] Warning: could not fetch {sym}: {e}")

    if not series:
        return 0

    # Combine into a single DataFrame indexed by date
    combined = pd.DataFrame(series)
    combined.index = pd.to_datetime(combined.index).date

    rows = [
        {
            "date":            str(idx),
            "brent_crude_usd": float(row.get("brent_crude_usd", None))
                                if pd.notna(row.get("brent_crude_usd")) else None,
            "gas_ttf_eur":     float(row.get("gas_ttf_eur", None))
                                if pd.notna(row.get("gas_ttf_eur")) else None,
            "gbpusd":          float(row.get("gbpusd", None))
                                if pd.notna(row.get("gbpusd")) else None,
            "usd_index":       float(row.get("usd_index", None))
                                if pd.notna(row.get("usd_index")) else None,
            "carbon_ets_gbp":  float(row.get("carbon_ets_gbp", None))
                                if pd.notna(row.get("carbon_ets_gbp")) else None,
        }
        for idx, row in combined.iterrows()
    ]

    n = db.upsert_commodity(rows)
    db.log_fetch("commodity", date_from, date_to, n)
    return n


def missing_commodity_ranges(date_from: date, date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet in the DB."""
    min_dt, max_dt = db.get_commodity_date_range()

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
