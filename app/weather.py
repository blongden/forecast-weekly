"""
Open-Meteo weather data fetching (historical archive + forecast).

UK weather sites: temperature, solar radiation, precipitation averaged across
six representative UK cities (see config.UK_WEATHER_SITES).

Wind: fetched only from wind farm sites (see config.WIND_SITES).
"""
from datetime import date, timedelta

import requests

from app import config, db


HISTORICAL_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL    = "https://api.open-meteo.com/v1/forecast"
UK_WEATHER_VARS = "temperature_2m,shortwave_radiation,precipitation"
WIND_SITE_VAR   = "wind_speed_100m"  # 100m hub height — better proxy for offshore turbines



def _parse_uk_hourly(data: dict, site_id: str) -> list[dict]:
    """Parse UK weather site data (temperature, solar, precipitation — no wind)."""
    times  = data["hourly"]["time"]
    temps  = data["hourly"]["temperature_2m"]
    rads   = data["hourly"]["shortwave_radiation"]
    precip = data["hourly"]["precipitation"]
    return [
        {
            "datetime":            t,
            "site_id":             site_id,
            "temperature_2m":      temps[i],
            "shortwave_radiation": rads[i],
            "precipitation":       precip[i],
        }
        for i, t in enumerate(times)
        if temps[i] is not None
    ]


# ── UK weather sites (historical) ─────────────────────────────────────────────

def fetch_uk_site_historical(site_id: str, lat: float, lon: float,
                              date_from: date, date_to: date) -> int:
    """Fetch hourly temperature/solar/precip for one UK site and upsert into DB."""
    resp = requests.get(
        HISTORICAL_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": str(date_from),
            "end_date":   str(date_to),
            "hourly":     UK_WEATHER_VARS,
            "timezone":   config.TIMEZONE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows = _parse_uk_hourly(resp.json(), site_id)
    n = db.upsert_uk_sites(rows)
    db.log_fetch(f"uk_weather_{site_id}", date_from, date_to, n)
    return n


def missing_uk_site_ranges(site_id: str, date_from: date,
                            date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored for this UK weather site."""
    min_dt, max_dt = db.get_uk_site_date_range(site_id)

    if min_dt is None:
        return [(date_from, date_to)]

    stored_min = date.fromisoformat(min_dt[:10])
    stored_max = date.fromisoformat(max_dt[:10])

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        safe_to = min(date_to, date.today() - timedelta(days=2))
        if stored_max < safe_to:
            gaps.append((stored_max + timedelta(days=1), safe_to))
    return gaps


# ── UK weather average forecast ───────────────────────────────────────────────

def fetch_uk_avg_forecast(days: int = 7) -> "pd.DataFrame":
    """
    Fetch hourly forecast from all UK_WEATHER_SITES and average into a single
    UK-average hourly DataFrame.  Returns columns: datetime, temperature_2m,
    shortwave_radiation, precipitation.  Not stored in DB.
    """
    import pandas as pd

    site_dfs = []
    for site_id, info in config.UK_WEATHER_SITES.items():
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude":      info["lat"],
                "longitude":     info["lon"],
                "hourly":        UK_WEATHER_VARS,
                "timezone":      config.TIMEZONE,
                "forecast_days": days,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data   = resp.json()
        times  = data["hourly"]["time"]
        temps  = data["hourly"]["temperature_2m"]
        rads   = data["hourly"]["shortwave_radiation"]
        precip = data["hourly"]["precipitation"]
        site_dfs.append(pd.DataFrame({
            "datetime":            pd.to_datetime(times),
            "temperature_2m":      temps,
            "shortwave_radiation": rads,
            "precipitation":       precip,
        }))

    combined = pd.concat(site_dfs)
    avg = (
        combined.groupby("datetime")
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            shortwave_radiation=("shortwave_radiation", "mean"),
            precipitation=("precipitation", "mean"),
        )
        .reset_index()
    )
    return avg.sort_values("datetime").reset_index(drop=True)


def daily_from_hourly(df: "pd.DataFrame") -> "pd.DataFrame":
    """Aggregate an hourly forecast DataFrame to daily averages/sums."""
    import pandas as pd

    d = df.copy()
    d["date"] = d["datetime"].dt.date
    daily = (
        d.groupby("date")
         .agg(
             temperature_2m=("temperature_2m", "mean"),
             shortwave_radiation=("shortwave_radiation", "mean"),
             precipitation=("precipitation", "sum"),
         )
         .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ── Edinburgh historical weather (kept for backwards compat / reference) ──────


# ── Offshore wind sites ───────────────────────────────────────────────────────

def fetch_wind_site_historical(site_id: str, lat: float, lon: float,
                               date_from: date, date_to: date) -> int:
    """Fetch hourly 100m wind speed for one offshore site and upsert into DB."""
    resp = requests.get(
        HISTORICAL_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": str(date_from),
            "end_date":   str(date_to),
            "hourly":     WIND_SITE_VAR,
            "timezone":   config.TIMEZONE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data   = resp.json()
    times  = data["hourly"]["time"]
    winds  = data["hourly"][WIND_SITE_VAR]
    rows   = [
        {"datetime": t, "site_id": site_id, "wind_speed": w}
        for t, w in zip(times, winds)
        if w is not None
    ]
    n = db.upsert_wind_sites(rows)
    db.log_fetch(f"wind_site_{site_id}", date_from, date_to, n)
    return n


def fetch_wind_site_forecast(site_id: str, lat: float, lon: float, days: int = 7) -> "pd.DataFrame":
    """Fetch hourly 100m wind speed forecast for one offshore site. Returns DataFrame."""
    import pandas as pd

    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude":      lat,
            "longitude":     lon,
            "hourly":        WIND_SITE_VAR,
            "timezone":      config.TIMEZONE,
            "forecast_days": days,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame({
        "datetime":  pd.to_datetime(data["hourly"]["time"]),
        "wind_speed": data["hourly"][WIND_SITE_VAR],
    }).dropna()
    df["site_id"] = site_id
    return df


def missing_wind_site_ranges(site_id: str, date_from: date,
                              date_to: date) -> list[tuple[date, date]]:
    """Return date ranges not yet stored for this wind site."""
    min_dt, max_dt = db.get_wind_site_date_range(site_id)

    if min_dt is None:
        return [(date_from, date_to)]

    stored_min = date.fromisoformat(min_dt[:10])
    stored_max = date.fromisoformat(max_dt[:10])

    gaps = []
    if date_from < stored_min:
        gaps.append((date_from, stored_min - timedelta(days=1)))
    if date_to > stored_max:
        safe_to = min(date_to, date.today() - timedelta(days=2))
        if stored_max < safe_to:
            gaps.append((stored_max + timedelta(days=1), safe_to))
    return gaps
