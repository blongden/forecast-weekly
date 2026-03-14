"""
Feature engineering: constants, data loading, and feature building.

All feature dictionaries (WEATHER_VARS, DAILY_FEATURES, HH_FEATURES, etc.)
and data-loading functions (load_daily_df, build_halfhourly_df) live here.
Other modules import constants and helpers from this file.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import holidays as holidays_lib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app import db
from app.config import WIND_SITES

LOCAL_TZ = ZoneInfo("Europe/London")

# UK bank holidays (England + Scotland union) for is_bank_holiday feature.
# Lazy-built on first access to keep startup fast.
_UK_HOLIDAYS: set[date] | None = None


def _uk_holidays() -> set[date]:
    global _UK_HOLIDAYS
    if _UK_HOLIDAYS is None:
        eng = holidays_lib.country_holidays("GB", subdiv="ENG", years=range(2023, 2030))
        sct = holidays_lib.country_holidays("GB", subdiv="SCT", years=range(2023, 2030))
        _UK_HOLIDAYS = set(eng.keys()) | set(sct.keys())
    return _UK_HOLIDAYS

# Base weather variables (used for correlations and scatter plots).
# UK averages across config.UK_WEATHER_SITES — no Edinburgh-only wind_speed_10m.
WEATHER_VARS = {
    "temperature_2m":      "Temperature (°C, UK avg)",
    "heating_dd":          "Heating Degree Days (base 15.5°C)",
    "shortwave_radiation": "Solar Radiation (W/m², UK avg)",
    "precipitation":       "Precipitation (mm/day, UK avg)",
}

# Actual GB solar generation from Sheffield Solar PV_Live (used in model).
# Replaces shortwave_radiation in regression — directly measures supply-side
# solar rather than proxying through radiation.  For the forecast horizon,
# solar_gw is estimated from shortwave_radiation (see estimate_solar_from_radiation).
SOLAR_FEATURES = {
    "solar_gw": "Solar Generation (GW, GB actual)",
}

# GB demand from Elexon BMRS INDO — strong positive correlator (high demand → high price).
# For the forecast horizon, demand is estimated from a historical day-of-week / hour profile.
DEMAND_FEATURES = {
    "demand_mw": "GB Demand (MW)",
}

# GB supply from Elexon BMRS FUELHH.
# wind_gen_mw: actual wind output — negatively correlated (more wind → lower price).
# imports_mw:  net interconnector imports — negatively correlated (cheap imports suppress UK price).
# For forecast: wind_gen_mw estimated from wind speed; imports_mw from historical profile.
SUPPLY_FEATURES = {
    "wind_gen_mw":         "GB Wind Generation (MW)",
    "gas_gen_mw":          "GB Gas Generation (MW)",
    "nuclear_mw":          "GB Nuclear Generation (MW)",
    "pumped_storage_mw":   "GB Pumped Storage (MW)",
    "hydro_mw":            "GB Hydro (MW)",
    "imports_mw":          "GB Net Interconnector Imports (MW)",
}

# Calendar features
CALENDAR_FEATURES = {
    "is_bank_holiday": "Bank Holiday",
    "is_weekend":      "Weekend (Sat/Sun)",
}

# EPEX SPOT GB day-ahead prices — 1-day lag.
# The day-ahead auction clears at ~12:00 for next-day delivery; yesterday's
# clearing price is therefore always available when forecasting today/tomorrow.
# Strong autocorrelation with today's Agile price (both driven by same market).
MIDPRICE_FEATURES = {
    "epex_lag1_gbp_mwh": "EPEX Day-Ahead Lag-1 (£/MWh)",
}

# Interaction terms added to regression models.
# uk_avg_wind is computed at feature-build time as the mean of all wind-farm site
# winds — it is NOT added as a standalone feature to avoid multicollinearity.
# temp_x_wind  : cold AND calm → demand high, wind supply low → price spike
# wind_x_solar : calm AND low solar → both renewables suppressed → gas-only market
INTERACTION_FEATURES = {
    "temp_x_wind":  "Temp × UK avg wind",
    "wind_x_solar": "UK avg wind × Solar Gen",
}

# Commodity features (gas/oil prices, lagged via 7-day rolling average)
# TTF is the European gas benchmark — the primary fuel-cost driver of UK electricity.
# Brent crude is included as a correlated proxy and user-requested feature.
# Rolling averages smooth out day-to-day noise; the lag is implicit (we use recent
# prices as the best available signal for the current and near-future electricity price).
COMMODITY_FEATURES = {
    "gas_ttf_roll7":    "TTF Gas 7-day avg (€/MWh)",
    "brent_roll7":      "Brent Crude 7-day avg ($/bbl)",
    "gbpusd_roll7":     "GBP/USD 7-day avg",
    "dxy_roll7":        "US Dollar Index 7-day avg",
    "carbon_roll7":     "EU Carbon 7-day avg (\u00a3/tonne)",
}

# Inventory / storage features — slower-moving but structurally important.
# Gas storage fill % is a major seasonal driver (low storage = winter anxiety → higher prices).
# US crude stocks delta captures surprise weekly builds/draws that move global oil prices.
INVENTORY_FEATURES = {
    "eu_gas_storage_pct":     "EU Gas Storage Fill (%)",
    "gb_gas_storage_pct":     "GB Gas Storage Fill (%)",
    "us_crude_stocks_delta":  "US Crude Stocks WoW Change (mb)",
}

# Lag and rolling features — capture autoregressive patterns, volatility, momentum.
# All computed from data available at forecast time (shifted by ≥1 day).
LAG_ROLLING_FEATURES = {
    "epex_lag7_gbp_mwh":  "EPEX Day-Ahead Lag-7 (£/MWh)",
    "epex_roll7_std":     "EPEX 7-Day Volatility (£/MWh)",
    "epex_roll7_min":     "EPEX 7-Day Min (£/MWh)",
    "epex_roll7_max":     "EPEX 7-Day Max (£/MWh)",
    "epex_momentum_7":    "EPEX 7-Day Momentum (£/MWh)",
}

# Wind site features — one per offshore/onshore site
# uk_avg_wind is intentionally excluded to avoid multicollinearity.
# Ridge regularisation handles remaining inter-site correlations.
WIND_SITE_FEATURES = {f"wind_{sid}": info["label"] for sid, info in WIND_SITES.items()}

# ENTSO-E Transparency Platform features — interconnector flows & generation outages.
# All genuinely forecastable: scheduled exchanges are published day-ahead; REMIT outages
# are published ahead of time (planned weeks ahead, forced within hours).
ENTSOE_FEATURES = {
    "net_scheduled_imports_mw":  "Net Scheduled Imports (MW, day-ahead)",
    "nuclear_unavailable_mw":    "Nuclear Unavailable (MW, planned+forced)",
    "total_unavailable_mw":      "Total Unavailable Capacity (MW)",
}

# System price features — balancing mechanism signals market stress.
# Lag-1 daily average system price and mean absolute imbalance volume.
SYSTEM_PRICE_FEATURES = {
    "sysprice_lag1_gbp_mwh": "System Price Lag-1 (£/MWh)",
    "abs_imbalance_lag1_mw": "Abs Imbalance Vol Lag-1 (MW)",
}

# Ramp rate features — hour-to-hour changes capture volatility events.
RAMP_FEATURES = {
    "wind_ramp_mw":  "Wind Ramp (MW, 24h change)",
    "solar_ramp_gw": "Solar Ramp (GW, 24h change)",
}

# Human-readable labels for ALL regression features
ALL_FEATURE_LABELS = {
    **WEATHER_VARS, **SOLAR_FEATURES, **DEMAND_FEATURES, **SUPPLY_FEATURES,
    **CALENDAR_FEATURES, **MIDPRICE_FEATURES,
    **INTERACTION_FEATURES, **COMMODITY_FEATURES, **INVENTORY_FEATURES,
    **LAG_ROLLING_FEATURES, **WIND_SITE_FEATURES, **ENTSOE_FEATURES,
    **SYSTEM_PRICE_FEATURES, **RAMP_FEATURES,
    "net_residual_mw": "Net Residual Demand (MW)",
    "price_lag1_slot": "EPEX Lag-1 Same Slot (p/kWh)",
    "is_peak":         "Peak (16:00–19:00)",
    "hour_sin":        "Hour sin",
    "hour_cos":        "Hour cos",
    "doy_sin":         "Day-of-year sin",
    "doy_cos":         "Day-of-year cos",
}

# Ordered feature lists for each model.
# solar_gw replaces shortwave_radiation as the primary solar signal in the model.
# shortwave_radiation is kept in WEATHER_VARS for correlation display only.
DAILY_FEATURES = (["temperature_2m", "heating_dd", "solar_gw",
                   "wind_gen_mw",
                   "precipitation", "is_bank_holiday", "is_weekend",
                   "epex_lag1_gbp_mwh"] +
                  list(INTERACTION_FEATURES.keys()) +
                  list(COMMODITY_FEATURES.keys()) +
                  list(INVENTORY_FEATURES.keys()) +
                  list(LAG_ROLLING_FEATURES.keys()) +
                  list(ENTSOE_FEATURES.keys()) +
                  list(SYSTEM_PRICE_FEATURES.keys()) +
                  list(RAMP_FEATURES.keys()) +
                  list(WIND_SITE_FEATURES.keys()))

HH_FEATURES = [
    # Weather — forecastable from NWP
    "temperature_2m", "heating_dd", "precipitation",
    # Generation forecastable from weather
    "solar_gw", "wind_gen_mw",
    # Net residual demand — demand minus renewables/nuclear; marginal price signal
    "net_residual_mw",
    # Wind site speeds — hourly forecast available
    "wind_dogger_bank", "wind_hornsea", "wind_walney",
    "wind_whitelee", "wind_clyde_wind", "wind_pen_y_cymoedd",
    # Interaction terms
    "temp_x_wind", "wind_x_solar",
    # Commodity prices & currency — daily, always known ahead of time
    "gas_ttf_roll7", "brent_roll7", "gbpusd_roll7", "dxy_roll7", "carbon_roll7",
    # Inventory / storage — daily (gas) or weekly (oil), always known ahead of time
    "eu_gas_storage_pct", "gb_gas_storage_pct", "us_crude_stocks_delta",
    # Lag / rolling — autoregressive patterns (daily values, known ahead of time)
    "epex_lag7_gbp_mwh", "epex_roll7_std", "epex_roll7_min", "epex_roll7_max",
    "epex_momentum_7",
    # Price autoregression — yesterday's same slot, always known
    "price_lag1_slot",
    # ENTSO-E — interconnector scheduled flows & generation outages (forecastable)
    "net_scheduled_imports_mw", "nuclear_unavailable_mw", "total_unavailable_mw",
    # System price — balancing mechanism signals (lagged 1 day)
    "sysprice_lag1_gbp_mwh", "abs_imbalance_lag1_mw",
    # Ramp rates — 24h change in wind/solar supply
    "wind_ramp_mw", "solar_ramp_gw",
    # Calendar
    "is_bank_holiday", "is_weekend", "is_peak",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


# ── Feature building helpers ──────────────────────────────────────────────────

def _add_solar_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Join daily average solar generation (GW) from PVLIVE into the daily df."""
    rows = db.get_daily_solar(date_from, date_to)
    if not rows:
        return df
    solar_df = pd.DataFrame(rows, columns=["date", "solar_gw", "solar_gwh"])
    solar_df["date"] = pd.to_datetime(solar_df["date"])
    return pd.merge(df, solar_df[["date", "solar_gw"]], on="date", how="left")


def _add_demand_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Join daily average demand (MW) from INDO into the daily df."""
    rows = db.get_daily_demand(date_from, date_to)
    if not rows:
        return df
    demand_df = pd.DataFrame(rows, columns=["date", "demand_mw"])
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    return pd.merge(df, demand_df, on="date", how="left")


def build_demand_profile(df: pd.DataFrame,
                         dt_col: str = "datetime_local") -> dict:
    """
    Compute historical average demand by (day_of_week, hour) for use as a
    demand estimate over the forecast horizon where actual data is unavailable.
    Returns {(day_of_week, hour): mean_demand_mw}.
    """
    if "demand_mw" not in df.columns or df["demand_mw"].isna().all():
        return {}
    d = df.dropna(subset=["demand_mw"]).copy()
    d["_dow"]  = d[dt_col].dt.dayofweek
    d["_hour"] = d[dt_col].dt.hour
    return d.groupby(["_dow", "_hour"])["demand_mw"].mean().to_dict()


def _add_supply_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Join daily average wind generation and imports from FUELHH into the daily df."""
    rows = db.get_daily_generation(date_from, date_to)
    if not rows:
        return df
    gen_df = pd.DataFrame(rows, columns=["date", "wind_gen_mw", "gas_gen_mw",
                                          "nuclear_mw", "pumped_storage_mw",
                                          "hydro_mw", "imports_mw"])
    gen_df["date"] = pd.to_datetime(gen_df["date"])
    return pd.merge(df, gen_df[["date", "wind_gen_mw", "gas_gen_mw", "nuclear_mw",
                                 "pumped_storage_mw", "hydro_mw", "imports_mw"]], on="date", how="left")


def _add_midprice_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Join EPEX day-ahead price (1-day lag) from market_index_halfhourly into the daily df."""
    rows = db.get_daily_midprice(date_from, date_to)
    if not rows:
        return df
    mp_df = pd.DataFrame(rows, columns=["date", "epex_gbp_mwh"])
    mp_df["date"] = pd.to_datetime(mp_df["date"])
    # Lag by 1 day: use yesterday's price to predict today's
    mp_df["epex_lag1_gbp_mwh"] = mp_df["epex_gbp_mwh"].shift(1)
    return pd.merge(df, mp_df[["date", "epex_lag1_gbp_mwh"]], on="date", how="left")


def estimate_wind_gen_from_speed(df_historical: pd.DataFrame,
                                  wind_speed_series: pd.Series) -> pd.Series:
    """
    Estimate wind_gen_mw from uk_avg_wind using a polynomial Ridge model fitted on
    historical (wind_speed, wind_gen_mw) pairs.

    Uses wind_speed, wind_speed² and wind_speed³ to capture the nonlinear turbine
    power curve (cubic below rated speed, plateau at rated, cut-out at high speed).
    Returns a non-negative Series aligned with wind_speed_series.
    """
    if "wind_gen_mw" not in df_historical.columns:
        return pd.Series(np.nan, index=wind_speed_series.index)
    # Build wind speed proxy for historical daily df from wind site columns
    wind_cols = [c for c in df_historical.columns if c.startswith("wind_")]
    if not wind_cols:
        return pd.Series(np.nan, index=wind_speed_series.index)
    hist = df_historical.copy()
    hist["_uk_avg_wind"] = hist[wind_cols].mean(axis=1)
    valid = hist[["_uk_avg_wind", "wind_gen_mw"]].dropna()
    if len(valid) < 30:
        return pd.Series(np.nan, index=wind_speed_series.index)
    ws = valid["_uk_avg_wind"].values.reshape(-1, 1)
    X = np.hstack([ws, ws ** 2, ws ** 3])
    y = valid["wind_gen_mw"].values
    scaler = StandardScaler()
    model  = Ridge(alpha=1.0)
    model.fit(scaler.fit_transform(X), y)
    ws_fc = wind_speed_series.values.reshape(-1, 1)
    X_fc = np.hstack([ws_fc, ws_fc ** 2, ws_fc ** 3])
    est  = model.predict(scaler.transform(X_fc))
    return pd.Series(np.clip(est, 0, None), index=wind_speed_series.index)


def build_imports_profile(df: pd.DataFrame,
                           dt_col: str = "datetime_local") -> dict:
    """
    Compute historical average imports_mw by (day_of_week, hour).
    Returns {(day_of_week, hour): mean_imports_mw}.
    """
    if "imports_mw" not in df.columns or df["imports_mw"].isna().all():
        return {}
    d = df.dropna(subset=["imports_mw"]).copy()
    d["_dow"]  = d[dt_col].dt.dayofweek
    d["_hour"] = d[dt_col].dt.hour
    return d.groupby(["_dow", "_hour"])["imports_mw"].mean().to_dict()


def estimate_solar_from_radiation(df_historical: pd.DataFrame,
                                   shortwave_series: "pd.Series") -> "pd.Series":
    """
    Estimate solar_gw from shortwave_radiation using a linear model fitted on
    historical PVLIVE data.  Used for the forecast horizon where actual
    generation is not yet available.
    Returns a non-negative Series aligned with shortwave_series.
    """
    hist = df_historical[["shortwave_radiation", "solar_gw"]].dropna()
    if len(hist) < 30:
        return pd.Series(0.0, index=shortwave_series.index)
    X = hist[["shortwave_radiation"]].values
    y = hist["solar_gw"].values
    scaler = StandardScaler()
    model  = Ridge(alpha=1.0)
    model.fit(scaler.fit_transform(X), y)
    X_fc = shortwave_series.values.reshape(-1, 1)
    est  = model.predict(scaler.transform(X_fc))
    return pd.Series(np.clip(est, 0, None), index=shortwave_series.index)


def _add_wind_site_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """
    Join daily avg wind speeds from each offshore site and compute uk_avg_wind.
    Silently skips sites with no data yet.
    """
    rows = db.get_daily_wind_sites(date_from, date_to)
    if not rows:
        return df

    wind_df = pd.DataFrame(rows, columns=["date", "site_id", "wind_speed"])
    wind_df["date"] = pd.to_datetime(wind_df["date"])
    # Pivot to one column per site
    pivoted = wind_df.pivot(index="date", columns="site_id", values="wind_speed")
    pivoted.columns = [f"wind_{c}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    df = pd.merge(df, pivoted, on="date", how="left")

    # Composite: mean across all available site columns
    return df


def _add_commodity_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """
    Join commodity prices, forward-fill weekends/holidays, and add
    7-day rolling averages.  Rows missing both commodity values are dropped.
    """
    commodity = pd.DataFrame(
        db.get_commodity_prices(date_from, date_to),
        columns=["date", "brent_crude_usd", "gas_ttf_eur", "gbpusd", "usd_index", "carbon_ets_gbp"],
    )
    if commodity.empty:
        return df  # commodity data not yet fetched; skip silently

    commodity["date"] = pd.to_datetime(commodity["date"])
    df = pd.merge(df, commodity, on="date", how="left")

    # Forward-fill gaps (weekends, public holidays) up to 5 days
    df["brent_crude_usd"] = df["brent_crude_usd"].ffill(limit=5)
    df["gas_ttf_eur"]     = df["gas_ttf_eur"].ffill(limit=5)
    df["gbpusd"]          = df["gbpusd"].ffill(limit=5)
    df["usd_index"]       = df["usd_index"].ffill(limit=5)
    df["carbon_ets_gbp"]  = df["carbon_ets_gbp"].ffill(limit=5)

    # 7-day rolling average to smooth short-term noise and capture lag
    df["brent_roll7"]   = df["brent_crude_usd"].rolling(7, min_periods=3).mean()
    df["gas_ttf_roll7"] = df["gas_ttf_eur"].rolling(7, min_periods=3).mean()
    df["gbpusd_roll7"]  = df["gbpusd"].rolling(7, min_periods=3).mean()
    df["dxy_roll7"]     = df["usd_index"].rolling(7, min_periods=3).mean()
    df["carbon_roll7"]  = df["carbon_ets_gbp"].rolling(7, min_periods=3).mean()

    return df


def _add_inventory_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """
    Join gas storage and oil inventory data into the daily df.
    Gas storage: forward-filled daily (reported daily, may lag 1-2 days).
    Oil inventory: weekly → forward-filled to daily, plus week-over-week delta.
    """
    # ── Gas storage (AGSI+) ────────────────────────────────────────────────
    storage_rows = db.get_gas_storage(date_from, date_to)
    if storage_rows:
        storage_df = pd.DataFrame(storage_rows,
                                   columns=["date", "eu_gas_pct", "eu_gas_twh",
                                            "gb_gas_pct", "gb_gas_twh"])
        storage_df["date"] = pd.to_datetime(storage_df["date"])
        df = pd.merge(df, storage_df[["date", "eu_gas_pct", "gb_gas_pct"]],
                       on="date", how="left")
        df["eu_gas_pct"] = df["eu_gas_pct"].ffill(limit=5)
        df["gb_gas_pct"] = df["gb_gas_pct"].ffill(limit=5)
        df["eu_gas_storage_pct"] = df["eu_gas_pct"]
        df["gb_gas_storage_pct"] = df["gb_gas_pct"]

    # ── Oil inventory (EIA) ────────────────────────────────────────────────
    oil_rows = db.get_oil_inventory(date_from, date_to)
    if oil_rows:
        oil_df = pd.DataFrame(oil_rows, columns=["date", "us_crude_stocks_mb"])
        oil_df["date"] = pd.to_datetime(oil_df["date"])
        df = pd.merge(df, oil_df, on="date", how="left")
        df["us_crude_stocks_mb"] = df["us_crude_stocks_mb"].ffill(limit=7)
        # Week-over-week change (positive = build, negative = draw)
        df["us_crude_stocks_delta"] = df["us_crude_stocks_mb"].diff()
        # First value after ffill will be 0 diff; keep NaN for the very first row
        df.loc[df["us_crude_stocks_mb"].isna(), "us_crude_stocks_delta"] = np.nan

    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add autoregressive lag and rolling-window features from the EPEX price series.
    All features use shift(≥1) to avoid look-ahead bias — values at time t
    are computed from t-1 and earlier only.
    """
    if "epex_lag1_gbp_mwh" not in df.columns:
        return df
    price = df["epex_lag1_gbp_mwh"]  # already shifted by 1 day
    # Lag-7: same day last week (shifted 1 more from lag1 = total shift of 7+1, but
    # lag1 is already yesterday's price; we want price from 7 days ago directly)
    if "avg_epex_p_kwh" in df.columns:
        raw_price = df["avg_epex_p_kwh"]
        df["epex_lag7_gbp_mwh"] = raw_price.shift(7) * 10  # p/kWh → £/MWh for consistency
    # Rolling stats on the lag-1 series (already shifted, so no leakage)
    df["epex_roll7_std"] = price.rolling(7, min_periods=3).std()
    df["epex_roll7_min"] = price.rolling(7, min_periods=3).min()
    df["epex_roll7_max"] = price.rolling(7, min_periods=3).max()
    # Momentum: lag1 minus lag7 (positive = prices rising)
    if "epex_lag7_gbp_mwh" in df.columns:
        df["epex_momentum_7"] = price - df["epex_lag7_gbp_mwh"]
    return df


def _add_system_price_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """Join daily system price lag features into the daily df."""
    rows = db.get_daily_system_prices(date_from, date_to)
    if not rows:
        return df
    sp_df = pd.DataFrame(rows, columns=["date", "avg_system_price", "avg_abs_imbalance_mw"])
    sp_df["date"] = pd.to_datetime(sp_df["date"])
    df = pd.merge(df, sp_df, on="date", how="left")
    # Lag by 1 day — system prices settle next day, so yesterday's is always known
    df["sysprice_lag1_gbp_mwh"] = df["avg_system_price"].shift(1)
    df["abs_imbalance_lag1_mw"] = df["avg_abs_imbalance_mw"].shift(1)
    return df


def _add_entsoe_features(df: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """
    Join ENTSO-E scheduled imports and generation unavailability into the daily df.
    Silently skips if no ENTSO-E data is available.
    """
    # ── Scheduled imports ──────────────────────────────────────────────────
    import_rows = db.get_daily_scheduled_imports(date_from, date_to)
    if import_rows:
        imp_df = pd.DataFrame(import_rows, columns=["date", "net_scheduled_imports_mw"])
        imp_df["date"] = pd.to_datetime(imp_df["date"])
        df = pd.merge(df, imp_df, on="date", how="left")

    # ── Generation unavailability ──────────────────────────────────────────
    unavail_rows = db.get_daily_unavailability(date_from, date_to)
    if unavail_rows:
        unavail_df = pd.DataFrame(unavail_rows, columns=["date", "fuel_type", "unavailable_mw"])
        unavail_df["date"] = pd.to_datetime(unavail_df["date"])
        pivot = unavail_df.pivot_table(
            index="date", columns="fuel_type",
            values="unavailable_mw", aggfunc="sum", fill_value=0,
        ).reset_index()
        pivot["total_unavailable_mw"] = pivot.select_dtypes(include="number").sum(axis=1)
        pivot["nuclear_unavailable_mw"] = pivot.get("nuclear", pd.Series(0, index=pivot.index))
        df = pd.merge(df, pivot[["date", "total_unavailable_mw", "nuclear_unavailable_mw"]],
                       on="date", how="left")

    return df


# ── Data loading ──────────────────────────────────────────────────────────────

def load_daily_df(date_from: date, date_to: date) -> pd.DataFrame:
    """Load and merge daily EPEX wholesale prices + UK-average weather (+ commodity prices)."""
    epex_rows = db.get_daily_midprice(date_from, date_to)
    epex = pd.DataFrame(epex_rows, columns=["date", "price_gbp_mwh"])
    epex["avg_epex_p_kwh"] = epex["price_gbp_mwh"] * 0.1
    epex["date"] = pd.to_datetime(epex["date"])

    weather = pd.DataFrame(db.get_daily_uk_avg(date_from, date_to),
                           columns=["date", "temperature_2m",
                                    "shortwave_radiation", "precipitation"])
    weather["date"] = pd.to_datetime(weather["date"])

    df = pd.merge(epex[["date", "avg_epex_p_kwh"]], weather, on="date", how="inner").dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = _add_commodity_features(df, date_from, date_to)
    df = _add_inventory_features(df, date_from, date_to)
    df = _add_wind_site_features(df, date_from, date_to)
    df = _add_solar_features(df, date_from, date_to)
    df = _add_demand_features(df, date_from, date_to)
    df = _add_supply_features(df, date_from, date_to)
    df = _add_midprice_features(df, date_from, date_to)
    df = df.sort_values("date").reset_index(drop=True)
    # Heating degree days: non-linear demand signal (cold snaps drive more demand)
    df["heating_dd"] = (15.5 - df["temperature_2m"]).clip(lower=0.0)
    # Bank holiday flag: demand profile resembles Sunday regardless of calendar day
    bh = _uk_holidays()
    df["is_bank_holiday"] = df["date"].apply(lambda d: 1 if d.date() in bh else 0)
    # Weekend flag: Saturday/Sunday have systematically lower demand and prices
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    # Interaction terms using mean of wind-farm sites as the wind signal
    wind_site_cols = [c for c in df.columns if c.startswith("wind_")]
    uk_avg_wind = df[wind_site_cols].mean(axis=1) if wind_site_cols else pd.Series(0.0, index=df.index)
    solar_signal = df["solar_gw"] if "solar_gw" in df.columns else df.get("shortwave_radiation", pd.Series(0.0, index=df.index))
    df["temp_x_wind"]  = df["temperature_2m"] * uk_avg_wind
    df["wind_x_solar"] = uk_avg_wind * solar_signal
    df = _add_lag_rolling_features(df)
    # Ramp rates: 24h change in wind/solar captures rapid supply shifts
    if "wind_gen_mw" in df.columns:
        df["wind_ramp_mw"] = df["wind_gen_mw"].diff()
    if "solar_gw" in df.columns:
        df["solar_ramp_gw"] = df["solar_gw"].diff()
    df = _add_entsoe_features(df, date_from, date_to)
    df = _add_system_price_features(df, date_from, date_to)
    return df.sort_values("date").reset_index(drop=True)


def compute_correlations(df: pd.DataFrame) -> dict:
    """Return dict of {var: {r, p, label}} Pearson correlations with ex-VAT price."""
    results = {}
    vars_to_correlate = {**WEATHER_VARS, **SOLAR_FEATURES, **DEMAND_FEATURES,
                         **SUPPLY_FEATURES, **MIDPRICE_FEATURES, **COMMODITY_FEATURES,
                         **INVENTORY_FEATURES, **ENTSOE_FEATURES,
                         **SYSTEM_PRICE_FEATURES, **RAMP_FEATURES}
    for var, label in vars_to_correlate.items():
        if var not in df.columns or df[var].isna().all():
            continue
        valid = df[["avg_epex_p_kwh", var]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.pearsonr(valid[var], valid["avg_epex_p_kwh"])
        results[var] = {"r": r, "p": p, "label": label}
    return results


def _time_features(dt_local: datetime) -> dict:
    """Cyclic hour-of-day and day-of-year features from a local datetime."""
    hour = dt_local.hour + dt_local.minute / 60
    doy  = dt_local.timetuple().tm_yday
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin":  np.sin(2 * np.pi * doy  / 365),
        "doy_cos":  np.cos(2 * np.pi * doy  / 365),
    }


def build_halfhourly_df(date_from: date, date_to: date) -> pd.DataFrame:
    """
    Join half-hourly prices to hourly weather by local time and add
    cyclic time features.  Prices (UTC) are converted to Europe/London
    to find the matching weather hour.  Daily commodity rolling averages
    are joined by date.
    """
    weather_rows = db.get_hourly_uk_avg(date_from, date_to)

    # Build per-slot EPEX price lookup keyed by UTC datetime string (£/MWh → p/kWh)
    epex_rows = db.get_halfhourly_midprice(date_from, date_to)
    epex_by_utc: dict[str, float] = {
        row["datetime_utc"]: row["price_gbp_mwh"] * 0.1
        for row in epex_rows
        if row["price_gbp_mwh"] is not None
    }

    # Build weather lookup: "YYYY-MM-DDTHH:00" → row dict
    weather = {row["datetime"]: dict(row) for row in weather_rows}

    # Build 30-min solar lookup keyed by UTC datetime string
    solar_rows = db.get_halfhourly_solar(date_from, date_to)
    solar_by_gmt: dict[str, float] = {
        row["datetime_gmt"]: (row["generation_mw"] / 1000.0 if row["generation_mw"] is not None else 0.0)
        for row in solar_rows
    }

    # Build 30-min demand lookup keyed by UTC datetime string
    demand_rows = db.get_halfhourly_demand(date_from, date_to)
    demand_by_utc: dict[str, float] = {
        row["datetime_utc"]: row["demand_mw"]
        for row in demand_rows
        if row["demand_mw"] is not None
    }

    # Build 30-min generation lookup (wind, gas, nuclear, pumped storage, hydro, imports)
    gen_rows = db.get_halfhourly_generation(date_from, date_to)
    wind_gen_by_utc:    dict[str, float] = {}
    gas_gen_by_utc:     dict[str, float] = {}
    nuclear_by_utc:     dict[str, float] = {}
    pumped_storage_by_utc: dict[str, float] = {}
    hydro_by_utc:       dict[str, float] = {}
    imports_by_utc:     dict[str, float] = {}
    for row in gen_rows:
        dt_key = row["datetime_utc"]
        if row["wind_mw"]           is not None: wind_gen_by_utc[dt_key]    = row["wind_mw"]
        if row["gas_mw"]            is not None: gas_gen_by_utc[dt_key]     = row["gas_mw"]
        if row["nuclear_mw"]        is not None: nuclear_by_utc[dt_key]     = row["nuclear_mw"]
        if row["pumped_storage_mw"] is not None: pumped_storage_by_utc[dt_key] = row["pumped_storage_mw"]
        if row["hydro_mw"]          is not None: hydro_by_utc[dt_key]       = row["hydro_mw"]
        if row["imports_mw"]        is not None: imports_by_utc[dt_key]     = row["imports_mw"]

    # Build commodity + EPEX lag lookup keyed by date (wind sites now per-hour)
    _temp_daily = load_daily_df(date_from, date_to)
    daily_by_date: dict[str, dict] = {}
    for _, row in _temp_daily.iterrows():
        dk = str(row["date"].date())
        daily_by_date[dk] = {
            "brent_roll7":           row.get("brent_roll7"),
            "gas_ttf_roll7":         row.get("gas_ttf_roll7"),
            "gbpusd_roll7":          row.get("gbpusd_roll7"),
            "dxy_roll7":             row.get("dxy_roll7"),
            "carbon_roll7":          row.get("carbon_roll7"),
            "eu_gas_storage_pct":    row.get("eu_gas_storage_pct"),
            "gb_gas_storage_pct":    row.get("gb_gas_storage_pct"),
            "us_crude_stocks_delta": row.get("us_crude_stocks_delta"),
            "epex_lag7_gbp_mwh":     row.get("epex_lag7_gbp_mwh"),
            "epex_roll7_std":        row.get("epex_roll7_std"),
            "epex_roll7_min":        row.get("epex_roll7_min"),
            "epex_roll7_max":        row.get("epex_roll7_max"),
            "epex_momentum_7":       row.get("epex_momentum_7"),
            "wind_ramp_mw":          row.get("wind_ramp_mw"),
            "solar_ramp_gw":         row.get("solar_ramp_gw"),
            "sysprice_lag1_gbp_mwh": row.get("sysprice_lag1_gbp_mwh"),
            "abs_imbalance_lag1_mw": row.get("abs_imbalance_lag1_mw"),
        }

    # Build hourly wind-site lookup: "YYYY-MM-DDTHH:00" → {site_id: wind_speed}
    hourly_wind_rows = db.get_hourly_wind_sites(date_from, date_to)
    hourly_wind: dict[str, dict[str, float]] = {}
    for r in hourly_wind_rows:
        # datetime is "YYYY-MM-DDTHH:MM" — normalise to "YYYY-MM-DDTHH:00"
        hkey = r["datetime"][:13] + ":00"
        hourly_wind.setdefault(hkey, {})[r["site_id"]] = r["wind_speed"]

    # Build hourly scheduled imports lookup: UTC "YYYY-MM-DDTHH:MM:SS" → net MW
    hourly_imports_rows = db.get_hourly_scheduled_imports(date_from, date_to)
    entsoe_imports_by_utc: dict[str, float] = {}
    for r in hourly_imports_rows:
        # Key format from DB: "YYYY-MM-DDTHH:MM:SS" — store both hour start variants
        dt_str = r[0] if isinstance(r, (tuple, list)) else r["datetime_utc"]
        net_mw = r[1] if isinstance(r, (tuple, list)) else r["net_mw"]
        entsoe_imports_by_utc[dt_str] = float(net_mw) if net_mw is not None else 0.0

    # Build daily unavailability lookup: "YYYY-MM-DD" → {nuclear_mw, total_mw}
    unavail_rows = db.get_daily_unavailability(date_from, date_to)
    daily_unavail: dict[str, dict[str, float]] = {}
    for r in unavail_rows:
        d_str = r[0] if isinstance(r, (tuple, list)) else r["date"]
        fuel  = r[1] if isinstance(r, (tuple, list)) else r["fuel_type"]
        mw    = float(r[2] if isinstance(r, (tuple, list)) else r["unavailable_mw"])
        entry = daily_unavail.setdefault(d_str, {"nuclear": 0.0, "total": 0.0})
        entry["total"] += mw
        if fuel == "nuclear":
            entry["nuclear"] += mw

    records = []
    for row in epex_rows:
        if row["price_gbp_mwh"] is None:
            continue
        dt_utc   = datetime.fromisoformat(row["datetime_utc"] + "+00:00")
        dt_local = dt_utc.astimezone(LOCAL_TZ)
        key      = dt_local.strftime("%Y-%m-%dT%H:00")
        w        = weather.get(key)
        if w is None:
            continue
        date_key = dt_local.strftime("%Y-%m-%d")
        d = daily_by_date.get(date_key, {})
        # Per-hour wind site speeds (fall back to empty if no data for this hour)
        hw = hourly_wind.get(key, {})
        wind_site_cols = {f"wind_{sid}": hw.get(sid) for sid in WIND_SITES}
        wind_vals = [v for v in wind_site_cols.values() if v is not None]
        uk_avg_wind = float(np.mean(wind_vals)) if wind_vals else 0.0
        # Solar generation: PVLIVE uses period-END timestamps (e.g. "00:30" = slot 00:00-00:30),
        # Octopus uses period-START ("00:00" = same slot), so shift +30 min to match.
        pvlive_key = (dt_utc + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
        solar_gw = solar_by_gmt.get(pvlive_key)
        if solar_gw is None:
            solar_gw = float(np.nan)
        # Demand and generation: UTC period-start matches Octopus convention directly
        utc_key    = dt_utc.strftime("%Y-%m-%dT%H:%M:%S")
        lag_utc_key = (dt_utc - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        # Lag uses yesterday's EPEX price at the same slot — the model trains on EPEX target
        price_lag1_slot = epex_by_utc.get(lag_utc_key, float(np.nan))
        demand_mw         = demand_by_utc.get(utc_key, float(np.nan))
        wind_gen_mw       = wind_gen_by_utc.get(utc_key, float(np.nan))
        gas_gen_mw        = gas_gen_by_utc.get(utc_key, float(np.nan))
        nuclear_mw        = nuclear_by_utc.get(utc_key, float(np.nan))
        pumped_storage_mw = pumped_storage_by_utc.get(utc_key, float(np.nan))
        hydro_mw          = hydro_by_utc.get(utc_key, float(np.nan))
        imports_mw        = imports_by_utc.get(utc_key, float(np.nan))
        is_bh = 1 if dt_local.date() in _uk_holidays() else 0
        is_wkend = 1 if dt_local.weekday() >= 5 else 0
        is_peak = 1 if 16 <= dt_local.hour < 19 else 0
        # ENTSO-E: hourly scheduled imports (match to nearest hour UTC)
        entsoe_hour_key = dt_utc.strftime("%Y-%m-%dT%H:00:00")
        net_sched_imports = entsoe_imports_by_utc.get(entsoe_hour_key, float(np.nan))
        # ENTSO-E: daily unavailability (by date in UTC)
        entsoe_date_key = dt_utc.strftime("%Y-%m-%d")
        ua = daily_unavail.get(entsoe_date_key, {})
        nuclear_unavail = ua.get("nuclear", float(np.nan)) if ua else float(np.nan)
        total_unavail   = ua.get("total", float(np.nan)) if ua else float(np.nan)

        records.append({
            "datetime_local":      dt_local,
            "datetime_key":        key,
            "epex_price_p_kwh":    epex_by_utc.get(utc_key, float(np.nan)),
            "is_peak":             is_peak,
            "temperature_2m":      w["temperature_2m"],
            "heating_dd":          max(0.0, 15.5 - w["temperature_2m"]),
            "shortwave_radiation": w["shortwave_radiation"],
            "precipitation":       w["precipitation"],
            "solar_gw":            solar_gw,
            "demand_mw":           demand_mw,
            "wind_gen_mw":         wind_gen_mw,
            "gas_gen_mw":          gas_gen_mw,
            "nuclear_mw":          nuclear_mw,
            "pumped_storage_mw":   pumped_storage_mw,
            "hydro_mw":            hydro_mw,
            "imports_mw":          imports_mw,
            "is_bank_holiday":     is_bh,
            "is_weekend":          is_wkend,
            "temp_x_wind":         w["temperature_2m"] * uk_avg_wind,
            "wind_x_solar":        uk_avg_wind * (solar_gw if not np.isnan(solar_gw) else 0.0),
            "brent_roll7":          d.get("brent_roll7"),
            "gas_ttf_roll7":        d.get("gas_ttf_roll7"),
            "gbpusd_roll7":         d.get("gbpusd_roll7"),
            "dxy_roll7":            d.get("dxy_roll7"),
            "carbon_roll7":         d.get("carbon_roll7"),
            "eu_gas_storage_pct":    d.get("eu_gas_storage_pct"),
            "gb_gas_storage_pct":    d.get("gb_gas_storage_pct"),
            "us_crude_stocks_delta": d.get("us_crude_stocks_delta"),
            "epex_lag7_gbp_mwh":     d.get("epex_lag7_gbp_mwh"),
            "epex_roll7_std":        d.get("epex_roll7_std"),
            "epex_roll7_min":        d.get("epex_roll7_min"),
            "epex_roll7_max":        d.get("epex_roll7_max"),
            "epex_momentum_7":       d.get("epex_momentum_7"),
            "wind_ramp_mw":          d.get("wind_ramp_mw"),
            "solar_ramp_gw":         d.get("solar_ramp_gw"),
            "sysprice_lag1_gbp_mwh": d.get("sysprice_lag1_gbp_mwh"),
            "abs_imbalance_lag1_mw": d.get("abs_imbalance_lag1_mw"),
            "price_lag1_slot":       price_lag1_slot,
            "net_scheduled_imports_mw": net_sched_imports,
            "nuclear_unavailable_mw":   nuclear_unavail,
            "total_unavailable_mw":     total_unavail,
            "net_residual_mw":     demand_mw - wind_gen_mw - (solar_gw * 1000 if not np.isnan(solar_gw) else 0.0) - nuclear_mw - hydro_mw,
            **wind_site_cols,
            **_time_features(dt_local),
        })

    return pd.DataFrame(records).dropna(
        subset=["temperature_2m", "shortwave_radiation", "epex_price_p_kwh"]
    ).reset_index(drop=True)
