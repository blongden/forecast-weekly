"""
Correlation analysis, regression model, and price prediction.
Daily model: weather → daily avg ex-VAT price.
Half-hourly model: weather + time features → per-slot ex-VAT price.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app import db
from app.config import AGILE_VAT, AGILE_D, WIND_SITES

LOCAL_TZ = ZoneInfo("Europe/London")

# Base weather variables (used for correlations and scatter plots).
# UK averages across config.UK_WEATHER_SITES — no Edinburgh-only wind_speed_10m.
WEATHER_VARS = {
    "temperature_2m":      "Temperature (°C, UK avg)",
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
    "wind_gen_mw": "GB Wind Generation (MW)",
    "imports_mw":  "GB Net Interconnector Imports (MW)",
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
}

# Wind site features — one per offshore/onshore site
# uk_avg_wind is intentionally excluded to avoid multicollinearity.
# Ridge regularisation handles remaining inter-site correlations.
WIND_SITE_FEATURES = {f"wind_{sid}": info["label"] for sid, info in WIND_SITES.items()}

# Human-readable labels for ALL regression features
ALL_FEATURE_LABELS = {
    **WEATHER_VARS, **SOLAR_FEATURES, **DEMAND_FEATURES, **SUPPLY_FEATURES,
    **INTERACTION_FEATURES, **COMMODITY_FEATURES, **WIND_SITE_FEATURES,
}

# Ordered feature lists for each model.
# solar_gw replaces shortwave_radiation as the primary solar signal in the model.
# shortwave_radiation is kept in WEATHER_VARS for correlation display only.
DAILY_FEATURES = (["temperature_2m", "solar_gw", "demand_mw",
                   "wind_gen_mw", "imports_mw", "precipitation"] +
                  list(INTERACTION_FEATURES.keys()) +
                  list(COMMODITY_FEATURES.keys()) +
                  list(WIND_SITE_FEATURES.keys()))

HH_FEATURES = [
    "temperature_2m", "solar_gw", "demand_mw",
    "wind_gen_mw", "imports_mw", "precipitation",
    "temp_x_wind", "wind_x_solar",
    "gas_ttf_roll7", "brent_roll7",
    "wind_dogger_bank", "wind_hornsea", "wind_walney",
    "wind_whitelee", "wind_clyde_wind", "wind_pen_y_cymoedd",
    "is_peak", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


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
                                          "nuclear_mw", "imports_mw"])
    gen_df["date"] = pd.to_datetime(gen_df["date"])
    return pd.merge(df, gen_df[["date", "wind_gen_mw", "imports_mw"]], on="date", how="left")


def estimate_wind_gen_from_speed(df_historical: pd.DataFrame,
                                  wind_speed_series: pd.Series) -> pd.Series:
    """
    Estimate wind_gen_mw from uk_avg_wind using a Ridge linear model fitted on
    historical (wind_speed, wind_gen_mw) pairs.
    Used for the forecast horizon where actual generation data is unavailable.
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
    X = valid[["_uk_avg_wind"]].values
    y = valid["wind_gen_mw"].values
    scaler = StandardScaler()
    model  = Ridge(alpha=1.0)
    model.fit(scaler.fit_transform(X), y)
    X_fc = wind_speed_series.values.reshape(-1, 1)
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
        columns=["date", "brent_crude_usd", "gas_ttf_eur"],
    )
    if commodity.empty:
        return df  # commodity data not yet fetched; skip silently

    commodity["date"] = pd.to_datetime(commodity["date"])
    df = pd.merge(df, commodity, on="date", how="left")

    # Forward-fill gaps (weekends, public holidays) up to 3 days
    df["brent_crude_usd"] = df["brent_crude_usd"].ffill(limit=3)
    df["gas_ttf_eur"]     = df["gas_ttf_eur"].ffill(limit=3)

    # 7-day rolling average to smooth short-term noise and capture lag
    df["brent_roll7"]   = df["brent_crude_usd"].rolling(7, min_periods=3).mean()
    df["gas_ttf_roll7"] = df["gas_ttf_eur"].rolling(7, min_periods=3).mean()

    return df


def load_daily_df(date_from: date, date_to: date) -> pd.DataFrame:
    """Load and merge daily prices + UK-average weather (+ commodity prices if available)."""
    prices  = pd.DataFrame(db.get_daily_prices(date_from, date_to),
                           columns=["date", "avg_price_inc_vat",
                                    "avg_price_ex_vat", "avg_wholesale_price"])
    weather = pd.DataFrame(db.get_daily_uk_avg(date_from, date_to),
                           columns=["date", "temperature_2m",
                                    "shortwave_radiation", "precipitation"])

    df = pd.merge(prices, weather, on="date", how="inner").dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = _add_commodity_features(df, date_from, date_to)
    df = _add_wind_site_features(df, date_from, date_to)
    df = _add_solar_features(df, date_from, date_to)
    df = _add_demand_features(df, date_from, date_to)
    df = _add_supply_features(df, date_from, date_to)
    # Interaction terms using mean of wind-farm sites as the wind signal
    wind_site_cols = [c for c in df.columns if c.startswith("wind_")]
    uk_avg_wind = df[wind_site_cols].mean(axis=1) if wind_site_cols else pd.Series(0.0, index=df.index)
    solar_signal = df["solar_gw"] if "solar_gw" in df.columns else df.get("shortwave_radiation", pd.Series(0.0, index=df.index))
    df["temp_x_wind"]  = df["temperature_2m"] * uk_avg_wind
    df["wind_x_solar"] = uk_avg_wind * solar_signal
    return df.sort_values("date").reset_index(drop=True)


def compute_correlations(df: pd.DataFrame) -> dict:
    """Return dict of {var: {r, p, label}} Pearson correlations with ex-VAT price."""
    results = {}
    vars_to_correlate = {**WEATHER_VARS, **SOLAR_FEATURES, **DEMAND_FEATURES, **SUPPLY_FEATURES}
    for var, label in vars_to_correlate.items():
        if var not in df.columns or df[var].isna().all():
            continue
        valid = df[["avg_price_ex_vat", var]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.pearsonr(valid[var], valid["avg_price_ex_vat"])
        results[var] = {"r": r, "p": p, "label": label}
    return results


def fit_model(df: pd.DataFrame):
    """
    Fit a multiple linear regression of weather → ex-VAT Agile price
    (network charges retained; only VAT stripped).
    Returns (model, scaler, r2, feature_cols).
    """
    # Include features present with at least some data; dropna below removes sparse rows
    feature_cols = [c for c in DAILY_FEATURES if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols)
    X = df_model[feature_cols].values
    y = df_model["avg_price_ex_vat"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred  = model.predict(X_scaled)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot

    return model, scaler, r2, feature_cols


def predict_from_forecast(forecast_df: pd.DataFrame,
                          model: Ridge,
                          scaler: StandardScaler,
                          feature_cols: list[str],
                          latest_commodity: dict | None = None,
                          site_forecasts: dict | None = None,
                          df_historical: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply trained model to forecast weather.
    latest_commodity: dict of {feature_col: value} for commodity rolling averages —
    the most recent known values are held constant across the forecast horizon.
    site_forecasts: dict of {site_id: DataFrame} with wind speed forecasts per site.
    """
    from app.config import AGILE_D, AGILE_VAT

    fc = forecast_df.copy()
    for col in COMMODITY_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)

    # Join daily avg wind from each site forecast
    if site_forecasts:
        for sid, sdf in site_forecasts.items():
            col = f"wind_{sid}"
            if col in feature_cols:
                sdf_daily = sdf.copy()
                sdf_daily["date"] = pd.to_datetime(sdf_daily["datetime"].dt.date)
                daily_avg = sdf_daily.groupby("date")["wind_speed"].mean().reset_index()
                daily_avg.columns = ["date", col]
                fc = pd.merge(fc, daily_avg, on="date", how="left")

    # Estimate solar_gw from shortwave_radiation only when actual values are absent
    # (forecast path).  Backtests and any caller that already populates solar_gw
    # will have real data present — don't overwrite it.
    if "solar_gw" in feature_cols:
        if "solar_gw" not in fc.columns or fc["solar_gw"].isna().all():
            fc["solar_gw"] = estimate_solar_from_radiation(
                df_historical if df_historical is not None else pd.DataFrame(),
                fc["shortwave_radiation"],
            )

    # Estimate demand_mw from historical day-of-week averages when absent (forecast path)
    if "demand_mw" in feature_cols:
        if "demand_mw" not in fc.columns or fc["demand_mw"].isna().all():
            if df_historical is not None and "demand_mw" in df_historical.columns:
                hist = df_historical.dropna(subset=["demand_mw"])
                dow_profile = hist.groupby(hist["date"].dt.dayofweek)["demand_mw"].mean().to_dict()
            else:
                dow_profile = {}
            overall_mean = np.nanmean(list(dow_profile.values())) if dow_profile else np.nan
            fc["demand_mw"] = fc["date"].apply(
                lambda d: dow_profile.get(d.dayofweek, overall_mean)
            )

    # Estimate wind_gen_mw for forecast from wind speed (linear model on history)
    if "wind_gen_mw" in feature_cols:
        if "wind_gen_mw" not in fc.columns or fc["wind_gen_mw"].isna().all():
            if df_historical is not None:
                wind_cols = [c for c in df_historical.columns if c.startswith("wind_")]
                if wind_cols:
                    hist_wind = df_historical.copy()
                    hist_wind["_uk_avg_wind"] = hist_wind[wind_cols].mean(axis=1)
                    # Build forecast wind speed from site forecasts if available
                    wind_site_cols_fc = [c for c in fc.columns if c.startswith("wind_")]
                    if wind_site_cols_fc:
                        fc_wind = fc[wind_site_cols_fc].mean(axis=1)
                    else:
                        fc_wind = pd.Series(0.0, index=fc.index)
                    fc["wind_gen_mw"] = estimate_wind_gen_from_speed(hist_wind, fc_wind)
                else:
                    fc["wind_gen_mw"] = np.nan

    # Estimate imports_mw for forecast from historical day-of-week profile
    if "imports_mw" in feature_cols:
        if "imports_mw" not in fc.columns or fc["imports_mw"].isna().all():
            if df_historical is not None and "imports_mw" in df_historical.columns:
                hist = df_historical.dropna(subset=["imports_mw"])
                dow_profile = hist.groupby(hist["date"].dt.dayofweek)["imports_mw"].mean().to_dict()
            else:
                dow_profile = {}
            overall_mean = np.nanmean(list(dow_profile.values())) if dow_profile else 0.0
            fc["imports_mw"] = fc["date"].apply(
                lambda d: dow_profile.get(d.dayofweek, overall_mean)
            )

    # Interaction terms using mean of wind-farm sites as the wind signal
    wind_site_cols = [c for c in fc.columns if c.startswith("wind_") and c in feature_cols]
    uk_avg_wind = fc[wind_site_cols].mean(axis=1) if wind_site_cols else pd.Series(0.0, index=fc.index)
    solar_signal = fc["solar_gw"] if "solar_gw" in fc.columns else fc.get("shortwave_radiation", pd.Series(0.0, index=fc.index))
    fc["temp_x_wind"]  = fc["temperature_2m"] * uk_avg_wind
    fc["wind_x_solar"] = uk_avg_wind * solar_signal.fillna(0.0)

    X_fc = fc[feature_cols].values
    preds = model.predict(scaler.transform(X_fc))

    result = forecast_df[["date"]].copy()
    result["predicted_ex_vat_p_kwh"]  = preds
    result["predicted_inc_vat_p_kwh"] = preds * AGILE_VAT
    # Indicative wholesale: ex-VAT ÷ D (daily avg; peak premium averages out)
    result["predicted_wholesale_p_kwh"] = preds / AGILE_D
    return result


# ── Half-hourly model ──────────────────────────────────────────────────────────

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
    price_rows   = db.get_halfhourly_prices(date_from, date_to)
    weather_rows = db.get_hourly_uk_avg(date_from, date_to)

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

    # Build 30-min generation lookup (wind, imports)
    gen_rows = db.get_halfhourly_generation(date_from, date_to)
    wind_gen_by_utc: dict[str, float] = {}
    imports_by_utc:  dict[str, float] = {}
    for row in gen_rows:
        dt_key = row["datetime_utc"]
        if row["wind_mw"]    is not None: wind_gen_by_utc[dt_key] = row["wind_mw"]
        if row["imports_mw"] is not None: imports_by_utc[dt_key]  = row["imports_mw"]

    # Build commodity + wind-site lookup keyed by date
    _temp_daily = load_daily_df(date_from, date_to)
    daily_by_date: dict[str, dict] = {}
    for _, row in _temp_daily.iterrows():
        dk = str(row["date"].date())
        daily_by_date[dk] = {
            "brent_roll7":   row.get("brent_roll7"),
            "gas_ttf_roll7": row.get("gas_ttf_roll7"),
        }
        for col in _temp_daily.columns:
            if col.startswith("wind_"):
                daily_by_date[dk][col] = row.get(col)

    records = []
    for row in price_rows:
        dt_utc   = datetime.fromisoformat(row["datetime"].replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(LOCAL_TZ)
        key      = dt_local.strftime("%Y-%m-%dT%H:00")
        w        = weather.get(key)
        if w is None:
            continue
        date_key = dt_local.strftime("%Y-%m-%d")
        d = daily_by_date.get(date_key, {})
        wind_vals = [d[k] for k in d if k.startswith("wind_") and d[k] is not None]
        uk_avg_wind = float(np.mean(wind_vals)) if wind_vals else 0.0
        # Solar generation: PVLIVE uses period-END timestamps (e.g. "00:30" = slot 00:00-00:30),
        # Octopus uses period-START ("00:00" = same slot), so shift +30 min to match.
        pvlive_key = (dt_utc + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
        solar_gw = solar_by_gmt.get(pvlive_key)
        if solar_gw is None:
            solar_gw = float(np.nan)
        # Demand and generation: UTC period-start matches Octopus convention directly
        utc_key    = dt_utc.strftime("%Y-%m-%dT%H:%M:%S")
        demand_mw  = demand_by_utc.get(utc_key, float(np.nan))
        wind_gen_mw = wind_gen_by_utc.get(utc_key, float(np.nan))
        imports_mw  = imports_by_utc.get(utc_key, float(np.nan))
        records.append({
            "datetime_local":      dt_local,
            "datetime_key":        key,
            "price_ex_vat":        row["price_ex_vat"],
            "price_inc_vat":       row["price_inc_vat"],
            "is_peak":             row["is_peak"],
            "temperature_2m":      w["temperature_2m"],
            "shortwave_radiation": w["shortwave_radiation"],
            "precipitation":       w["precipitation"],
            "solar_gw":            solar_gw,
            "demand_mw":           demand_mw,
            "wind_gen_mw":         wind_gen_mw,
            "imports_mw":          imports_mw,
            "temp_x_wind":         w["temperature_2m"] * uk_avg_wind,
            "wind_x_solar":        uk_avg_wind * (solar_gw if not np.isnan(solar_gw) else 0.0),
            "brent_roll7":         d.get("brent_roll7"),
            "gas_ttf_roll7":       d.get("gas_ttf_roll7"),
            **{k: d.get(k) for k in d if k.startswith("wind_")},
            **_time_features(dt_local),
        })

    return pd.DataFrame(records).dropna(
        subset=["temperature_2m", "shortwave_radiation", "price_ex_vat"]
    ).reset_index(drop=True)


def fit_halfhourly_model(df: pd.DataFrame):
    """
    Train a LinearRegression on half-hourly data.
    Returns (model, scaler, r2).
    """
    feature_cols = [c for c in HH_FEATURES if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols)
    X = df_model[feature_cols].values
    y = df_model["price_ex_vat"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot

    return model, scaler, r2, feature_cols


def predict_halfhourly_forecast(
    forecast_hourly_df: pd.DataFrame,
    model: Ridge,
    scaler: StandardScaler,
    feature_cols: list[str] | None = None,
    latest_commodity: dict | None = None,
    site_forecasts: dict | None = None,
    df_historical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Expand hourly forecast to half-hourly slots and predict ex-VAT price.
    The forecast_hourly_df datetimes are local (Europe/London) timestamps.
    site_forecasts: dict of {site_id: DataFrame} with 'datetime' and 'wind_speed'.
    df_historical: historical df used to fit the solar_gw estimator.
    """
    # Build per-hour wind lookup for each site
    site_wind_by_hour: dict[str, dict] = {}  # {site_id: {"YYYY-MM-DDTHH:00": wind}}
    if site_forecasts:
        for sid, sdf in site_forecasts.items():
            site_wind_by_hour[sid] = {
                r["datetime"].strftime("%Y-%m-%dT%H:00"): r["wind_speed"]
                for _, r in sdf.iterrows()
            }

    # Pre-compute estimated solar_gw from shortwave_radiation
    cols_needed  = feature_cols if feature_cols is not None else HH_FEATURES
    need_solar   = "solar_gw"    in cols_needed
    need_demand  = "demand_mw"   in cols_needed
    need_wind_gen = "wind_gen_mw" in cols_needed
    need_imports  = "imports_mw"  in cols_needed
    solar_est_by_hour: dict[str, float] = {}
    if need_solar and df_historical is not None and "shortwave_radiation" in forecast_hourly_df.columns:
        est_series = estimate_solar_from_radiation(
            df_historical, forecast_hourly_df["shortwave_radiation"].reset_index(drop=True)
        )
        for i, row in forecast_hourly_df.reset_index(drop=True).iterrows():
            solar_est_by_hour[row["datetime"].strftime("%Y-%m-%dT%H:00")] = float(est_series.iloc[i])

    # Pre-compute demand estimates from historical (day_of_week, hour) profile
    demand_profile: dict = {}
    if need_demand and df_historical is not None:
        demand_profile = build_demand_profile(df_historical)

    # Pre-compute wind generation estimates from wind speed (linear model)
    wind_gen_est_by_hour: dict[str, float] = {}
    if need_wind_gen and df_historical is not None and site_forecasts:
        # Build forecast uk_avg_wind series (one value per forecast hour)
        site_wind_speeds = []
        for sid, sdf in site_forecasts.items():
            s = {r["datetime"].strftime("%Y-%m-%dT%H:00"): r["wind_speed"]
                 for _, r in sdf.iterrows()}
            site_wind_speeds.append(s)
        fc_wind = pd.Series([
            np.nanmean([lkp.get(row["datetime"].strftime("%Y-%m-%dT%H:00"), np.nan)
                        for lkp in site_wind_speeds])
            for _, row in forecast_hourly_df.iterrows()
        ])
        wind_gen_series = estimate_wind_gen_from_speed(df_historical, fc_wind)
        for i, row in forecast_hourly_df.reset_index(drop=True).iterrows():
            wind_gen_est_by_hour[row["datetime"].strftime("%Y-%m-%dT%H:00")] = float(wind_gen_series.iloc[i])

    # Pre-compute imports estimates from historical (day_of_week, hour) profile
    imports_profile: dict = {}
    if need_imports and df_historical is not None:
        imports_profile = build_imports_profile(df_historical)

    records = []
    for _, row in forecast_hourly_df.iterrows():
        hour_key = row["datetime"].strftime("%Y-%m-%dT%H:00")
        for minute in (0, 30):
            dt_local = row["datetime"].replace(minute=minute, second=0, microsecond=0)
            is_peak  = 1 if 16 <= dt_local.hour < 19 else 0
            site_winds = []
            site_wind_vals = {}
            for sid, lookup in site_wind_by_hour.items():
                w = lookup.get(hour_key)
                site_wind_vals[f"wind_{sid}"] = w
                if w is not None:
                    site_winds.append(w)
            uk_avg_wind = float(np.mean(site_winds)) if site_winds else 0.0
            solar_gw = solar_est_by_hour.get(hour_key, 0.0) if need_solar else None
            solar_signal = solar_gw if solar_gw is not None else row.get("shortwave_radiation", 0.0)
            dow  = dt_local.dayofweek if hasattr(dt_local, "dayofweek") else dt_local.weekday()
            hour = dt_local.hour
            demand_mw    = demand_profile.get((dow, hour), np.nan) if need_demand else None
            wind_gen_mw  = wind_gen_est_by_hour.get(hour_key, np.nan) if need_wind_gen else None
            imports_mw   = imports_profile.get((dow, hour), np.nan) if need_imports else None
            rec = {
                "datetime_local":      dt_local,
                "is_peak":             is_peak,
                "temperature_2m":      row["temperature_2m"],
                "shortwave_radiation": row["shortwave_radiation"],
                "precipitation":       row["precipitation"],
                "solar_gw":            solar_gw,
                "demand_mw":           demand_mw,
                "wind_gen_mw":         wind_gen_mw,
                "imports_mw":          imports_mw,
                "temp_x_wind":         row["temperature_2m"] * uk_avg_wind,
                "wind_x_solar":        uk_avg_wind * (solar_signal or 0.0),
                "brent_roll7":         (latest_commodity or {}).get("brent_roll7"),
                "gas_ttf_roll7":       (latest_commodity or {}).get("gas_ttf_roll7"),
                **site_wind_vals,
                **_time_features(dt_local),
            }
            records.append(rec)

    df = pd.DataFrame(records)
    cols = feature_cols if feature_cols is not None else HH_FEATURES
    cols = [c for c in cols if c in df.columns]
    X  = df[cols].values
    df["predicted_ex_vat"]  = model.predict(scaler.transform(X))
    df["predicted_inc_vat"] = df["predicted_ex_vat"] * AGILE_VAT
    return df


def run_backtest(df: pd.DataFrame, holdout_days: int = 30) -> tuple:
    """
    Hold-out backtest: train on data up to `holdout_days` ago, predict the
    hold-out period using actual weather (perfect-forecast proxy), compare vs actuals.

    Returns (comparison_df, metrics_dict).
    comparison_df columns: date, actual_ex_vat, predicted_ex_vat, error

    This is the best achievable accuracy given our weather inputs — actual
    weather is used instead of a forecast so it removes forecast error,
    showing the ceiling of the weather→price relationship.
    """
    if len(df) < holdout_days + 30:
        return pd.DataFrame(), {}

    train = df.iloc[:-holdout_days].copy()
    test  = df.iloc[-holdout_days:].copy()

    model_bt, scaler_bt, r2_train, fcols_bt = fit_model(train)

    # Latest commodity from training data only (as would be available on cutoff date)
    latest_commodity_bt = {
        col: train[col].dropna().iloc[-1]
        if col in train.columns and train[col].notna().any() else None
        for col in COMMODITY_FEATURES
    }

    # Build test feature set — include all columns the model needs
    base_cols = ["date", "temperature_2m", "shortwave_radiation", "precipitation"]
    extra_cols = [c for c in fcols_bt if c in test.columns and c not in base_cols]
    test_fc = test[base_cols + extra_cols].copy()

    preds = predict_from_forecast(test_fc, model_bt, scaler_bt, fcols_bt,
                                  latest_commodity_bt)

    result = pd.merge(
        test[["date", "avg_price_ex_vat"]],
        preds[["date", "predicted_ex_vat_p_kwh"]],
        on="date",
    )
    result["error"] = result["predicted_ex_vat_p_kwh"] - result["avg_price_ex_vat"]

    mae  = result["error"].abs().mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    # Mean absolute percentage error
    mape = (result["error"].abs() / result["avg_price_ex_vat"].abs()).mean() * 100

    metrics = {
        "r2_train":    r2_train,
        "mae":         mae,
        "rmse":        rmse,
        "mape":        mape,
        "holdout_days": holdout_days,
        "train_days":  len(train),
    }
    return result, metrics


def run_halfhourly_backtest(df_hh: pd.DataFrame, holdout_days: int = 30) -> tuple:
    """
    Hold-out backtest on the half-hourly model.
    Trains on slots before the cutoff, tests on the following `holdout_days`.
    Returns (comparison_df, metrics_dict).
    comparison_df columns: datetime_local, actual, predicted, error, is_peak
    """
    if df_hh.empty:
        return pd.DataFrame(), {}

    cutoff = df_hh["datetime_local"].max() - pd.Timedelta(days=holdout_days)
    train  = df_hh[df_hh["datetime_local"] < cutoff].copy()
    test   = df_hh[df_hh["datetime_local"] >= cutoff].copy()

    if len(train) < 1000 or len(test) < 100:
        return pd.DataFrame(), {}

    model_bt, scaler_bt, r2_train, fcols_bt = fit_halfhourly_model(train)

    test_clean = test.dropna(subset=fcols_bt)
    X_test = test_clean[fcols_bt].values
    test_clean = test_clean.copy()
    test_clean["predicted"] = model_bt.predict(scaler_bt.transform(X_test))

    result = test_clean[["datetime_local", "price_ex_vat", "predicted", "is_peak"]].copy()
    result.columns = ["datetime_local", "actual", "predicted", "is_peak"]
    result["error"] = result["predicted"] - result["actual"]

    mae  = result["error"].abs().mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    mape = (result["error"].abs() / result["actual"].abs()).mean() * 100

    # Peak vs off-peak accuracy split
    peak_mae    = result[result["is_peak"] == 1]["error"].abs().mean()
    offpeak_mae = result[result["is_peak"] == 0]["error"].abs().mean()

    metrics = {
        "r2_train":     r2_train,
        "mae":          mae,
        "rmse":         rmse,
        "mape":         mape,
        "peak_mae":     peak_mae,
        "offpeak_mae":  offpeak_mae,
        "holdout_days": holdout_days,
    }
    return result, metrics


def print_summary(df: pd.DataFrame, correlations: dict, r2: float,
                  model, scaler, feature_cols: list[str],
                  predictions: pd.DataFrame, r2_hh: float | None = None) -> None:
    """Print a formatted summary to the terminal."""
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Date range        : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Days              : {len(df)}")
    print(f"  Mean ex-VAT price : {df['avg_price_ex_vat'].mean():.2f} p/kWh  "
          f"(network charges retained)")
    print(f"  Mean inc-VAT price: {df['avg_price_inc_vat'].mean():.2f} p/kWh")
    print(f"  Mean wholesale    : {df['avg_wholesale_price'].mean():.2f} p/kWh  "
          f"(EPEX estimate, for reference)")
    print(f"  Ex-VAT range      : {df['avg_price_ex_vat'].min():.2f} – "
          f"{df['avg_price_ex_vat'].max():.2f} p/kWh")

    print("\n" + "=" * 65)
    print("PEARSON CORRELATIONS  (ex-VAT Agile price vs weather)")
    print("=" * 65)
    for var, info in sorted(correlations.items(),
                            key=lambda x: abs(x[1]["r"]), reverse=True):
        sig = ("***" if info["p"] < 0.001 else
               ("**"  if info["p"] < 0.01  else
                ("*"   if info["p"] < 0.05  else "")))
        print(f"  {info['label']:<30s}  r = {info['r']:+.4f}   "
              f"p = {info['p']:.2e}  {sig}")

    print(f"\n  Daily model R²       = {r2:.4f}  ({r2*100:.1f}% — in-sample)")
    if r2_hh is not None:
        print(f"  Half-hourly model R² = {r2_hh:.4f}  ({r2_hh*100:.1f}% — in-sample, incl. time features)")
    print("  Daily model coefficients (standardised):")
    for col, coef in zip(feature_cols, model.coef_):
        print(f"    {ALL_FEATURE_LABELS[col]:<32s}  β = {coef:+.4f}")

    n_days = len(predictions)
    print("\n" + "=" * 65)
    print(f"{n_days}-DAY PRICE PREDICTIONS  (daily model, weather-only)")
    print("=" * 65)
    for _, row in predictions.iterrows():
        ds = row["date"].strftime("%A %d %b %Y")
        print(f"  {ds:<25s}  "
              f"ex-VAT {row['predicted_ex_vat_p_kwh']:+.2f} p/kWh  "
              f"/ inc-VAT {row['predicted_inc_vat_p_kwh']:+.2f} p/kWh  "
              f"/ wholesale ~{row['predicted_wholesale_p_kwh']:+.2f} p/kWh")
    print("=" * 65)
