"""
Tariff design: collapses half-hourly EPEX wholesale forecasts into
time-of-use retail tariff bands with network charges and margins.
"""
from datetime import date

import pandas as pd

from app.config import (DUOS_RATES, TNUOS_RATE, BSUOS_BUFFER,
                         SUPPLIER_MARGIN, SUPPLIER_MULTIPLIER, SUPPLIER_MARGIN_MODE)


def design_tariff(
    hh_forecast: pd.DataFrame,
    slots: str = "3",
    mae_buffer: float = 1.2,
    margin: float = SUPPLIER_MARGIN,
    margin_mode: str = SUPPLIER_MARGIN_MODE,
    multiplier: float = SUPPLIER_MULTIPLIER,
    duos_rates: dict | None = None,
    tnuos_rate: float | None = None,
    bsuos_buffer: float | None = None,
) -> pd.DataFrame:
    """
    Design a simple time-of-use tariff from the HH EPEX wholesale forecast.

    Collapses 48 half-hourly slots into 3 or 4 price bands and layers network
    charges, a MAE buffer, and a supplier margin on top of the wholesale cost.

    Slot boundaries
    ---------------
    3-slot: off-peak (23:00-07:00) | standard (07:00-16:00 + 19:00-23:00) | peak (16:00-19:00)
    4-slot: night (00:00-07:00) | day (07:00-16:00) | peak (16:00-19:00) | evening (19:00-23:00)

    Parameters
    ----------
    hh_forecast  : DataFrame with 'datetime_local' and 'predicted_epex_p_kwh' columns.
    slots        : "3" or "4"
    mae_buffer   : p/kWh -- added to every band to absorb forecast error (~1 MAE unit).
    margin       : p/kWh -- flat margin added per band (used when margin_mode="flat").
    margin_mode  : "flat" -- fixed p/kWh adder; "multiplier" -- wholesale scaled by `multiplier`.
    multiplier   : wholesale scaling factor (e.g. 2.1 like Octopus Agile).
                   Margin is implicit: margin_p_kwh = epex_mean x (multiplier - 1).
                   Used only when margin_mode="multiplier".
    duos_rates   : dict with keys "red", "amber", "green" (p/kWh). Defaults to config.DUOS_RATES.
    tnuos_rate   : float p/kWh. Defaults to config.TNUOS_RATE.
    bsuos_buffer : float p/kWh. Defaults to config.BSUOS_BUFFER.

    Returns
    -------
    DataFrame with columns:
        band, hours, epex_mean_p_kwh, duos_p_kwh, tnuos_p_kwh,
        bsuos_p_kwh, mae_buffer_p_kwh, margin_p_kwh, total_p_kwh
    """
    duos  = duos_rates   if duos_rates   is not None else DUOS_RATES
    tnuos = tnuos_rate   if tnuos_rate   is not None else TNUOS_RATE
    bsuos = bsuos_buffer if bsuos_buffer is not None else BSUOS_BUFFER

    df = hh_forecast.copy()
    df["_hour"] = df["datetime_local"].dt.hour

    if slots == "4":
        band_defs = [
            ("night",   "00:00\u201307:00", df["_hour"].between(0, 6),   "green"),
            ("day",     "07:00\u201316:00", df["_hour"].between(7, 15),  "amber"),
            ("peak",    "16:00\u201319:00", df["_hour"].between(16, 18), "red"),
            ("evening", "19:00\u201323:00", df["_hour"].between(19, 22), "amber"),
        ]
    else:  # 3-slot default
        band_defs = [
            ("off-peak", "23:00\u201307:00", (df["_hour"] >= 23) | (df["_hour"] <= 6), "green"),
            ("standard", "07:00\u201316:00 + 19:00\u201323:00",
             df["_hour"].between(7, 15) | df["_hour"].between(19, 22), "amber"),
            ("peak",     "16:00\u201319:00", df["_hour"].between(16, 18), "red"),
        ]

    rows = []
    for band, hours, mask, duos_band in band_defs:
        subset = df[mask]
        epex_mean = subset["predicted_epex_p_kwh"].mean() if not subset.empty else float("nan")
        duos_rate = duos.get(duos_band, 0.0)
        if margin_mode == "multiplier":
            # Wholesale is scaled; margin is implicit: EPEX x (mult - 1)
            margin_p_kwh = epex_mean * (multiplier - 1.0)
            total = epex_mean * multiplier + duos_rate + tnuos + bsuos + mae_buffer
        else:
            margin_p_kwh = margin
            total = epex_mean + duos_rate + tnuos + bsuos + mae_buffer + margin_p_kwh
        rows.append({
            "band":             band,
            "hours":            hours,
            "epex_mean_p_kwh":  round(epex_mean, 3),
            "duos_p_kwh":       duos_rate,
            "tnuos_p_kwh":      tnuos,
            "bsuos_p_kwh":      bsuos,
            "mae_buffer_p_kwh": mae_buffer,
            "margin_p_kwh":     round(margin_p_kwh, 3),
            "total_p_kwh":      round(total, 3),
        })

    return pd.DataFrame(rows)


def design_daily_tariffs(
    hh_forecast: pd.DataFrame,
    days: int = 3,
    slots: str = "3",
    mae_buffer: float = 1.2,
    margin: float = SUPPLIER_MARGIN,
    margin_mode: str = SUPPLIER_MARGIN_MODE,
    multiplier: float = SUPPLIER_MULTIPLIER,
    duos_rates: dict | None = None,
    tnuos_rate: float | None = None,
    bsuos_buffer: float | None = None,
) -> list[tuple[date, pd.DataFrame]]:
    """
    Run design_tariff() independently for each of the first `days` forecast days.

    Returns a list of (date, tariff_df) tuples, one per day, ordered chronologically.
    Restricting to the first 3 days reflects the reliability window of NWP forecasts.
    """
    dates = sorted(hh_forecast["datetime_local"].dt.date.unique())[:days]
    result = []
    for d in dates:
        day_df = hh_forecast[hh_forecast["datetime_local"].dt.date == d]
        t = design_tariff(
            day_df, slots=slots, mae_buffer=mae_buffer,
            margin=margin, margin_mode=margin_mode, multiplier=multiplier,
            duos_rates=duos_rates, tnuos_rate=tnuos_rate, bsuos_buffer=bsuos_buffer,
        )
        result.append((d, t))
    return result
