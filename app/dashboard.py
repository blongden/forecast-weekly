"""
HTML dashboard generation using Plotly.
Produces a single self-contained index.html file.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os

from app.config import BASE_DIR
from app.features import ALL_FEATURE_LABELS

DASHBOARD_PATH = Path(os.environ.get("DASHBOARD_PATH", str(BASE_DIR / "index.html")))

_TEMPLATE     = "plotly_white"
_PRICE_COL    = "#e74c3c"
_WHOLESALE    = "#c0392b"
_PEAK_COL     = "rgba(255,200,100,0.18)"
_FORECAST_COL = "#2980b9"
_DEMAND_COL   = "#8e44ad"
_SOLAR_COL    = "#f39c12"
_COLOURS      = ["#8e44ad", "#e74c3c", "#f39c12", "#27ae60"]


# ── Individual chart builders ──────────────────────────────────────────────────

def _fig_halfhourly_forecast(hh_pred: pd.DataFrame, hist_mean: float,
                              daily_tariffs_3: list | None = None,
                              mae_band: float | None = None) -> go.Figure:
    """7-day half-hourly forecast as a line chart with peak shading.

    mae_band: if provided, draws a ±MAE uncertainty envelope around the forecast.
    """
    # Band colours for tariff overlay (off-peak, standard, peak)
    _BAND_COLOURS = {"off-peak": "#27ae60", "standard": "#f39c12", "peak": "#e74c3c",
                     "night": "#27ae60", "day": "#f39c12", "evening": "#f39c12"}

    fig = go.Figure()

    dates = hh_pred["datetime_local"].dt.date.unique()
    for d in dates:
        fig.add_vrect(
            x0=f"{d}T16:00", x1=f"{d}T19:00",
            fillcolor=_PEAK_COL, line_width=0,
            annotation_text="Peak" if d == dates[0] else "",
            annotation_position="top left",
            annotation_font_size=9,
        )

    # ── Uncertainty band (±MAE) ───────────────────────────────────────────────
    if mae_band is not None:
        xs = hh_pred["datetime_local"]
        ys = hh_pred["predicted_epex_p_kwh"]
        fig.add_trace(go.Scatter(
            x=xs, y=ys - mae_band,
            mode="lines", line=dict(width=0, color=_FORECAST_COL),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys + mae_band,
            mode="lines", line=dict(width=0, color=_FORECAST_COL),
            fill="tonexty", fillcolor="rgba(41,128,185,0.12)",
            name=f"±{mae_band:.2f}p MAE band",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Split actual (D+1) vs forecast (D+2+) traces
    _ACTUAL_COL = "#27ae60"  # green for settled prices
    has_actual = "is_actual" in hh_pred.columns and hh_pred["is_actual"].any()
    if has_actual:
        actual_mask = hh_pred["is_actual"].astype(bool)
        hh_actual = hh_pred[actual_mask]
        hh_forecast = hh_pred[~actual_mask]

        fig.add_trace(go.Scatter(
            x=hh_actual["datetime_local"], y=hh_actual["predicted_epex_p_kwh"],
            mode="lines", name="D+1 Actual (EPEX settled)",
            line=dict(color=_ACTUAL_COL, width=2),
            hovertemplate="%{x|%a %d %b %H:%M}<br>Actual: %{y:.2f}p/kWh<extra></extra>",
        ))
        if not hh_forecast.empty:
            fig.add_trace(go.Scatter(
                x=hh_forecast["datetime_local"], y=hh_forecast["predicted_epex_p_kwh"],
                mode="lines", name="D+2+ Forecast (model)",
                line=dict(color=_FORECAST_COL, width=1.5),
                hovertemplate="%{x|%a %d %b %H:%M}<br>Forecast: %{y:.2f}p/kWh<extra></extra>",
            ))
        # Add boundary annotation
        if not hh_actual.empty and not hh_forecast.empty:
            boundary_x = hh_actual["datetime_local"].max()
            fig.add_vline(
                x=boundary_x, line_dash="dot", line_color="grey", line_width=1,
                annotation_text="Actual → Forecast",
                annotation_position="top",
                annotation_font_size=9,
            )
    else:
        fig.add_trace(go.Scatter(
            x=hh_pred["datetime_local"], y=hh_pred["predicted_epex_p_kwh"],
            mode="lines", name="Predicted wholesale (EPEX)",
            line=dict(color=_FORECAST_COL, width=1.5),
            hovertemplate="%{x|%a %d %b %H:%M}<br>Wholesale: %{y:.2f}p/kWh<extra></extra>",
        ))

    # ── Tariff overlay: step line per band per day (first 3 days only) ────────
    if daily_tariffs_3:
        # Collect (x_start, x_end, price) segments per band across days
        band_segments: dict[str, list] = {}  # band → [(x0, x1, price), …]

        for d, tariff in daily_tariffs_3:
            day_slots = hh_pred[hh_pred["datetime_local"].dt.date == d].copy()
            if day_slots.empty:
                continue
            hour_arr = day_slots["datetime_local"].dt.hour

            for _, trow in tariff.iterrows():
                band = trow["band"]
                total = trow["total_p_kwh"]
                # Determine which slots belong to this band
                if band == "off-peak":
                    mask = (hour_arr >= 23) | (hour_arr <= 6)
                elif band == "standard":
                    mask = hour_arr.between(7, 15) | hour_arr.between(19, 22)
                elif band == "peak":
                    mask = hour_arr.between(16, 18)
                elif band == "night":
                    mask = hour_arr.between(0, 6)
                elif band == "day":
                    mask = hour_arr.between(7, 15)
                elif band == "evening":
                    mask = hour_arr.between(19, 22)
                else:
                    continue
                slots = day_slots[mask]["datetime_local"]
                if slots.empty:
                    continue
                band_segments.setdefault(band, []).append(
                    (slots.min(), slots.max(), total)
                )

        # One trace per band (with gaps between days)
        shown_bands: set[str] = set()
        for band, segs in band_segments.items():
            colour = _BAND_COLOURS.get(band, "#95a5a6")
            # Build x/y with NaN gaps between days
            xs, ys = [], []
            for x0, x1, price in segs:
                xs += [x0, x1, None]
                ys += [price, price, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                name=f"Tariff {band}" if band not in shown_bands else band,
                showlegend=(band not in shown_bands),
                line=dict(color=colour, width=2, dash="dot"),
                connectgaps=False,
                hovertemplate=f"Tariff ({band}): %{{y:.2f}}p/kWh<extra></extra>",
            ))
            shown_bands.add(band)

    if hist_mean is not None:
        fig.add_hline(
            y=hist_mean, line_dash="dash", line_color="grey", line_width=1,
            annotation_text=f"7-day forecast avg {hist_mean:.2f}p",
            annotation_position="bottom right",
            annotation_font_size=10,
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="7-Day Half-Hourly Wholesale Forecast  (shaded = 16:00–19:00 peak period)",
        yaxis_title="Price (p/kWh)",
        yaxis=dict(rangemode="tozero"),
        xaxis_title="",
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
        height=460,
        margin=dict(t=90, b=40),
    )
    return fig


def _fig_predicted_vs_actual(predicted_on: str) -> go.Figure | None:
    """Overlay stored HH predictions against actual EPEX prices.

    Shows: actual prices (solid green) where available, predicted prices
    (dashed blue) for comparison on the same slots, then the future forecast
    (solid blue) where no actuals exist yet.  Returns None if no data.
    """
    from app import db as _db
    from zoneinfo import ZoneInfo as _ZI
    _london = _ZI("Europe/London")

    with _db.get_conn() as conn:
        pred_rows = conn.execute(
            """SELECT datetime_utc, predicted_epex_p_kwh, is_actual
               FROM halfhourly_predictions
               WHERE predicted_on = ?
               ORDER BY datetime_utc""",
            (predicted_on,),
        ).fetchall()
    if not pred_rows:
        return None

    # Build prediction DataFrame
    pred_data = []
    for r in pred_rows:
        ts = pd.Timestamp(r[0])
        dt_utc = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        dt_local = dt_utc.tz_convert(_london)
        pred_data.append({
            "datetime_local": dt_local,
            "datetime_utc": r[0],
            "predicted_p_kwh": r[1],
            "is_actual_pred": bool(r[2]) if r[2] is not None else False,
        })
    pdf = pd.DataFrame(pred_data)

    # Get actual EPEX prices for the date range covered by predictions
    date_min = pdf["datetime_local"].dt.date.min()
    date_max = pdf["datetime_local"].dt.date.max()
    from datetime import date as _date
    actual_rows = _db.get_halfhourly_midprice(_date.fromisoformat(str(date_min)),
                                               _date.fromisoformat(str(date_max)))
    actual_map = {}
    for r in actual_rows:
        ts = pd.Timestamp(r[0])
        dt_utc = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        dt_local = dt_utc.tz_convert(_london)
        actual_map[dt_local] = r[1] * 0.1  # £/MWh → p/kWh

    pdf["actual_p_kwh"] = pdf["datetime_local"].map(actual_map)
    has_actuals = pdf["actual_p_kwh"].notna().any()

    fig = go.Figure()

    # Peak shading
    dates = pdf["datetime_local"].dt.date.unique()
    for d in dates:
        fig.add_vrect(
            x0=f"{d}T16:00", x1=f"{d}T19:00",
            fillcolor=_PEAK_COL, line_width=0,
            annotation_text="Peak" if d == dates[0] else "",
            annotation_position="top left",
            annotation_font_size=9,
        )

    if has_actuals:
        # Split into slots with and without actuals
        with_actual = pdf[pdf["actual_p_kwh"].notna()]
        without_actual = pdf[pdf["actual_p_kwh"].isna()]

        # Actual prices — solid green
        fig.add_trace(go.Scatter(
            x=with_actual["datetime_local"], y=with_actual["actual_p_kwh"],
            mode="lines", name="Actual EPEX price",
            line=dict(color="#27ae60", width=2.5),
            hovertemplate="%{x|%a %d %b %H:%M}<br>Actual: %{y:.2f}p/kWh<extra></extra>",
        ))

        # Prediction on same slots — dashed blue (shows the error)
        fig.add_trace(go.Scatter(
            x=with_actual["datetime_local"], y=with_actual["predicted_p_kwh"],
            mode="lines", name="What we predicted",
            line=dict(color=_FORECAST_COL, width=1.5, dash="dash"),
            hovertemplate="%{x|%a %d %b %H:%M}<br>Predicted: %{y:.2f}p/kWh<extra></extra>",
        ))

        # Error shading between predicted and actual
        fig.add_trace(go.Scatter(
            x=pd.concat([with_actual["datetime_local"], with_actual["datetime_local"][::-1]]),
            y=pd.concat([with_actual["actual_p_kwh"], with_actual["predicted_p_kwh"][::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Boundary annotation
        if not without_actual.empty:
            boundary_x = with_actual["datetime_local"].max()
            fig.add_shape(
                type="line",
                x0=str(boundary_x), x1=str(boundary_x),
                y0=0, y1=1, yref="paper",
                line=dict(dash="dot", color="grey", width=1),
            )
            fig.add_annotation(
                x=str(boundary_x), y=1, yref="paper",
                text="Now → Forecast", showarrow=False,
                font=dict(size=9),
            )

        # Future forecast — solid blue
        if not without_actual.empty:
            # Connect to last actual point for visual continuity
            bridge = with_actual.iloc[-1:]
            future = pd.concat([bridge, without_actual], ignore_index=True)
            fig.add_trace(go.Scatter(
                x=future["datetime_local"], y=future["predicted_p_kwh"],
                mode="lines", name="Forecast (model)",
                line=dict(color=_FORECAST_COL, width=1.5),
                hovertemplate="%{x|%a %d %b %H:%M}<br>Forecast: %{y:.2f}p/kWh<extra></extra>",
            ))

        # Compute MAE for the overlap
        mae = (with_actual["actual_p_kwh"] - with_actual["predicted_p_kwh"]).abs().mean()
        title_suffix = f"  (MAE on actuals so far: {mae:.2f}p/kWh)"
    else:
        # No actuals yet — just show prediction
        fig.add_trace(go.Scatter(
            x=pdf["datetime_local"], y=pdf["predicted_p_kwh"],
            mode="lines", name="Forecast (model)",
            line=dict(color=_FORECAST_COL, width=1.5),
            hovertemplate="%{x|%a %d %b %H:%M}<br>Forecast: %{y:.2f}p/kWh<extra></extra>",
        ))
        title_suffix = ""

    fig.update_layout(
        template=_TEMPLATE,
        title=f"Predicted vs Actual — run of {predicted_on}{title_suffix}",
        yaxis_title="Price (p/kWh)",
        yaxis=dict(rangemode="tozero"),
        xaxis_title="",
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
        height=460,
        margin=dict(t=90, b=40),
    )
    return fig


def _fig_history(df: pd.DataFrame) -> go.Figure:
    """12-month daily EPEX wholesale price history."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX day-ahead (p/kWh)",
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>EPEX: %{y:.2f}p/kWh<extra></extra>",
    ))

    mean_ex = df["avg_epex_p_kwh"].mean()
    fig.add_hline(
        y=mean_ex, line_dash="dash", line_color=_PRICE_COL, line_width=0.8,
        annotation_text=f"Mean {mean_ex:.2f}p",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    fig.update_layout(
        template=_TEMPLATE,
        title="12-Month Daily Price History  (ex-VAT, network charges retained)",
        yaxis_title="Price (p/kWh)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=380,
        margin=dict(t=80, b=40),
    )
    return fig


def _fig_commodity(df: pd.DataFrame) -> go.Figure:
    """Dual-axis: Agile price vs TTF gas + Brent crude."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX wholesale (p/kWh)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Agile: %{y:.2f}p/kWh<extra></extra>",
    ), secondary_y=False)

    if "gas_ttf_roll7" in df.columns and df["gas_ttf_roll7"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["gas_ttf_roll7"],
            mode="lines", name="TTF Gas 7-day avg (€/MWh)",
            line=dict(color="#27ae60", width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>TTF: €%{y:.1f}/MWh<extra></extra>",
        ), secondary_y=True)

    if "brent_roll7" in df.columns and df["brent_roll7"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["brent_roll7"],
            mode="lines", name="Brent Crude 7-day avg ($/bbl)",
            line=dict(color="#f39c12", width=1.5, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Brent: $%{y:.1f}/bbl<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE,
        title="EPEX Wholesale vs Gas & Oil  (TTF gas is the primary driver of UK wholesale electricity)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="EPEX Wholesale (p/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Commodity Price  (€/MWh or $/bbl)", secondary_y=True)
    return fig


def _fig_demand_solar(df: pd.DataFrame) -> go.Figure:
    """Dual-axis: Agile price (left) vs GB demand + solar generation (right)."""
    has_demand = "demand_mw" in df.columns and df["demand_mw"].notna().any()
    has_solar  = "solar_gw"  in df.columns and df["solar_gw"].notna().any()

    if not has_demand and not has_solar:
        fig = go.Figure()
        fig.update_layout(title="Demand & Solar — no data yet", height=380,
                          template=_TEMPLATE)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX wholesale (p/kWh)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:.2f}p/kWh<extra></extra>",
    ), secondary_y=False)

    if has_demand:
        demand_gw = df["demand_mw"] / 1000.0
        fig.add_trace(go.Scatter(
            x=df["date"], y=demand_gw,
            mode="lines", name="GB Demand (GW, daily avg)",
            line=dict(color=_DEMAND_COL, width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>Demand: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    if has_solar:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["solar_gw"],
            mode="lines", name="GB Solar (GW, daily avg)",
            line=dict(color=_SOLAR_COL, width=1.2, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Solar: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE,
        title="EPEX Wholesale vs GB Demand & Solar Generation  "
              "(high demand → higher price; high solar → lower price)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="EPEX Wholesale (p/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="GB Generation / Demand (GW)", secondary_y=True)
    return fig


def _fig_generation_mix(df: pd.DataFrame) -> go.Figure:
    """Dual-axis: Agile price (left) vs GB generation mix (right)."""
    has_wind    = "wind_gen_mw" in df.columns and df["wind_gen_mw"].notna().any()
    has_gas     = "gas_gen_mw"  in df.columns and df["gas_gen_mw"].notna().any()
    has_nuclear = "nuclear_mw"  in df.columns and df["nuclear_mw"].notna().any()
    has_imports = "imports_mw"  in df.columns and df["imports_mw"].notna().any()

    if not any([has_wind, has_gas, has_nuclear, has_imports]):
        fig = go.Figure()
        fig.update_layout(title="GB Generation Mix — no data yet", height=380,
                          template=_TEMPLATE)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX wholesale (p/kWh)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:.2f}p/kWh<extra></extra>",
    ), secondary_y=False)

    if has_wind:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["wind_gen_mw"] / 1000.0,
            mode="lines", name="Wind (GW)",
            line=dict(color="#27ae60", width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>Wind: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    if has_gas:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["gas_gen_mw"] / 1000.0,
            mode="lines", name="Gas (GW)",
            line=dict(color="#e67e22", width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>Gas: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    if has_nuclear:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["nuclear_mw"] / 1000.0,
            mode="lines", name="Nuclear (GW)",
            line=dict(color="#9b59b6", width=1.2, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>Nuclear: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    if has_imports:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["imports_mw"] / 1000.0,
            mode="lines", name="Net Imports (GW, + = importing)",
            line=dict(color="#3498db", width=1.2, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>Imports: %{y:.1f} GW<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE,
        title="EPEX Wholesale vs GB Generation Mix  "
              "(gas ↑ → price up; wind / imports ↑ → price down)",
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
        height=420,
        margin=dict(t=90, b=40),
    )
    fig.update_yaxes(title_text="EPEX Wholesale (p/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Generation (GW)", secondary_y=True)
    return fig


def _fig_scatter(df: pd.DataFrame) -> go.Figure:
    """3×2 scatter plots: price vs the six most informative predictors."""
    all_vars = [
        ("gas_gen_mw",         "GB Gas Generation (MW)",       "#e67e22"),
        ("demand_mw",          "GB Demand (MW)",                "#8e44ad"),
        ("wind_gen_mw",        "GB Wind Generation (MW)",       "#27ae60"),
        ("epex_lag1_gbp_mwh",  "EPEX Day-Ahead Lag-1 (£/MWh)", "#2980b9"),
        ("pumped_storage_mw",  "GB Pumped Storage (MW)",        "#e74c3c"),
        ("solar_gw",           "Solar Generation (GW)",         "#f39c12"),
    ]
    var_list = [(v, l, c) for v, l, c in all_vars
                if v in df.columns and df[v].notna().any()]

    n    = len(var_list)
    ncols = 2
    nrows = (n + 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[l for _, l, _ in var_list])

    for idx, (var, label, colour) in enumerate(var_list):
        r, c = divmod(idx, ncols)
        valid = df[["avg_epex_p_kwh", var]].dropna()
        x = valid[var].values
        y = valid["avg_epex_p_kwh"].values

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=label,
            marker=dict(color=colour, size=5, opacity=0.4),
            showlegend=False,
            hovertemplate=f"{label}: %{{x:.1f}}<br>Price: %{{y:.2f}}p<extra></extra>",
        ), row=r + 1, col=c + 1)

        m, b = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", line=dict(color="black", width=1.5),
            showlegend=False,
        ), row=r + 1, col=c + 1)

        fig.update_xaxes(title_text=label, row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="Price (p/kWh)", row=r + 1, col=c + 1)

    fig.update_layout(
        template=_TEMPLATE,
        title="Price vs Key Predictors  (scatter with linear trend line)",
        height=max(500, 340 * nrows),
        margin=dict(t=80, b=40),
    )
    return fig


def _fig_backtest(backtest_df: pd.DataFrame, metrics: dict,
                  verifiable_df=None) -> go.Figure:
    """Daily hold-out backtest: actual vs predicted."""
    fig = go.Figure()

    if not backtest_df.empty:
        fig.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["avg_epex_p_kwh"],
            mode="lines+markers", name="Actual",
            line=dict(color=_PRICE_COL, width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.2f}p/kWh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["predicted_epex_p_kwh"],
            mode="lines+markers", name="Predicted (out-of-sample)",
            line=dict(color=_FORECAST_COL, width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:.2f}p/kWh<extra></extra>",
        ))

    if verifiable_df is not None and len(verifiable_df) > 0:
        vdf = pd.DataFrame(
            verifiable_df,
            columns=["predicted_on", "date", "predicted_epex_p_kwh", "actual_epex_p_kwh"],
        )
        vdf["date"] = pd.to_datetime(vdf["date"])
        latest_pred_on = vdf["predicted_on"].max()
        vdf_latest = vdf[vdf["predicted_on"] == latest_pred_on]
        if not vdf_latest.empty:
            fig.add_trace(go.Scatter(
                x=vdf_latest["date"], y=vdf_latest["predicted_epex_p_kwh"],
                mode="lines+markers", name=f"Stored forecast (made {latest_pred_on})",
                line=dict(color="#27ae60", width=2, dash="dot"),
                marker=dict(size=7, symbol="star"),
                hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:.2f}p/kWh<extra></extra>",
            ))

    mae_str  = f"{metrics.get('mae', 0):.2f}p"
    rmse_str = f"{metrics.get('rmse', 0):.2f}p"
    mape_str = f"{metrics.get('mape', 0):.1f}%"
    r2_str   = f"{metrics.get('r2', 0):.3f}"
    hold     = metrics.get("holdout_days", 30)
    train    = metrics.get("train_days", "?")

    fig.update_layout(
        template=_TEMPLATE,
        title=(f"Daily Prediction Accuracy — {hold}-Day Hold-Out  "
               f"(MAE {mae_str} · R² {r2_str} · RMSE {rmse_str} · MAPE {mape_str})<br>"
               f"<sup>Trained on {train} days, tested on following {hold} days "
               f"using actual weather (best-case accuracy ceiling)</sup>"),
        yaxis_title="EPEX Wholesale (p/kWh)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
        height=420,
        margin=dict(t=100, b=40),
    )
    return fig


def _fig_hh_backtest(hh_bt_df, hh_bt_metrics: dict) -> go.Figure:
    """Half-hourly backtest: actual vs predicted for each slot over hold-out period."""
    fig = go.Figure()

    if hh_bt_df is None or (hasattr(hh_bt_df, "empty") and hh_bt_df.empty):
        fig.update_layout(title="Half-Hourly Accuracy — insufficient data",
                          height=380, template=_TEMPLATE)
        return fig

    df = hh_bt_df.copy()
    dates = df["datetime_local"].dt.date.unique()
    for d in dates:
        fig.add_vrect(
            x0=f"{d}T16:00", x1=f"{d}T19:00",
            fillcolor=_PEAK_COL, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=df["datetime_local"], y=df["actual"],
        mode="lines", name="Actual",
        line=dict(color=_PRICE_COL, width=1),
        hovertemplate="%{x|%d %b %H:%M}<br>Actual: %{y:.2f}p<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime_local"], y=df["predicted"],
        mode="lines", name="Predicted (out-of-sample)",
        line=dict(color=_FORECAST_COL, width=1, dash="dash"),
        hovertemplate="%{x|%d %b %H:%M}<br>Predicted: %{y:.2f}p<extra></extra>",
    ))

    mae      = hh_bt_metrics.get("mae", 0)
    r2       = hh_bt_metrics.get("r2", 0)
    peak_mae = hh_bt_metrics.get("peak_mae", 0)
    op_mae   = hh_bt_metrics.get("offpeak_mae", 0)
    hold     = hh_bt_metrics.get("holdout_days", 30)

    fig.update_layout(
        template=_TEMPLATE,
        title=(f"Half-Hourly Prediction Accuracy — {hold}-Day Hold-Out  "
               f"(MAE {mae:.2f}p · R² {r2:.3f} · Peak MAE {peak_mae:.2f}p · "
               f"Off-peak MAE {op_mae:.2f}p)"),
        yaxis_title="Wholesale EPEX (p/kWh)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
    )
    return fig


def _fig_correlation_bars(correlations: dict) -> go.Figure:
    """Horizontal bar chart of all Pearson correlations, sorted by magnitude."""
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]["r"])
    labels = [v["label"] for _, v in sorted_corr]
    r_vals = [v["r"] for _, v in sorted_corr]
    colours = [
        "#27ae60" if r < -0.3 else ("#a9dfbf" if r < 0 else ("#f1948a" if r < 0.3 else "#e74c3c"))
        for r in r_vals
    ]

    fig = go.Figure(go.Bar(
        x=r_vals, y=labels,
        orientation="h",
        marker_color=colours,
        hovertemplate="%{y}<br>r = %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#2c3e50")
    fig.update_layout(
        template=_TEMPLATE,
        title="Price Driver Correlations  (Pearson r with ex-VAT Agile price)",
        xaxis_title="Correlation coefficient r",
        xaxis=dict(range=[-1, 1]),
        height=max(300, 28 * len(labels) + 80),
        margin=dict(t=60, b=40, l=260, r=20),
    )
    return fig


def _fig_generation_stacked(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of GB generation mix (GW) over time, with price on secondary axis."""
    cols = [
        ("nuclear_mw",       "Nuclear",  "#9b59b6", "rgba(155,89,182,0.7)"),
        ("hydro_mw",         "Hydro",    "#1abc9c", "rgba(26,188,156,0.7)"),
        ("solar_gw_scaled",  "Solar",    "#f39c12", "rgba(243,156,18,0.7)"),
        ("wind_gen_mw",      "Wind",     "#27ae60", "rgba(39,174,96,0.7)"),
        ("gas_gen_mw",       "Gas",      "#e67e22", "rgba(230,126,34,0.7)"),
    ]

    # Scale solar to MW for stacking
    df = df.copy()
    if "solar_gw" in df.columns:
        df["solar_gw_scaled"] = df["solar_gw"] * 1000.0

    available = [(col, label, colour, fill) for col, label, colour, fill in cols
                 if col in df.columns and df[col].notna().any()]

    if not available:
        fig = go.Figure()
        fig.update_layout(title="Generation Mix — no data yet", height=400,
                          template=_TEMPLATE)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col, label, colour, fill in available:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col] / 1000.0,
            mode="lines", name=label,
            stackgroup="gen",
            line=dict(width=0.5, color=colour),
            fillcolor=fill,
            hovertemplate=f"%{{x|%d %b}}<br>{label}: %{{y:.1f}} GW<extra></extra>",
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX wholesale (p/kWh)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:.2f}p/kWh<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE,
        title="GB Generation Mix Over Time  (stacked area = GW dispatched)",
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
        height=440,
        margin=dict(t=90, b=40),
    )
    fig.update_yaxes(title_text="Generation (GW)", secondary_y=False)
    fig.update_yaxes(title_text="EPEX Wholesale (p/kWh)", secondary_y=True)
    return fig


def _fig_epex_vs_agile(df: pd.DataFrame) -> go.Figure:
    """Scatter / dual-axis: EPEX SPOT day-ahead price vs Agile tariff over time."""
    has_epex = "epex_lag1_gbp_mwh" in df.columns and df["epex_lag1_gbp_mwh"].notna().any()
    if not has_epex:
        fig = go.Figure()
        fig.update_layout(title="EPEX Day-Ahead vs Agile — no data yet",
                          height=380, template=_TEMPLATE)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_epex_p_kwh"],
        mode="lines", name="EPEX wholesale (p/kWh)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Agile: %{y:.2f}p/kWh<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["epex_lag1_gbp_mwh"],
        mode="lines", name="EPEX SPOT day-ahead lag-1 (£/MWh)",
        line=dict(color="#2980b9", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>EPEX: £%{y:.1f}/MWh<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE,
        title="EPEX Wholesale vs EPEX Day-Ahead  (Agile is priced from the day-ahead wholesale auction)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="EPEX Wholesale (p/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="EPEX Day-Ahead (£/MWh)", secondary_y=True)
    return fig


def _daily_forecast_table(predictions: pd.DataFrame, hist_mean: float) -> go.Figure:
    """Compact table of 7-day daily EPEX predictions from the daily model."""
    rows = []
    for _, r in predictions.iterrows():
        d    = pd.to_datetime(r["date"])
        pred = r["predicted_epex_p_kwh"]
        vs   = pred - hist_mean
        sign = "▲" if vs >= 0 else "▼"
        is_actual = bool(r.get("is_actual", False))
        rows.append({
            "day":   d.strftime("%A"),
            "date":  d.strftime("%d %b %Y"),
            "type":  "Actual" if is_actual else "Forecast",
            "pred":  f"{pred:.2f}p",
            "vs":    f"{sign} {abs(vs):.2f}p vs 12m avg",
            "_pred": pred,
            "_actual": is_actual,
        })

    df = pd.DataFrame(rows)

    cell_colours = [
        "#d5f5e3" if a else ("#ffeaea" if v > hist_mean else "#eafff0")
        for v, a in zip(df["_pred"], df["_actual"])
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Day</b>", "<b>Date</b>", "<b>Type</b>",
                    "<b>EPEX (daily avg)</b>", "<b>vs 12m avg</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=13),
            align="left", height=36,
        ),
        cells=dict(
            values=[df["day"], df["date"], df["type"], df["pred"], df["vs"]],
            align="left", font=dict(size=12), height=32,
            fill_color=[cell_colours] * 5,
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=f"7-Day Outlook  (D+1 = settled auction price · D+2+ = model forecast · 12m avg = {hist_mean:.2f}p)",
        height=max(300, 120 + 36 * len(df)),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


def _corr_table(correlations: dict) -> go.Figure:
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    labels  = [v["label"] for _, v in sorted_corr]
    r_vals  = [f"{v['r']:+.4f}" for _, v in sorted_corr]
    p_vals  = [f"{v['p']:.2e}" for _, v in sorted_corr]
    sig     = [
        "***" if v["p"] < 0.001 else ("**" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else ""))
        for _, v in sorted_corr
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Variable</b>", "<b>r</b>", "<b>p-value</b>", "<b>Sig.</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=13),
            align="left", height=36,
        ),
        cells=dict(
            values=[labels, r_vals, p_vals, sig],
            align="left", font=dict(size=12), height=32,
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Pearson Correlations — Detail Table",
        height=max(300, 116 + 32 * len(labels)),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


def _model_table(r2_daily: float, r2_hh: float | None,
                 model, feature_cols: list[str]) -> go.Figure:
    rows_labels = [ALL_FEATURE_LABELS.get(c, c) for c in feature_cols]
    rows_coef   = [f"{c:+.4f}" for c in model.coef_]

    cell_metrics = [
        "Daily model R² (in-sample)",
        "Half-hourly R² (in-sample, incl. time features)",
        "Intercept",
    ] + rows_labels
    cell_values = [
        f"{r2_daily:.4f}  ({r2_daily*100:.1f}%)",
        f"{r2_hh:.4f}  ({r2_hh*100:.1f}%)" if r2_hh is not None else "n/a",
        f"{model.intercept_:+.4f}",
    ] + rows_coef

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Feature</b>", "<b>Coefficient (standardised β)</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=13),
            align="left", height=36,
        ),
        cells=dict(
            values=[cell_metrics, cell_values],
            align="left", font=dict(size=12), height=32,
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Daily Regression Model — Coefficients",
        height=max(300, 116 + 32 * len(cell_metrics)),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


def _fig_leadtime_from_predictions(verifiable_df, max_lead: int = 7) -> go.Figure:
    """
    Build a lead-time accuracy chart from stored daily predictions vs actuals.
    verifiable_df is a list of sqlite3.Row or similar with
    (predicted_on, date, predicted_epex_p_kwh, actual_epex_p_kwh).
    """
    import pandas as _pd
    vdf = _pd.DataFrame(
        verifiable_df,
        columns=["predicted_on", "date", "predicted_epex_p_kwh", "actual_epex_p_kwh"],
    )
    vdf["predicted_on"] = _pd.to_datetime(vdf["predicted_on"])
    vdf["date"] = _pd.to_datetime(vdf["date"])
    vdf["lead_days"] = (vdf["date"] - vdf["predicted_on"]).dt.days
    vdf["abs_error"] = (vdf["predicted_epex_p_kwh"] - vdf["actual_epex_p_kwh"]).abs()
    vdf = vdf[(vdf["lead_days"] >= 0) & (vdf["lead_days"] <= max_lead)]

    if vdf.empty:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, height=340,
                          title="Forecast Accuracy by Lead Time — No Data Yet")
        return fig

    grouped = vdf.groupby("lead_days").agg(
        mae=("abs_error", "mean"),
        n=("abs_error", "count"),
    ).reset_index()

    leads = grouped["lead_days"].tolist()
    mae_vals = grouped["mae"].tolist()
    n_vals = grouped["n"].tolist()
    labels = [f"Day+{l}" for l in leads]
    baseline = mae_vals[0] if mae_vals else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=mae_vals,
        name="MAE (p/kWh)",
        marker_color=[
            "#e74c3c" if v > baseline * 1.5 else
            "#f39c12" if v > baseline * 1.1 else
            "#27ae60"
            for v in mae_vals
        ],
        text=[f"{v:.2f}p<br>n={n}" for v, n in zip(mae_vals, n_vals)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f}p/kWh<extra></extra>",
    ))

    # Add individual prediction scatter
    for _, row in vdf.iterrows():
        fig.add_trace(go.Scatter(
            x=[f"Day+{int(row['lead_days'])}"],
            y=[row["abs_error"]],
            mode="markers",
            marker=dict(size=8, color="#2980b9", opacity=0.5),
            showlegend=False,
            hovertemplate=(
                f"Made {row['predicted_on'].strftime('%d %b')}, "
                f"for {row['date'].strftime('%d %b')}<br>"
                f"Pred: {row['predicted_epex_p_kwh']:.2f}p, "
                f"Actual: {row['actual_epex_p_kwh']:.2f}p<br>"
                f"Error: {row['abs_error']:.2f}p<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=_TEMPLATE,
        title=(
            "Stored Prediction Accuracy by Lead Time  "
            "<sup>Comparing daily stored forecasts to actual EPEX prices · "
            "dots show individual predictions</sup>"
        ),
        yaxis_title="Absolute Error (p/kWh)", hovermode="x unified", height=400,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=80, b=80),
    )
    return fig


def _fig_leadtime_accuracy(detail_df: pd.DataFrame, metrics_by_lead: dict,
                            max_lead: int = 7,
                            verifiable_df=None) -> go.Figure:
    """
    Bar chart of real-world forecast MAE by lead time, from the archived forecast backtest.
    Falls back to stored predictions vs actuals when not enough archive days are available.
    """
    if not metrics_by_lead:
        # Try stored predictions as a fallback
        if verifiable_df is not None and len(verifiable_df) > 0:
            return _fig_leadtime_from_predictions(verifiable_df, max_lead)
        fig = go.Figure()
        fig.update_layout(
            template=_TEMPLATE, height=340,
            title="Forecast Accuracy by Lead Time — Collecting Data",
            annotations=[dict(
                text=(
                    "Archive forecasts are being collected from today.<br>"
                    "This chart will populate after ~7+ days of daily runs."
                ),
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="#7f8c8d"),
            )],
        )
        return fig

    leads     = sorted(metrics_by_lead.keys())
    mae_vals  = [metrics_by_lead[l]["mae"]  for l in leads]
    rmse_vals = [metrics_by_lead[l]["rmse"] for l in leads]
    n_vals    = [metrics_by_lead[l]["n"]    for l in leads]
    labels    = [f"Day+{l}" for l in leads]
    baseline  = mae_vals[0] if mae_vals else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=mae_vals,
        name="MAE (p/kWh)",
        marker_color=[
            "#e74c3c" if v > baseline * 1.5 else
            "#f39c12" if v > baseline * 1.1 else
            "#27ae60"
            for v in mae_vals
        ],
        text=[f"{v:.2f}p<br>n={n}" for v, n in zip(mae_vals, n_vals)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f}p/kWh<br>n=%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=rmse_vals,
        name="RMSE (p/kWh)", mode="lines+markers",
        line=dict(color="#2980b9", width=2), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f}p/kWh<extra></extra>",
    ))
    fig.add_hline(
        y=baseline, line_dash="dot", line_color="#27ae60",
        annotation_text=f"Day+1 baseline {baseline:.2f}p",
        annotation_position="top right",
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=(
            "Real-World Forecast Accuracy by Lead Time — Day+1 to Day+7  "
            "<sup>Uses actual archived NWP forecasts · "
            "each bar shows the average error for all days predicted at that lead time</sup>"
        ),
        yaxis_title="Error (p/kWh)", hovermode="x unified", height=400,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=80, b=80),
    )
    return fig


def _fig_hourly_profile(hourly_profile: pd.DataFrame) -> go.Figure:
    """
    Bar chart of mean EPEX price by hour of day, with weekday/weekend split
    and a shaded ±1 std band. Directly informs tariff band boundary choices.
    """
    if hourly_profile is None or hourly_profile.empty:
        fig = go.Figure()
        fig.update_layout(title="Hourly Price Profile — no data", height=380,
                          template=_TEMPLATE)
        return fig

    fig = go.Figure()

    for is_wd, label, colour, dash in [
        (True,  "Weekday", _PRICE_COL, "solid"),
        (False, "Weekend", "#2980b9",  "dot"),
    ]:
        sub = hourly_profile[hourly_profile["is_weekday"] == is_wd].sort_values("hour")
        if sub.empty:
            continue
        # ±1 std band
        fig.add_trace(go.Scatter(
            x=list(sub["hour"]) + list(sub["hour"])[::-1],
            y=list(sub["mean"] + sub["std"]) + list((sub["mean"] - sub["std"]).clip(lower=0))[::-1],
            fill="toself",
            fillcolor="rgba(231,76,60,0.08)" if is_wd else "rgba(41,128,185,0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["mean"],
            mode="lines+markers", name=label,
            line=dict(color=colour, width=2, dash=dash),
            marker=dict(size=5),
            hovertemplate="Hour %{x}:00<br>Mean: %{y:.2f}p/kWh<extra></extra>",
        ))

    # Peak period shading
    fig.add_vrect(x0=16, x1=19, fillcolor=_PEAK_COL, line_width=0,
                  annotation_text="DUoS red (peak)", annotation_position="top right",
                  annotation_font_size=9)
    # Amber band boundaries
    fig.add_vrect(x0=7,  x1=16, fillcolor="rgba(243,156,18,0.06)", line_width=0)
    fig.add_vrect(x0=19, x1=23, fillcolor="rgba(243,156,18,0.06)", line_width=0)

    fig.update_layout(
        template=_TEMPLATE,
        title="Average EPEX Wholesale Price by Hour of Day  "
              "(shading = DUoS band zones · band = ±1 std dev)",
        xaxis_title="Hour of day",
        yaxis_title="Mean EPEX (p/kWh)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, range=[-0.5, 23.5]),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
    )
    return fig



# ── Ensemble / LightGBM charts ──────────────────────────────────────────────
# (tariff/simulation chart functions removed — moved to scottishpower/tariff)


def _fig_lgbm_importance(ensemble: dict, label: str = "") -> go.Figure:
    """Horizontal bar chart of LightGBM feature importances."""
    lgbm_model = ensemble["lgbm"]["model"]
    feature_cols = ensemble["lgbm"]["feature_cols"]
    importances = lgbm_model.feature_importances_
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(20)

    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker_color="#27ae60",
    ))
    title = f"{label} LightGBM Feature Importance (top 20)" if label else "LightGBM Feature Importance (top 20)"
    fig.update_layout(
        title=title,
        xaxis_title="Split count",
        height=max(350, len(df) * 22 + 100),
        margin=dict(l=180, t=50, b=40, r=20),
    )
    return fig


def _fig_walkforward_cv(wfcv_detail: pd.DataFrame, wfcv_metrics: dict) -> go.Figure:
    """Walk-forward CV actual vs predicted with fold shading."""
    if wfcv_detail.empty:
        fig = go.Figure()
        fig.update_layout(title="Walk-Forward CV — no data", height=300)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wfcv_detail["date"], y=wfcv_detail["actual"],
        mode="lines", name="Actual", line=dict(color="#2c3e50", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=wfcv_detail["date"], y=wfcv_detail["predicted"],
        mode="lines", name="Ensemble", line=dict(color="#e74c3c", width=1.5),
    ))
    if "pred_ridge" in wfcv_detail.columns:
        fig.add_trace(go.Scatter(
            x=wfcv_detail["date"], y=wfcv_detail["pred_ridge"],
            mode="lines", name="Ridge", line=dict(color="#3498db", width=1, dash="dot"),
        ))
    if "pred_lgbm" in wfcv_detail.columns:
        fig.add_trace(go.Scatter(
            x=wfcv_detail["date"], y=wfcv_detail["pred_lgbm"],
            mode="lines", name="LightGBM", line=dict(color="#27ae60", width=1, dash="dot"),
        ))

    # Shade fold regions
    folds = sorted(wfcv_detail["fold"].unique())
    colors = ["rgba(52,152,219,0.08)", "rgba(46,204,113,0.08)"]
    for i, fold in enumerate(folds):
        fold_data = wfcv_detail[wfcv_detail["fold"] == fold]
        fig.add_vrect(
            x0=fold_data["date"].min(), x1=fold_data["date"].max(),
            fillcolor=colors[i % 2], line_width=0,
            annotation_text=f"Fold {fold+1}", annotation_position="top left",
        )

    mae_r = wfcv_metrics.get("mae_ridge", 0)
    mae_l = wfcv_metrics.get("mae_lgbm", 0)
    mae_e = wfcv_metrics.get("mae_ensemble", 0)
    fig.update_layout(
        title=f"Walk-Forward CV — Ridge {mae_r:.2f}p | LightGBM {mae_l:.2f}p | Ensemble {mae_e:.2f}p",
        xaxis_title="Date", yaxis_title="p/kWh",
        legend=dict(orientation="h", y=1.12),
        height=400, margin=dict(t=80, b=50),
    )
    return fig


def _fig_prediction_intervals(daily_predictions: pd.DataFrame) -> go.Figure | None:
    """Forecast with 80% prediction interval from quantile regression."""
    if daily_predictions is None or daily_predictions.empty:
        return None
    if "pred_q10" not in daily_predictions.columns or "pred_q90" not in daily_predictions.columns:
        return None

    fig = go.Figure()
    # Prediction interval band
    fig.add_trace(go.Scatter(
        x=pd.concat([daily_predictions["date"], daily_predictions["date"][::-1]]),
        y=pd.concat([daily_predictions["pred_q90"], daily_predictions["pred_q10"][::-1]]),
        fill="toself", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% interval",
    ))
    # Ensemble forecast line
    fig.add_trace(go.Scatter(
        x=daily_predictions["date"], y=daily_predictions["predicted_epex_p_kwh"],
        mode="lines+markers", name="Ensemble forecast",
        line=dict(color="#e74c3c", width=2),
    ))
    if "pred_ridge" in daily_predictions.columns:
        fig.add_trace(go.Scatter(
            x=daily_predictions["date"], y=daily_predictions["pred_ridge"],
            mode="lines", name="Ridge",
            line=dict(color="#3498db", width=1, dash="dot"),
        ))
    if "pred_lgbm" in daily_predictions.columns:
        fig.add_trace(go.Scatter(
            x=daily_predictions["date"], y=daily_predictions["pred_lgbm"],
            mode="lines", name="LightGBM",
            line=dict(color="#27ae60", width=1, dash="dot"),
        ))

    fig.update_layout(
        title="7-Day Forecast with 80% Prediction Interval",
        xaxis_title="Date", yaxis_title="p/kWh",
        legend=dict(orientation="h", y=1.12),
        height=380, margin=dict(t=70, b=50),
    )
    return fig


def _forecast_summary_html(summary: dict | None) -> str:
    """Render the LLM week-ahead summary as an HTML block."""
    if not summary:
        return ""
    days_html = ""
    for d in summary.get("days", []):
        days_html += (
            '<div style="padding:6px 10px; background:#f8f9fa; '
            'border-radius:4px; font-size:0.9em;">'
            f'<strong>{d["date"]}</strong>: {d["summary"]}</div>'
        )
    return (
        '<div class="forecast-summary" style="max-width:1200px; margin:0 auto 24px; '
        'padding:16px 24px; background:#fff; border-radius:8px; border-left:4px solid #3498db;">'
        f'<p style="font-size:1.05em; margin:0 0 12px; color:#2c3e50;">'
        f'<strong>Week Ahead:</strong> {summary["week_summary"]}</p>'
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:8px;">'
        f'{days_html}</div></div>'
    )


# ── Main generator ─────────────────────────────────────────────────────────────

def generate(
    df_daily: pd.DataFrame,
    hh_pred: pd.DataFrame,
    correlations: dict,
    r2_daily: float,
    r2_hh: float | None,
    model,
    feature_cols: list[str],
    backtest_df=None,
    backtest_metrics=None,
    verifiable_df=None,
    hh_backtest_df=None,
    hh_backtest_metrics=None,
    daily_predictions=None,
    leadtime_detail_df=None,
    leadtime_metrics=None,
    hh_hourly_profile=None,
    ensemble=None,
    hh_ensemble=None,
    wfcv_detail=None,
    wfcv_metrics=None,
    forecast_summary=None,
) -> Path:
    """Build and write the HTML dashboard. Returns the path."""

    hist_mean = df_daily["avg_epex_p_kwh"].mean()
    # EPEX mean for the HH forecast reference line (different scale to Agile retail)
    epex_mean = hh_pred["predicted_epex_p_kwh"].mean() if "predicted_epex_p_kwh" in hh_pred.columns else None
    updated   = datetime.now().strftime("%d %b %Y %H:%M")

    min_day = df_daily.loc[df_daily["avg_epex_p_kwh"].idxmin(), "date"].strftime("%d %b")
    max_day = df_daily.loc[df_daily["avg_epex_p_kwh"].idxmax(), "date"].strftime("%d %b")

    # Use out-of-sample MAE for stat cards — more meaningful than in-sample R²
    daily_mae_str = (f"{backtest_metrics['mae']:.2f}p"
                     if backtest_metrics else "n/a")
    hh_mae_str    = (f"{hh_backtest_metrics['mae']:.2f}p"
                     if hh_backtest_metrics else "n/a")

    # Today's forecast headline
    today_slots = hh_pred[hh_pred["datetime_local"].dt.date == hh_pred["datetime_local"].dt.date.min()]
    peak_slots    = today_slots[today_slots["predicted_epex_p_kwh"] == today_slots[
        today_slots["datetime_local"].dt.hour.between(16, 18)]["predicted_epex_p_kwh"].max()] if not today_slots.empty else None
    offpeak_mean  = today_slots[~today_slots["datetime_local"].dt.hour.between(16, 18)]["predicted_epex_p_kwh"].mean() if not today_slots.empty else None

    def _t(term: str, definition: str) -> str:
        """Wrap a TLA in an abbr tag with custom tooltip."""
        return f'<abbr title="{definition}">{term}</abbr>'

    MAE   = _t("MAE",  "Mean Absolute Error — average absolute difference between predicted and actual price")
    RMSE  = _t("RMSE", "Root Mean Square Error — like MAE but penalises large errors more heavily")
    MAPE  = _t("MAPE", "Mean Absolute Percentage Error — average error as a % of actual price")
    R2    = _t("R²",   "R-squared — proportion of price variance explained by the model (1.0 = perfect, 0 = no better than mean)")
    EPEX  = _t("EPEX", "European Power Exchange — the day-ahead spot auction that sets GB wholesale electricity prices for next-day delivery")
    TTF   = _t("TTF",  "Title Transfer Facility — the Dutch natural gas hub; the benchmark price for European gas and the main driver of UK electricity prices")
    BMRS  = _t("BMRS", "Balancing Mechanism Reporting Service — Elexon's data portal for GB electricity market data including generation and demand")
    kWh   = _t("kWh",  "kilowatt-hour — unit of energy; 1 kWh = the energy used by a 1,000W device for one hour")
    MWh   = _t("MWh",  "megawatt-hour — 1,000 kWh; wholesale electricity is typically priced in £/MWh")
    GW    = _t("GW",   "gigawatt — 1,000 MW; total GB electricity demand is typically 25–45 GW")
    MW    = _t("MW",   "megawatt — unit of power; a large wind turbine produces ~5 MW")
    VAT   = _t("VAT",  "Value Added Tax — UK electricity VAT rate is 5%")
    NWP   = _t("NWP",  "Numerical Weather Prediction — computer weather forecast models (e.g. ECMWF, GFS)")
    SPD   = _t("SPD",  "SP Distribution — the distribution network operator (DNO) for Central Scotland; publishes DUoS charges annually in the LC14 Charging Statement")
    INDO  = _t("INDO", "Initial National Demand Outturn — Elexon's half-hourly estimate of total GB electricity demand")
    PVLIVE = _t("PV_Live", "Sheffield Solar PV_Live — real-time API providing estimates of GB-wide solar photovoltaic generation")
    FUELHH = _t("FUELHH", "Fuel Half-Hourly — Elexon BMRS dataset reporting actual GB generation by fuel type every 30 minutes")
    MID   = _t("MID",  "Market Index Data — Elexon dataset reporting EPEX SPOT GB day-ahead auction clearing prices")
    APXMIDP = _t("APXMIDP", "APX Market Index Data Price — historical name for the GB day-ahead electricity market index, now EPEX SPOT GB")

    def _div(fig, full_width=True) -> str:
        cls = "chart-full" if full_width else "chart-half"
        inner = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={"displayModeBar": False})
        return f'<div class="{cls}">{inner}</div>'

    def _section(title: str, subtitle: str = "") -> str:
        sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
        return f'<div class="section-header"><h2>{title}</h2>{sub}</div>'

    mae_band = hh_backtest_metrics.get("mae") if hh_backtest_metrics else None
    fig_forecast      = _fig_halfhourly_forecast(hh_pred, epex_mean, mae_band=mae_band)
    fig_history       = _fig_history(df_daily)
    fig_commodity     = _fig_commodity(df_daily)
    fig_demand        = _fig_demand_solar(df_daily)
    fig_gen_stacked   = _fig_generation_stacked(df_daily)
    fig_corr_bars     = _fig_correlation_bars(correlations)
    fig_scatter       = _fig_scatter(df_daily)
    fig_corr          = _corr_table(correlations)
    fig_model         = _model_table(r2_daily, r2_hh, model, feature_cols)

    bt_df      = backtest_df      if backtest_df      is not None else pd.DataFrame()
    bt_metrics = backtest_metrics if backtest_metrics is not None else {}
    fig_backtest    = _fig_backtest(bt_df, bt_metrics, verifiable_df)
    fig_hh_backtest = _fig_hh_backtest(hh_backtest_df, hh_backtest_metrics or {})
    fig_hourly_profile = _fig_hourly_profile(hh_hourly_profile)

    fig_leadtime    = _fig_leadtime_accuracy(
        leadtime_detail_df if leadtime_detail_df is not None else pd.DataFrame(),
        leadtime_metrics or {},
        verifiable_df=verifiable_df,
    )

    if daily_predictions is not None and not daily_predictions.empty:
        fig_daily_forecast = _daily_forecast_table(daily_predictions, hist_mean)
    else:
        fig_daily_forecast = None

    # Predicted vs Actual chart (latest run)
    fig_pred_vs_actual = _fig_predicted_vs_actual(datetime.now().strftime("%Y-%m-%d"))

    # Ensemble charts
    fig_lgbm_imp_daily = _fig_lgbm_importance(ensemble, "Daily") if ensemble else None
    fig_lgbm_imp_hh = _fig_lgbm_importance(hh_ensemble, "Half-Hourly") if hh_ensemble else None
    fig_wfcv = _fig_walkforward_cv(
        wfcv_detail if wfcv_detail is not None else pd.DataFrame(),
        wfcv_metrics or {},
    ) if wfcv_detail is not None and not wfcv_detail.empty else None
    fig_pred_interval = _fig_prediction_intervals(daily_predictions)

    offpeak_str = f"{offpeak_mean:.2f}p" if offpeak_mean is not None else "—"

    # Build model contribution descriptions
    def _blend_desc(ens: dict | None, label: str) -> str:
        if ens is None:
            return f"{label}: Ridge only"
        w = ens["blend_weight"]
        r2_r = ens["ridge"]["r2"]
        r2_l = ens["lgbm"]["r2"]
        if w >= 0.95:
            mix = "100% Ridge"
        elif w <= 0.05:
            mix = "100% LightGBM"
        else:
            mix = f"{w*100:.0f}% Ridge + {(1-w)*100:.0f}% LightGBM"
        return f"{label}: {mix} (Ridge R²={r2_r:.3f}, LightGBM R²={r2_l:.3f})"

    daily_blend_desc = _blend_desc(ensemble, "Daily")
    hh_blend_desc = _blend_desc(hh_ensemble, "Half-hourly")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scotland Energy Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f6f9; color: #2c3e50; }}
    header {{ background: #2c3e50; color: white; padding: 20px 32px;
              display: flex; justify-content: space-between; align-items: center; }}
    header h1 {{ font-size: 1.4rem; font-weight: 600; }}
    header .updated {{ font-size: 0.85rem; opacity: 0.7; }}
    nav {{ background: #34495e; display: flex; gap: 0; }}
    nav a {{ color: rgba(255,255,255,0.75); text-decoration: none; padding: 10px 20px;
              font-size: 0.85rem; transition: background 0.15s; }}
    nav a:hover {{ background: rgba(255,255,255,0.1); color: white; }}
    .stats {{ display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap; }}
    .stat-card {{ background: white; border-radius: 10px; padding: 16px 24px;
                  flex: 1; min-width: 150px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .stat-card .label {{ font-size: 0.72rem; color: #7f8c8d; text-transform: uppercase;
                         letter-spacing: .06em; margin-bottom: 4px; }}
    .stat-card .value {{ font-size: 1.6rem; font-weight: 700; color: #2c3e50; }}
    .stat-card .sub {{ font-size: 0.72rem; color: #95a5a6; margin-top: 2px; }}
    .section-header {{ padding: 24px 32px 4px; }}
    .section-header h2 {{ font-size: 1.1rem; font-weight: 600; color: #2c3e50;
                          border-left: 4px solid #3498db; padding-left: 12px; }}
    .section-sub {{ font-size: 0.82rem; color: #7f8c8d; margin-top: 4px; padding-left: 16px; }}
    .charts {{ padding: 0 32px 32px; display: flex; flex-wrap: wrap; gap: 20px; }}
    .chart-full {{ width: 100%; background: white; border-radius: 10px; padding: 12px;
                   box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .chart-half {{ width: calc(50% - 10px); background: white; border-radius: 10px;
                   padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    @media (max-width: 900px) {{ .chart-half {{ width: 100%; }} }}
    footer {{ text-align: center; padding: 20px 32px; font-size: 0.78rem; color: #95a5a6;
              border-top: 1px solid #e0e0e0; margin-top: 12px; }}
    abbr {{
      text-decoration: underline dotted #95a5a6;
      cursor: help;
      position: relative;
      white-space: nowrap;
    }}
    abbr[title]::after {{
      content: attr(title);
      display: none;
      position: absolute;
      bottom: calc(100% + 6px);
      left: 50%;
      transform: translateX(-50%);
      background: #2c3e50;
      color: #fff;
      padding: 5px 10px;
      border-radius: 5px;
      font-size: 0.75rem;
      white-space: normal;
      max-width: 260px;
      min-width: 120px;
      text-align: center;
      z-index: 9999;
      pointer-events: none;
      line-height: 1.4;
      font-style: normal;
      font-weight: normal;
    }}
    abbr[title]::before {{
      content: '';
      display: none;
      position: absolute;
      bottom: calc(100% + 1px);
      left: 50%;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: #2c3e50;
      z-index: 9999;
      pointer-events: none;
    }}
    abbr[title]:hover::after,
    abbr[title]:hover::before {{ display: block; }}
  </style>
</head>
<body>

<header>
  <h1>⚡ UK Electricity Price Analysis</h1>
  <span class="updated">Updated {updated}</span>
</header>

<nav>
  <a href="#forecast">Forecast</a>
  <a href="#drivers">Price Drivers</a>
  <a href="#history">History</a>
  <a href="#accuracy">Model Accuracy</a>
  <a href="#model">Model Detail</a>
</nav>

<div class="stats">
  <div class="stat-card">
    <div class="label">12-Month Mean Price</div>
    <div class="value">{hist_mean:.2f}p</div>
    <div class="sub">ex-{VAT} per {kWh}</div>
  </div>
  <div class="stat-card">
    <div class="label">Today — Off-Peak Forecast</div>
    <div class="value">{offpeak_str}</div>
    <div class="sub">predicted wholesale avg</div>
  </div>
  <div class="stat-card">
    <div class="label">Cheapest Day (12 months)</div>
    <div class="value">{df_daily["avg_epex_p_kwh"].min():.2f}p</div>
    <div class="sub">{min_day}</div>
  </div>
  <div class="stat-card">
    <div class="label">Most Expensive Day</div>
    <div class="value">{df_daily["avg_epex_p_kwh"].max():.2f}p</div>
    <div class="sub">{max_day}</div>
  </div>
  <div class="stat-card">
    <div class="label">Forecast Accuracy ({MAE})</div>
    <div class="value">{daily_mae_str}</div>
    <div class="sub">daily avg · 30-day hold-out</div>
  </div>
  <div class="stat-card">
    <div class="label">Half-Hourly {MAE}</div>
    <div class="value">{hh_mae_str}</div>
    <div class="sub">per slot · 30-day hold-out</div>
  </div>

</div>

<div id="forecast">
  {_section("7-Day Forecast",
            f"Predicted half-hourly {EPEX} wholesale prices for the next 7 days. "
            f"Shaded bands = peak rate period (16:00–19:00). "
            f"<strong>{hh_blend_desc}</strong>. "
            f"Based on {NWP} weather forecast + current gas prices + yesterday's per-slot {EPEX} price as an autoregressive anchor. "
            f"These are raw wholesale costs — network charges and supplier margin are not included.")}
</div>
<div class="charts">
  {_div(fig_forecast)}
  {"" if fig_daily_forecast is None else _div(fig_daily_forecast)}
</div>
{"" if fig_pred_vs_actual is None else f'''
<div id="pred-vs-actual">
  {_section("Predicted vs Actual",
            "How did today's forecast compare to actual settled prices? "
            "Green = actual EPEX price, dashed blue = what the model predicted, "
            "pink shading = prediction error. Solid blue = future forecast.")}
</div>
<div class="charts">
  {_div(fig_pred_vs_actual)}
</div>
'''}
{_forecast_summary_html(forecast_summary)}

<div id="drivers">
  {_section("Price Drivers",
            f"What drives the wholesale price? Gas generation and demand push it up; "
            f"wind, solar, and imports push it down.")}
</div>
<div class="charts">
  {_div(fig_corr_bars)}
  {_div(fig_commodity)}
  {_div(fig_demand)}
</div>

<div id="history">
  {_section("12-Month Price History",
            "Daily average EPEX wholesale price and GB generation mix over the past 12 months.")}
</div>
<div class="charts">
  {_div(fig_history)}
  {_div(fig_gen_stacked)}
</div>

<div id="accuracy">
  {_section("Model Accuracy",
            f"Out-of-sample hold-out test: the model is trained on historical data and tested on "
            f"the most recent 30 days it has never seen, using actual observed weather throughout. "
            f"This is effectively a D+1 best-case accuracy ceiling — it does not reflect "
            f"multi-day forecast accuracy, where {NWP} weather forecast errors accumulate. "
            f"{MAE} · {RMSE} · {MAPE} · {R2}")}
</div>
<div class="charts">
  {_div(fig_backtest)}
  {_div(fig_hh_backtest)}
  {_div(fig_leadtime)}
</div>

{f'''
<div id="model">
  {_section("Model Detail",
            f"Two models — daily average and half-hourly per-slot — each using a Ridge + LightGBM ensemble. "
            f"The blend weight is optimised via walk-forward cross-validation. "
            f"<strong>{daily_blend_desc}</strong>. <strong>{hh_blend_desc}</strong>.")}
</div>

{_section("Daily Model",
          f"Predicts the daily average {EPEX} wholesale price. "
          f"{'<strong>' + daily_blend_desc + '</strong>. ' if ensemble else ''}"
          f"Ridge coefficients are standardised (β) — magnitude reflects relative importance.")}
<div class="charts">
  {"" if fig_wfcv is None else _div(fig_wfcv)}
  {"" if fig_lgbm_imp_daily is None else _div(fig_lgbm_imp_daily, full_width=False)}
  {_div(fig_model, full_width=False)}
  {"" if fig_pred_interval is None else _div(fig_pred_interval)}
  {_div(fig_scatter)}
  {_div(fig_corr, full_width=False)}
</div>

{_section("Half-Hourly Model",
          f"Predicts per-slot {EPEX} wholesale prices with recursive day-by-day lag updates. "
          f"{'<strong>' + hh_blend_desc + '</strong>. ' if hh_ensemble else ''}"
          f"Uses time-of-day and day-of-year cyclic features alongside weather, commodity, and inventory signals.")}
<div class="charts">
  {"" if fig_lgbm_imp_hh is None else _div(fig_lgbm_imp_hh)}
  {_div(fig_hourly_profile)}
</div>
'''}

<footer>
  Wholesale prices: GB-wide {EPEX} SPOT day-ahead ·
  Weather: Open-Meteo UK average (6 sites: Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff) ·
  Solar: Sheffield Solar {PVLIVE} (GB national generation) ·
  Demand: Elexon {BMRS} {INDO} ·
  Generation mix: Elexon {BMRS} {FUELHH} (wind, gas, nuclear, pumped hydro, hydro, interconnectors) ·
  Day-ahead prices: Elexon {BMRS} {MID} ({EPEX} SPOT GB / {APXMIDP}) ·
  Commodity: Yahoo Finance ({TTF} gas, Brent crude)
</footer>

</body>
</html>"""

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH
