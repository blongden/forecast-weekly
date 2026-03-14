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

from app.config import BASE_DIR, PUBLIC_MODE
from app.analysis import ALL_FEATURE_LABELS

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
        rows.append({
            "day":   d.strftime("%A"),
            "date":  d.strftime("%d %b %Y"),
            "pred":  f"{pred:.2f}p",
            "vs":    f"{sign} {abs(vs):.2f}p vs 12m avg",
            "_pred": pred,
        })

    df = pd.DataFrame(rows)

    cell_colours = [
        "#ffeaea" if v > hist_mean else "#eafff0"
        for v in df["_pred"]
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Day</b>", "<b>Date</b>",
                    "<b>Predicted EPEX (daily avg)</b>", "<b>vs 12m avg</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=13),
            align="left", height=36,
        ),
        cells=dict(
            values=[df["day"], df["date"], df["pred"], df["vs"]],
            align="left", font=dict(size=12), height=32,
            fill_color=[cell_colours] * 4,
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=f"7-Day Daily Forecast  (daily model · 12m avg = {hist_mean:.2f}p · red = above avg)",
        height=max(300, 120 + 36 * len(df)),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


def _fig_daily_tariff(daily_tariffs: list, slots_label: str,
                      agile_by_date: dict | None = None) -> go.Figure:
    """
    Grouped + stacked bar chart: x = band, groups = day, bars stacked by cost component.
    Shows one panel per day (up to 3) side by side.
    agile_by_date: optional dict {date → {band_name: avg_price_ex_vat}} for Agile overlay.
    """
    COMPONENTS = [
        ("epex_mean_p_kwh",  "EPEX Wholesale",  "#e74c3c"),
        ("duos_p_kwh",       "DUoS",            "#9b59b6"),
        ("tnuos_p_kwh",      "TNUoS",           "#2980b9"),
        ("bsuos_p_kwh",      "BSUoS buffer",    "#16a085"),
        ("mae_buffer_p_kwh", "Forecast buffer", "#f39c12"),
        ("margin_p_kwh",     "Margin",          "#27ae60"),
    ]

    n = len(daily_tariffs)
    day_labels = [pd.Timestamp(d).strftime("%a %d %b") for d, _ in daily_tariffs]

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=day_labels,
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )

    for col_idx, (d, tariff) in enumerate(daily_tariffs, start=1):
        band_labels = tariff["band"].tolist()
        for comp_col, comp_label, colour in COMPONENTS:
            vals = tariff[comp_col].tolist()
            fig.add_trace(go.Bar(
                name=comp_label,
                x=band_labels,
                y=vals,
                marker_color=colour,
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>{comp_label}</b><br>%{{x}}: %{{y:.2f}}p/kWh<extra></extra>",
            ), row=1, col=col_idx)

        for _, row in tariff.iterrows():
            fig.add_annotation(
                x=row["band"],
                y=row["total_p_kwh"] + 0.3,
                text=f"<b>{row['total_p_kwh']:.1f}p</b>",
                showarrow=False,
                font=dict(size=11),
                row=1, col=col_idx,
            )

        # Overlay Agile average per band as diamond markers
        if agile_by_date and d in agile_by_date:
            agile_bands = agile_by_date[d]
            agile_x = [b for b in band_labels if b in agile_bands]
            agile_y = [agile_bands[b] for b in agile_x]
            if agile_x:
                fig.add_trace(go.Scatter(
                    x=agile_x, y=agile_y,
                    mode="markers+text",
                    name="Octopus Agile (ex-VAT)",
                    marker=dict(size=12, symbol="diamond", color="#2ecc71",
                                line=dict(width=2, color="#27ae60")),
                    text=[f"{v:.1f}p" for v in agile_y],
                    textposition="top center",
                    textfont=dict(size=10, color="#27ae60"),
                    showlegend=(col_idx == 1),
                    hovertemplate="<b>Agile %{x}</b><br>%{y:.2f}p/kWh (ex-VAT)<extra></extra>",
                ), row=1, col=col_idx)

    fig.update_layout(
        template=_TEMPLATE,
        barmode="stack",
        title=f"{slots_label} — Indicative Cost Stack per Day  "
              "<sup>(EPEX forecast + DUoS + TNUoS + BSUoS + forecast buffer + margin · ex-VAT · "
              "3-day window only · diamonds = Octopus Agile)</sup>",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode="x",
        height=500,
        margin=dict(t=100, b=130),
    )
    fig.update_yaxes(title_text="p/kWh (ex-VAT)", col=1)
    return fig


def _daily_tariff_detail_table(daily_tariffs: list) -> go.Figure:
    """
    Summary table: rows = (day, band), columns = cost components + total.
    """
    all_rows = []
    for d, tariff in daily_tariffs:
        day_str = pd.Timestamp(d).strftime("%a %d %b")
        for _, r in tariff.iterrows():
            all_rows.append({
                "Day":            day_str,
                "Band":           r["band"],
                "Hours":          r["hours"],
                "EPEX wholesale": f"{r['epex_mean_p_kwh']:.2f}p",
                "DUoS":           f"{r['duos_p_kwh']:.2f}p",
                "TNUoS":          f"{r['tnuos_p_kwh']:.2f}p",
                "BSUoS":          f"{r['bsuos_p_kwh']:.2f}p",
                "Fcst buffer":    f"{r['mae_buffer_p_kwh']:.2f}p",
                "Margin":         f"{r['margin_p_kwh']:.2f}p",
                "Total (ex-VAT)": f"{r['total_p_kwh']:.2f}p",
                "_total":         r["total_p_kwh"],
                "_day":           day_str,
            })

    df = pd.DataFrame(all_rows)
    cols = ["Day", "Band", "Hours", "EPEX wholesale", "DUoS", "TNUoS",
            "BSUoS", "Fcst buffer", "Margin", "Total (ex-VAT)"]

    # Alternate row shading by day block
    day_vals = df["_day"].tolist()
    unique_days = list(dict.fromkeys(day_vals))
    row_colours = [
        "#f0f4ff" if day_vals[i] == unique_days[0] else
        ("#fff7f0" if day_vals[i] == unique_days[1] else "#f0fff4")
        for i in range(len(df))
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="left", height=32,
        ),
        cells=dict(
            values=[df[c] for c in cols],
            align="left", font=dict(size=11), height=28,
            fill_color=[row_colours] * len(cols),
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Full Cost Breakdown — 3 Days × All Bands  (p/kWh ex-VAT; +5% for VAT)",
        height=max(300, 110 + 32 * len(df)),
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


def _fig_tariff_vs_agile(
    price_hist: pd.DataFrame,
    hh_pred: pd.DataFrame,
    daily_tariffs_3: list,
    daily_tariffs_4: list | None = None,
    daily_tariffs_3_mult: list | None = None,
) -> go.Figure:
    """
    Our designed tariff: 60-day history (what our tariff would have charged on actual EPEX)
    plus 3-day forecast. All prices inc 5% VAT.
    """
    from app.config import (DUOS_RATES, TNUOS_RATE, BSUOS_BUFFER, SUPPLIER_MULTIPLIER)
    VAT = 1.05

    # ── Hour → DUoS band (consistent with design_tariff band definitions) ────
    def _hour_to_band(h: int) -> str:
        if 16 <= h <= 18:
            return "red"
        elif 7 <= h <= 15 or 19 <= h <= 22:
            return "amber"
        return "green"

    fig = go.Figure()

    # ── Historical: what our tariff would have charged on actual EPEX ────────
    if price_hist is not None and not price_hist.empty:
        ah = price_hist.copy()
        _dt = pd.to_datetime(ah["datetime"], utc=True)
        ah["dt_local"] = _dt.dt.tz_convert("Europe/London").dt.tz_localize(None)
        ah = ah.sort_values("dt_local")
        ah["date"]     = ah["dt_local"].dt.date
        ah["hour"]     = ah["dt_local"].dt.hour
        ah["duos_band"] = ah["hour"].map(_hour_to_band)

        band_means = (
            ah.groupby(["date", "duos_band"])["wholesale_price"]
            .mean()
            .reset_index()
            .rename(columns={"wholesale_price": "epex_band_mean"})
        )
        band_means["duos_p_kwh"] = band_means["duos_band"].map(DUOS_RATES)
        band_means["our_ex_vat"] = (
            band_means["epex_band_mean"] * SUPPLIER_MULTIPLIER
            + band_means["duos_p_kwh"] + TNUOS_RATE + BSUOS_BUFFER
        )
        band_means["our_inc_vat"] = band_means["our_ex_vat"] * VAT

        ah = ah.merge(band_means[["date", "duos_band", "our_inc_vat"]],
                      on=["date", "duos_band"], how="left")

        fig.add_trace(go.Scatter(
            x=ah["dt_local"], y=ah["our_inc_vat"],
            mode="lines", name=f"Our tariff ×{SUPPLIER_MULTIPLIER} (3-band, historical EPEX)",
            line=dict(color="#8e44ad", width=1.5),
            opacity=0.9,
            hovertemplate="%{x|%d %b %H:%M}<br>Ours: %{y:.2f}p/kWh<extra></extra>",
        ))

    # ── Historical Agile retail prices for comparison ────────────────────────
    if price_hist is not None and not price_hist.empty:
        ag = price_hist.copy()
        _dt_ag = pd.to_datetime(ag["datetime"], utc=True)
        ag["dt_local"] = _dt_ag.dt.tz_convert("Europe/London").dt.tz_localize(None)
        ag = ag.sort_values("dt_local")
        if "price_inc_vat" in ag.columns:
            fig.add_trace(go.Scatter(
                x=ag["dt_local"], y=ag["price_inc_vat"],
                mode="lines", name="Octopus Agile (inc VAT)",
                line=dict(color="#2ecc71", width=1.2),
                opacity=0.7,
                hovertemplate="%{x|%d %b %H:%M}<br>Agile: %{y:.2f}p/kWh<extra></extra>",
            ))

    # ── Forward: our designed tariff from forecast ────────────────────────────
    if hh_pred is not None and not hh_pred.empty:
        fwd = hh_pred.copy()
        fwd["hour"] = fwd["datetime_local"].dt.hour
        fwd["dow"]  = fwd["datetime_local"].dt.dayofweek  # 0=Mon

        # Our designed tariff: expand daily_tariffs_3 flat prices per-slot
        tariff_lookup: dict = {}
        band_to_duos = {"peak": "red", "standard": "amber", "off-peak": "green"}
        for d, t in (daily_tariffs_3 or []):
            for _, row in t.iterrows():
                duos_key = band_to_duos.get(row["band"], row["band"])
                tariff_lookup[(d, duos_key)] = row["total_p_kwh"] * VAT  # inc VAT

        fwd["date"]      = fwd["datetime_local"].dt.date
        fwd["duos_band"] = fwd["hour"].map(_hour_to_band)
        fwd["our_3band"] = fwd.apply(
            lambda r: tariff_lookup.get((r["date"], r["duos_band"]), float("nan")),
            axis=1,
        )

        # Multiplier-mode tariff overlay
        if daily_tariffs_3_mult:
            mult_lookup: dict = {}
            for d, t in daily_tariffs_3_mult:
                for _, row in t.iterrows():
                    duos_key = band_to_duos.get(row["band"], row["band"])
                    mult_lookup[(d, duos_key)] = row["total_p_kwh"] * VAT
            fwd["our_3band_mult"] = fwd.apply(
                lambda r: mult_lookup.get((r["date"], r["duos_band"]), float("nan")),
                axis=1,
            )
            fig.add_trace(go.Scatter(
                x=fwd["datetime_local"], y=fwd["our_3band_mult"],
                mode="lines", name="Our 3-band tariff — multiplier (forecast)",
                line=dict(color="#8e44ad", width=2.5, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>Our tariff (×mult): %{y:.2f}p/kWh<extra></extra>",
            ))

        # 4-band if provided
        if daily_tariffs_4:
            band4_to_duos = {"peak": "red", "day": "amber", "evening": "amber", "night": "green"}
            lookup4: dict = {}
            for d, t in daily_tariffs_4:
                for _, row in t.iterrows():
                    dk = band4_to_duos.get(row["band"], "green")
                    # For 4-band, map by actual band name per slot
                    lookup4[(d, row["band"])] = row["total_p_kwh"]

            def _hour_to_4band(h: int) -> str:
                if 0 <= h <= 6:     return "night"
                if 7 <= h <= 15:    return "day"
                if 16 <= h <= 18:   return "peak"
                return "evening"

            fwd["band4"]    = fwd["hour"].map(_hour_to_4band)
            fwd["our_4band"] = fwd.apply(
                lambda r: lookup4.get((r["date"], r["band4"]), float("nan")) * VAT,
                axis=1,
            )
            fig.add_trace(go.Scatter(
                x=fwd["datetime_local"], y=fwd["our_4band"],
                mode="lines", name="Our 4-band tariff (forecast)",
                line=dict(color="#e67e22", width=2.0, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>Our 4-band: %{y:.2f}p/kWh<extra></extra>",
            ))

    # "Now" vertical line
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=now_str, x1=now_str, y0=0, y1=1,
        line=dict(color="#7f8c8d", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=now_str, xref="x", yref="paper", y=1.02, showarrow=False,
        text="now", font=dict(color="#7f8c8d", size=11),
    )

    fig.update_layout(
        template=_TEMPLATE,
        title=(
            "Our Designed Tariff vs Octopus Agile  "
            "<sup>60-day history (actual EPEX) + 3-day forecast · all prices inc 5% VAT · "
            "historical prices use actual EPEX, no forecast buffer</sup>"
        ),
        yaxis_title="p/kWh (inc VAT)",
        hovermode="x unified",
        height=460,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=80, b=110),
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


def _fig_tariff_comparison_table(price_hist: pd.DataFrame) -> go.Figure:
    """
    Small summary table: Agile vs our tariff averages over the historical window,
    broken down by DUoS band (peak / standard / off-peak) + overall.
    """
    from app.config import (DUOS_RATES, TNUOS_RATE, BSUOS_BUFFER,
                             SUPPLIER_MULTIPLIER)
    VAT = 1.05

    if price_hist is None or price_hist.empty:
        fig = go.Figure()
        fig.update_layout(title="Tariff Comparison — no history data", height=220,
                          template=_TEMPLATE)
        return fig

    ah = price_hist.copy()
    _dt = pd.to_datetime(ah["datetime"], utc=True)
    ah["dt_local"] = _dt.dt.tz_convert("Europe/London").dt.tz_localize(None)
    ah["hour"] = ah["dt_local"].dt.hour

    def _band(h):
        if 16 <= h <= 18: return "peak"
        if 7 <= h <= 15 or 19 <= h <= 22: return "standard"
        return "off-peak"

    def _duos(h):
        if 16 <= h <= 18: return "red"
        if 7 <= h <= 15 or 19 <= h <= 22: return "amber"
        return "green"

    ah["band"] = ah["hour"].map(_band)
    ah["duos_band"] = ah["hour"].map(_duos)
    ah["duos_p_kwh"] = ah["duos_band"].map(DUOS_RATES)
    ah["our_inc_vat"] = (ah["wholesale_price"] * SUPPLIER_MULTIPLIER
                         + ah["duos_p_kwh"] + TNUOS_RATE + BSUOS_BUFFER) * VAT

    rows = []
    for band in ["peak", "standard", "off-peak", "all"]:
        sub = ah if band == "all" else ah[ah["band"] == band]
        if sub.empty:
            continue
        agile_avg = sub["price_inc_vat"].mean()
        ours_avg  = sub["our_inc_vat"].mean()
        diff      = ours_avg - agile_avg
        sign      = "▲" if diff >= 0 else "▼"
        rows.append({
            "Band":        band.title() if band != "all" else "Overall",
            "Agile avg":   f"{agile_avg:.2f}p",
            "Our tariff":  f"{ours_avg:.2f}p",
            "Difference":  f"{sign} {abs(diff):.2f}p",
            "_diff":       diff,
        })

    df = pd.DataFrame(rows)
    diff_colours = ["#ffeaea" if d >= 0 else "#eafff0" for d in df["_diff"]]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Band</b>", "<b>Octopus Agile avg</b>",
                    "<b>Our tariff avg</b>", "<b>Difference</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="left", height=32,
        ),
        cells=dict(
            values=[df["Band"], df["Agile avg"], df["Our tariff"], df["Difference"]],
            align="left", font=dict(size=12), height=28,
            fill_color=[["white"] * len(df)] * 3 + [diff_colours],
        ),
    ))
    n = len(df)
    fig.update_layout(
        template=_TEMPLATE,
        title=f"60-Day Tariff Comparison — Agile vs Our Tariff (×{SUPPLIER_MULTIPLIER})  "
              f"(inc 5% VAT · red = ours costs more · green = ours cheaper)",
        height=max(220, 110 + 32 * n),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


# ── Customer simulation charts ─────────────────────────────────────────────────

_SIM_LABELS = {
    "no_shift":     "No shifting<br>(price inelastic)",
    "light_shift":  "Light shifting<br>(dishwasher/washing)",
    "heavy_shift":  "Heavy shifting<br>(smart appliances)",
    "ev_household": "EV household<br>(+4 kWh/day overnight)",
}

_SIM_SUBTITLE = (
    "All-in annual bill including wholesale, network charges, policy levies "
    "(RO/CfD/CM ~3.3p/kWh), supplier operating costs (~1.5p/kWh), standing charge "
    "(~61p/day), and 5% VAT. Ofgem cap shown as reference."
)


def _fig_simulation_customer_bills(sim_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: all-in customer annual bill by scenario with cost breakdown."""
    from app.config import OFGEM_CAP_QUARTER

    # Use the default multiplier (×2.0) for the main chart
    default_mult = 2.0
    sub = sim_df[sim_df["multiplier"] == default_mult]
    if sub.empty:
        sub = sim_df[sim_df["multiplier"] == sim_df["multiplier"].iloc[0]]
        default_mult = sub["multiplier"].iloc[0]

    scenarios = list(dict.fromkeys(sub["scenario"]))
    sub = sub.set_index("scenario").loc[scenarios]
    labels = [_SIM_LABELS.get(s, s) for s in scenarios]

    # Cost components for stacked bars (all inc VAT, in £/year)
    wholesale_network = sub["cust_bill_ours_annual_gbp"].values
    levies = sub["levy_annual_gbp"].values
    opex = sub["opex_annual_gbp"].values
    standing = sub["standing_charge_annual_gbp"].values
    totals = sub["cust_allin_ours_annual_gbp"].values

    fig = go.Figure()

    # Stacked components for our tariff
    fig.add_trace(go.Bar(
        name="Wholesale + Network",
        x=labels, y=wholesale_network,
        marker_color="#9b59b6",
        hovertemplate="<b>%{x}</b><br>Wholesale + Network: £%{y:.0f}/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Policy Levies (RO/CfD/CM)",
        x=labels, y=levies,
        marker_color="#e67e22",
        hovertemplate="<b>%{x}</b><br>Policy levies: £%{y:.0f}/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Supplier Operating Costs",
        x=labels, y=opex,
        marker_color="#f39c12",
        hovertemplate="<b>%{x}</b><br>Supplier opex: £%{y:.0f}/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Standing Charge",
        x=labels, y=standing,
        marker_color="#95a5a6",
        hovertemplate="<b>%{x}</b><br>Standing charge: £%{y:.0f}/yr<extra></extra>",
    ))

    # Add total annotation on top of each stacked bar
    for i, total in enumerate(totals):
        fig.add_annotation(
            x=labels[i], y=total, text=f"<b>£{total:.0f}</b>",
            showarrow=False, yshift=15, font=dict(size=13, color="#2c3e50"),
        )

    # Ofgem cap reference line
    ofgem_caps = sub["ofgem_cap_annual_gbp"].values
    fig.add_trace(go.Scatter(
        name=f"Ofgem Cap ({OFGEM_CAP_QUARTER})",
        x=labels, y=ofgem_caps,
        mode="markers+lines",
        marker=dict(color="#e74c3c", size=10, symbol="diamond"),
        line=dict(color="#e74c3c", dash="dash", width=2),
        hovertemplate="<b>Ofgem Cap</b><br>%{x}: £%{y:.0f}/yr<extra></extra>",
    ))

    # Agile all-in comparison as scatter markers
    agile_totals = sub["cust_allin_agile_annual_gbp"].values
    fig.add_trace(go.Scatter(
        name="Agile All-In",
        x=labels, y=agile_totals,
        mode="markers",
        marker=dict(color="#3498db", size=12, symbol="circle"),
        hovertemplate="<b>Agile</b><br>%{x}: £%{y:.0f}/yr<extra></extra>",
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=(
            f"Customer Annual Electricity Bill — All-In Cost Breakdown  "
            f"(×{default_mult} · inc 5% VAT · annualised from 60-day history)"
        ),
        yaxis_title="Annual Bill (£)",
        barmode="stack",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=520,
        margin=dict(t=100, b=60),
    )
    return fig


def _fig_simulation_supplier_profit(sim_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: supplier annual profit per customer, one bar per multiplier + Agile."""
    multipliers = sorted(sim_df["multiplier"].unique(), reverse=True)
    scenarios   = list(dict.fromkeys(sim_df["scenario"]))
    labels      = [_SIM_LABELS.get(s, s) for s in scenarios]

    mult_colours = {2.1: "#27ae60", 2.0: "#2ecc71", 1.9: "#58d68d"}
    fig = go.Figure()

    for mult in multipliers:
        sub = sim_df[sim_df["multiplier"] == mult].set_index("scenario").loc[scenarios]
        colour = mult_colours.get(mult, "#27ae60")
        fig.add_trace(go.Bar(
            name=f"Our tariff ×{mult}",
            x=labels,
            y=sub["sup_profit_annual_gbp"].values,
            marker_color=colour,
            text=[f"£{v:.0f}" for v in sub["sup_profit_annual_gbp"]],
            textposition="outside",
            hovertemplate=f"×{mult} <b>%{{x}}</b><br>Profit: £%{{y:.0f}}/yr<extra></extra>",
        ))

    agile_vals = sim_df[sim_df["multiplier"] == multipliers[0]].set_index("scenario").loc[scenarios]
    fig.add_trace(go.Bar(
        name="Agile reseller equivalent",
        x=labels,
        y=agile_vals["agile_profit_annual_gbp"].values,
        marker_color="#e67e22",
        text=[f"£{v:.0f}" for v in agile_vals["agile_profit_annual_gbp"]],
        textposition="outside",
        hovertemplate="<b>Agile %{x}</b><br>Profit: £%{y:.0f}/yr<extra></extra>",
    ))

    fig.add_hline(y=0, line_width=1, line_color="#2c3e50")
    fig.update_layout(
        template=_TEMPLATE,
        title="Supplier Annual Profit per Customer  (ex-VAT · revenue − EPEX wholesale cost − network charges)",
        yaxis_title="Profit (£/customer/year)",
        barmode="group",
        legend=dict(orientation="h", y=1.10),
        height=480,
        margin=dict(t=80, b=60),
    )
    return fig


def _fig_simulation_table(sim_df: pd.DataFrame) -> go.Figure:
    """Summary table: all monetary outputs, one row per (multiplier, scenario)."""
    multipliers = sorted(sim_df["multiplier"].unique(), reverse=True)
    scenarios   = list(dict.fromkeys(sim_df["scenario"]))

    # Build rows in multiplier-then-scenario order
    rows_out = []
    for mult in multipliers:
        sub = sim_df[sim_df["multiplier"] == mult].set_index("scenario").loc[scenarios]
        for sc in scenarios:
            r = sub.loc[sc]
            rows_out.append({
                "mult":         f"×{mult}",
                "scenario":     _SIM_LABELS.get(sc, sc).replace("<br>", " "),
                "shift":        f"{int(r['shift_frac'] * 100)}%",
                "our_bill":     r.get("cust_allin_ours_annual_gbp", r["cust_bill_ours_annual_gbp"]),
                "agile_bill":   r.get("cust_allin_agile_annual_gbp", r["cust_bill_agile_annual_gbp"]),
                "saving":       r["cust_saving_vs_agile_gbp"],
                "ofgem_cap":    r.get("ofgem_cap_annual_gbp", float("nan")),
                "our_pkwh":     r.get("effective_allin_p_kwh_ours", r.get("effective_p_kwh_ours",  float("nan"))),
                "agile_pkwh":   r.get("effective_allin_p_kwh_agile", r.get("effective_p_kwh_agile", float("nan"))),
                "our_profit":   r["sup_profit_annual_gbp"],
                "agile_profit": r["agile_profit_annual_gbp"],
            })

    tdf = pd.DataFrame(rows_out)
    saving_colours = ["#eafff0" if v >= 0 else "#ffeaea" for v in tdf["saving"]]
    # Alternate shading by multiplier block
    n_sc = len(scenarios)
    row_colours = []
    for i, mult in enumerate(multipliers):
        shade = "#f0f4ff" if i % 2 == 0 else "#fff7f0"
        row_colours.extend([shade] * n_sc)

    fig = go.Figure(go.Table(
        header=dict(
            values=[
                "<b>Multiplier</b>", "<b>Scenario</b>", "<b>Shift %</b>",
                "<b>Our bill/yr</b>", "<b>Agile bill/yr</b>", "<b>Ofgem cap/yr</b>",
                "<b>Saving vs Agile</b>",
                "<b>Our p/kWh</b>", "<b>Agile p/kWh</b>",
                "<b>Supplier profit/yr</b>",
            ],
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="left", height=34,
        ),
        cells=dict(
            values=[
                tdf["mult"],
                tdf["scenario"],
                tdf["shift"],
                [f"£{v:.0f}" for v in tdf["our_bill"]],
                [f"£{v:.0f}" for v in tdf["agile_bill"]],
                [f"£{v:.0f}" for v in tdf["ofgem_cap"]],
                [f"{'▲' if v >= 0 else '▼'} £{abs(v):.0f}" for v in tdf["saving"]],
                [f"{v:.1f}p" for v in tdf["our_pkwh"]],
                [f"{v:.1f}p" for v in tdf["agile_pkwh"]],
                [f"£{v:.0f}" for v in tdf["our_profit"]],
            ],
            align="left", font=dict(size=12), height=30,
            fill_color=[
                row_colours,
                row_colours,
                row_colours,
                row_colours,
                row_colours,
                row_colours,
                saving_colours,
                row_colours,
                row_colours,
                row_colours,
            ],
        ),
    ))
    n = len(tdf)
    fig.update_layout(
        template=_TEMPLATE,
        title="Customer Simulation Summary  (all-in bills inc levies, standing charge, opex, VAT · ▲ = ours cheaper)",
        height=max(300, 120 + 32 * n),
        margin=dict(t=60, b=10, l=0, r=0),
    )
    return fig


def _fig_simulation_unit_rate(sim_df: pd.DataFrame) -> go.Figure:
    """
    Effective all-in unit rate (p/kWh) per scenario vs Ofgem price cap reference.
    Includes all cost components (wholesale, network, levies, opex, standing charge amortised).
    """
    from app.config import OFGEM_UNIT_RATE_P_KWH, OFGEM_CAP_QUARTER, OFGEM_STANDING_CHARGE_P_DAY

    multipliers = sorted(sim_df["multiplier"].unique(), reverse=True)
    scenarios   = list(dict.fromkeys(sim_df["scenario"]))
    labels      = [_SIM_LABELS.get(s, s) for s in scenarios]

    mult_colours = {2.1: "#8e44ad", 2.0: "#9b59b6"}
    fig = go.Figure()

    for mult in multipliers:
        sub = sim_df[sim_df["multiplier"] == mult].set_index("scenario").loc[scenarios]
        colour = mult_colours.get(mult, "#8e44ad")
        # All-in effective rate = total bill / annual kWh (inc standing charge amortised)
        allin_rate = sub["effective_allin_p_kwh_ours"].values if "effective_allin_p_kwh_ours" in sub.columns else sub["effective_p_kwh_ours"].values
        fig.add_trace(go.Bar(
            name=f"Our tariff ×{mult} (all-in)",
            x=labels,
            y=allin_rate,
            marker_color=colour,
            text=[f"{v:.1f}p" for v in allin_rate],
            textposition="outside",
            hovertemplate=f"×{mult} <b>%{{x}}</b><br>All-in rate: %{{y:.1f}}p/kWh<extra></extra>",
        ))

    agile_vals = sim_df[sim_df["multiplier"] == multipliers[0]].set_index("scenario").loc[scenarios]
    agile_allin = agile_vals["effective_allin_p_kwh_agile"].values if "effective_allin_p_kwh_agile" in agile_vals.columns else agile_vals["effective_p_kwh_agile"].values
    fig.add_trace(go.Bar(
        name="Octopus Agile (all-in)",
        x=labels,
        y=agile_allin,
        marker_color="#3498db",
        text=[f"{v:.1f}p" for v in agile_allin],
        textposition="outside",
        hovertemplate="<b>Agile %{x}</b><br>All-in rate: %{y:.1f}p/kWh<extra></extra>",
    ))

    # Ofgem cap reference — unit rate + standing charge amortised over annual kWh
    # Standing charge amortised varies by scenario due to different annual_kwh
    for i, sc in enumerate(scenarios):
        sc_row = agile_vals.loc[sc]
        annual_kwh = sc_row.get("annual_kwh", 2920)
        standing_amortised = OFGEM_STANDING_CHARGE_P_DAY * 365 / annual_kwh
        ofgem_allin = OFGEM_UNIT_RATE_P_KWH + standing_amortised

    fig.add_hline(
        y=OFGEM_UNIT_RATE_P_KWH + OFGEM_STANDING_CHARGE_P_DAY * 365 / 2920,
        line_dash="dash", line_color="#e74c3c", line_width=2,
        annotation_text=f"Ofgem cap {OFGEM_CAP_QUARTER}: ~{OFGEM_UNIT_RATE_P_KWH + OFGEM_STANDING_CHARGE_P_DAY * 365 / 2920:.1f}p/kWh all-in (8 kWh/day)",
        annotation_position="top right",
        annotation_font=dict(color="#e74c3c", size=11),
    )

    fig.update_layout(
        template=_TEMPLATE,
        title=(
            "Effective All-In Unit Rate (p/kWh) inc Standing Charge  "
            f"<sup>Wholesale + network + levies + opex + standing charge ÷ annual kWh · inc 5% VAT</sup>"
        ),
        yaxis_title="p/kWh (all-in)",
        barmode="group",
        legend=dict(orientation="h", y=1.12),
        height=460,
        margin=dict(t=90, b=60),
    )
    return fig


# ── Ensemble / LightGBM charts ────────────────────────────────────────────────

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
    daily_tariffs_3=None,
    daily_tariffs_4=None,
    daily_tariffs_3_mult=None,
    daily_tariffs_4_mult=None,
    daily_predictions=None,
    price_hist=None,
    leadtime_detail_df=None,
    leadtime_metrics=None,
    hh_hourly_profile=None,
    sim_df=None,
    ensemble=None,
    hh_ensemble=None,
    wfcv_detail=None,
    wfcv_metrics=None,
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

    # Ensemble charts
    fig_lgbm_imp_daily = _fig_lgbm_importance(ensemble, "Daily") if ensemble else None
    fig_lgbm_imp_hh = _fig_lgbm_importance(hh_ensemble, "Half-Hourly") if hh_ensemble else None
    fig_wfcv = _fig_walkforward_cv(
        wfcv_detail if wfcv_detail is not None else pd.DataFrame(),
        wfcv_metrics or {},
    ) if wfcv_detail is not None and not wfcv_detail.empty else None
    fig_pred_interval = _fig_prediction_intervals(daily_predictions)

    has_tariff = daily_tariffs_3_mult is not None and len(daily_tariffs_3_mult) > 0
    if has_tariff:
        from app.config import SUPPLIER_MULTIPLIER

        # Build Agile average ex-VAT price per band per forecast day for overlay
        agile_by_date: dict = {}
        if price_hist is not None and not price_hist.empty:
            _ag = price_hist.copy()
            _ag_dt = pd.to_datetime(_ag["datetime"], utc=True)
            _ag["dt_local"] = _ag_dt.dt.tz_convert("Europe/London").dt.tz_localize(None)
            _ag["_date"] = _ag["dt_local"].dt.date
            _ag["_hour"] = _ag["dt_local"].dt.hour

            def _hour_to_3band(h):
                if 16 <= h <= 18: return "peak"
                if 7 <= h <= 15 or 19 <= h <= 22: return "standard"
                return "off-peak"

            _ag["_band"] = _ag["_hour"].map(_hour_to_3band)
            for d, grp in _ag.groupby("_date"):
                band_avg = grp.groupby("_band")["price_ex_vat"].mean().to_dict()
                agile_by_date[d] = band_avg

        mult_label = f"3-Band Tariff (×{SUPPLIER_MULTIPLIER} wholesale)"
        fig_tariff_3_mult = _fig_daily_tariff(daily_tariffs_3_mult, mult_label,
                                               agile_by_date=agile_by_date)
        fig_tariff_4_mult = _fig_daily_tariff(daily_tariffs_4_mult,
                                               mult_label.replace("3-Band", "4-Band"),
                                               agile_by_date=agile_by_date) \
                            if daily_tariffs_4_mult else None
        # Detail table from multiplier tariff
        fig_tariff_detail = _daily_tariff_detail_table(daily_tariffs_3_mult)
        fig_tariff_vs_agile = _fig_tariff_vs_agile(
            price_hist, hh_pred, daily_tariffs_3, daily_tariffs_4,
            daily_tariffs_3_mult=daily_tariffs_3_mult,
        )
        fig_tariff_comparison = _fig_tariff_comparison_table(price_hist)

    from app.config import OFGEM_UNIT_RATE_P_KWH, OFGEM_CAP_QUARTER  # noqa: F401 (used in HTML fstring)
    has_simulation = sim_df is not None and not sim_df.empty
    if has_simulation:
        fig_sim_bills     = _fig_simulation_customer_bills(sim_df)
        fig_sim_profit    = _fig_simulation_supplier_profit(sim_df)
        fig_sim_table     = _fig_simulation_table(sim_df)
        fig_sim_unit_rate = _fig_simulation_unit_rate(sim_df)
        sim_n_days = len(price_hist["datetime"].unique()) if price_hist is not None and not price_hist.empty else 60

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
  {"" if PUBLIC_MODE else '<a href="#tariff">Tariff Design</a>'}
  {"" if PUBLIC_MODE else '<a href="#simulation">Customer Simulation</a>'}
  <a href="#drivers">Price Drivers</a>
  <a href="#history">History</a>
  <a href="#accuracy">Model Accuracy</a>
  {"" if PUBLIC_MODE else '<a href="#model">Model Detail</a>'}
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
  {"" if PUBLIC_MODE or not has_tariff else f"""
  <div class="stat-card">
    <div class="label">Today Peak Rate (3-band)</div>
    <div class="value">{daily_tariffs_3[0][1][daily_tariffs_3[0][1]['band']=='peak']['total_p_kwh'].values[0]:.1f}p</div>
    <div class="sub">16:00–19:00 · ex-VAT · today</div>
  </div>
  <div class="stat-card">
    <div class="label">Today Off-Peak Rate (3-band)</div>
    <div class="value">{daily_tariffs_3[0][1][daily_tariffs_3[0][1]['band']=='off-peak']['total_p_kwh'].values[0]:.1f}p</div>
    <div class="sub">23:00–07:00 · ex-VAT · today</div>
  </div>
  """}
</div>

<div id="forecast">
  {_section("7-Day Forecast",
            f"Predicted half-hourly {EPEX} wholesale prices for the next 7 days. "
            f"Shaded bands = peak rate period (16:00–19:00). "
            f"<strong>{hh_blend_desc}</strong>. "
            f"Based on {NWP} weather forecast + current gas prices + yesterday's per-slot {EPEX} price as an autoregressive anchor. "
            f"{'Network charges (' + _t('DUoS','Distribution Use of System') + '/' + _t('TNUoS','Transmission Network Use of System') + ') and supplier margin are not included — these are raw wholesale costs. See the Tariff Design section below for derived retail prices.' if not PUBLIC_MODE else 'These are raw wholesale costs — network charges and supplier margin are not included.'}")}
</div>
<div class="charts">
  {_div(fig_forecast)}
  {"" if fig_daily_forecast is None else _div(fig_daily_forecast)}
</div>

{"" if PUBLIC_MODE or not has_tariff else f'''
<div id="tariff">
  {_section("Tariff Design",
            f"Indicative time-of-use tariff for the next 3 days, priced from each day's {EPEX} wholesale forecast. "
            f"Beyond 3 days {NWP} forecast skill degrades significantly. "
            f"Each band = forecast {EPEX} mean × wholesale multiplier (GB-wide wholesale + implicit margin) + "
            f"{_t('DUoS','Distribution Use of System')} ({SPD} Central Scotland LV {_t('HH','Half-Hourly metered')} — region-specific) + "
            f"{_t('TNUoS','Transmission Network Use of System')} (GB-wide) + "
            f"{_t('BSUoS','Balancing Use of System')} buffer (GB-wide) + forecast {MAE} buffer. "
            f"Only DUoS varies by DNO region — verify rates from {SPD} LC14 Charging Statement. "
            f"All prices ex-{VAT}; add 5% for the domestic rate.")}
</div>
<div class="charts">
  {_div(fig_tariff_vs_agile)}
  {_div(fig_tariff_comparison)}
  {_div(fig_tariff_3_mult)}
  {_div(fig_tariff_4_mult) if fig_tariff_4_mult else ""}
  {_div(fig_tariff_detail)}
</div>
'''}
{"" if PUBLIC_MODE or not has_simulation else f'''
<div id="simulation">
  {_section("Customer Behaviour Simulation",
            f"Modelled annual bills for a UK average household (8 kWh/day base), "
            f"using Elexon PC1 seasonal load profile (winter/summer, weekday/weekend). "
            f"<strong>All-in annual bill</strong>: wholesale + network charges + policy levies "
            f"(RO/CfD/CM ~3.3p/kWh) + supplier opex (~1.5p/kWh) + standing charge (~61p/day) + 5% VAT. "
            f"Ofgem {OFGEM_CAP_QUARTER} price cap shown for reference. "
            f"Supplier cost = actual {EPEX} slot price × consumption (trading bought ahead at spot). "
            f"Annualised from {sim_n_days}-slot historical window.")}
</div>
<div class="charts">
  {_div(fig_sim_unit_rate)}
  {_div(fig_sim_bills)}
  {_div(fig_sim_profit)}
  {_div(fig_sim_table)}
</div>
'''}
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

{"" if PUBLIC_MODE else f'''
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
  {"" if PUBLIC_MODE else f"DUoS: {SPD} Central Scotland (region-specific; verify from LC14 Charging Statement) ·"}
  Weather: Open-Meteo UK average (6 sites: Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff) ·
  Solar: Sheffield Solar {PVLIVE} (GB national generation) ·
  Demand: Elexon {BMRS} {INDO} ·
  Generation mix: Elexon {BMRS} {FUELHH} (wind, gas, nuclear, pumped hydro, hydro, interconnectors) ·
  Day-ahead prices: Elexon {BMRS} {MID} ({EPEX} SPOT GB / {APXMIDP}) ·
  Commodity: Yahoo Finance ({TTF} gas, Brent crude)
  {"" if PUBLIC_MODE else f"· Network charges retained in price; {VAT} removed"}
</footer>

</body>
</html>"""

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH
