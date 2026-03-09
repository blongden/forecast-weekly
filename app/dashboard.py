"""
HTML dashboard generation using Plotly.
Produces a single self-contained dashboard.html file.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import BASE_DIR
from app.analysis import ALL_FEATURE_LABELS

DASHBOARD_PATH = BASE_DIR / "dashboard.html"

_TEMPLATE     = "plotly_white"
_PRICE_COL    = "#e74c3c"
_WHOLESALE    = "#c0392b"
_PEAK_COL     = "rgba(255,200,100,0.18)"
_FORECAST_COL = "#2980b9"
_DEMAND_COL   = "#8e44ad"
_SOLAR_COL    = "#f39c12"
_COLOURS      = ["#8e44ad", "#e74c3c", "#f39c12", "#27ae60"]


# ── Individual chart builders ──────────────────────────────────────────────────

def _fig_halfhourly_forecast(hh_pred: pd.DataFrame, hist_mean: float) -> go.Figure:
    """7-day half-hourly forecast as a line chart with peak shading."""
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

    fig.add_trace(go.Scatter(
        x=hh_pred["datetime_local"], y=hh_pred["predicted_ex_vat"],
        mode="lines", name="Predicted ex-VAT",
        line=dict(color=_FORECAST_COL, width=1.5),
        hovertemplate="%{x|%a %d %b %H:%M}<br>ex-VAT: %{y:.2f}p/kWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hh_pred["datetime_local"], y=hh_pred["predicted_inc_vat"],
        mode="lines", name="Predicted incl. VAT",
        line=dict(color=_PRICE_COL, width=1, dash="dot"),
        hovertemplate="%{x|%a %d %b %H:%M}<br>incl. VAT: %{y:.2f}p/kWh<extra></extra>",
    ))

    fig.add_hline(
        y=hist_mean, line_dash="dash", line_color="grey", line_width=1,
        annotation_text=f"12-month avg {hist_mean:.2f}p",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    fig.update_layout(
        template=_TEMPLATE,
        title="7-Day Half-Hourly Price Forecast  (shaded = peak 16:00–19:00)",
        yaxis_title="Predicted Price (p/kWh)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
    )
    return fig


def _fig_history(df: pd.DataFrame) -> go.Figure:
    """12-month daily price history with wholesale reference."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_price_ex_vat"],
        mode="lines", name="Agile ex-VAT",
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        line=dict(color=_PRICE_COL, width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>ex-VAT: %{y:.2f}p/kWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_wholesale_price"],
        mode="lines", name="Wholesale est.",
        line=dict(color=_WHOLESALE, width=1, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Wholesale: %{y:.2f}p/kWh<extra></extra>",
    ))

    mean_ex = df["avg_price_ex_vat"].mean()
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
        x=df["date"], y=df["avg_price_ex_vat"],
        mode="lines", name="Agile ex-VAT (p/kWh)",
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
        title="Agile Price vs Gas & Oil  (TTF gas is the primary driver of UK wholesale electricity)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="Agile Price ex-VAT (p/kWh)", secondary_y=False)
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
        x=df["date"], y=df["avg_price_ex_vat"],
        mode="lines", name="Agile ex-VAT (p/kWh)",
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
        title="Agile Price vs GB Demand & Solar Generation  "
              "(high demand → higher price; high solar → lower price)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="Agile Price ex-VAT (p/kWh)", secondary_y=False)
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
        x=df["date"], y=df["avg_price_ex_vat"],
        mode="lines", name="Agile ex-VAT (p/kWh)",
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
        title="Agile Price vs GB Generation Mix  "
              "(gas ↑ → price up; wind / imports ↑ → price down)",
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
        height=420,
        margin=dict(t=90, b=40),
    )
    fig.update_yaxes(title_text="Agile Price ex-VAT (p/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Generation (GW)", secondary_y=True)
    return fig


def _fig_scatter(df: pd.DataFrame) -> go.Figure:
    """
    2×2 scatter plots: price vs the four most informative predictors.
    Demand leads as it has the highest correlation (r ≈ +0.52).
    """
    var_list = [
        ("demand_mw",          "GB Demand (MW)"),
        ("wind_gen_mw",        "GB Wind Generation (MW)"),
        ("solar_gw",           "Solar Generation (GW, GB actual)"),
        ("temperature_2m",     "Temperature (°C, UK avg)"),
    ]
    # Only include variables that are present and have data
    var_list = [(v, l) for v, l in var_list
                if v in df.columns and df[v].notna().any()]

    n = len(var_list)
    cols = 2
    rows = (n + 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[l for _, l in var_list])

    for idx, ((var, label), colour) in enumerate(zip(var_list, _COLOURS)):
        r, c = divmod(idx, cols)
        valid = df[["avg_price_ex_vat", var]].dropna()
        x = valid[var].values
        y = valid["avg_price_ex_vat"].values

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
        fig.update_yaxes(title_text="Price ex-VAT (p/kWh)", row=r + 1, col=c + 1)

    fig.update_layout(
        template=_TEMPLATE,
        title="Price vs Key Predictors  (demand is the strongest single signal)",
        height=680,
        margin=dict(t=80, b=40),
    )
    return fig


def _fig_backtest(backtest_df: pd.DataFrame, metrics: dict,
                  verifiable_df=None) -> go.Figure:
    """Daily hold-out backtest: actual vs predicted."""
    fig = go.Figure()

    if not backtest_df.empty:
        fig.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["avg_price_ex_vat"],
            mode="lines+markers", name="Actual",
            line=dict(color=_PRICE_COL, width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.2f}p/kWh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["predicted_ex_vat_p_kwh"],
            mode="lines+markers", name="Predicted (out-of-sample)",
            line=dict(color=_FORECAST_COL, width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:.2f}p/kWh<extra></extra>",
        ))

    if verifiable_df is not None and len(verifiable_df) > 0:
        vdf = pd.DataFrame(
            verifiable_df,
            columns=["predicted_on", "date", "predicted_ex_vat_p_kwh",
                     "predicted_inc_vat_p_kwh", "actual_ex_vat", "actual_inc_vat"],
        )
        vdf["date"] = pd.to_datetime(vdf["date"])
        latest_pred_on = vdf["predicted_on"].max()
        vdf_latest = vdf[vdf["predicted_on"] == latest_pred_on]
        if not vdf_latest.empty:
            fig.add_trace(go.Scatter(
                x=vdf_latest["date"], y=vdf_latest["predicted_ex_vat_p_kwh"],
                mode="lines+markers", name=f"Stored forecast (made {latest_pred_on})",
                line=dict(color="#27ae60", width=2, dash="dot"),
                marker=dict(size=7, symbol="star"),
                hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:.2f}p/kWh<extra></extra>",
            ))

    mae_str  = f"{metrics.get('mae', 0):.2f}p"
    rmse_str = f"{metrics.get('rmse', 0):.2f}p"
    mape_str = f"{metrics.get('mape', 0):.1f}%"
    hold     = metrics.get("holdout_days", 30)
    train    = metrics.get("train_days", "?")

    fig.update_layout(
        template=_TEMPLATE,
        title=(f"Daily Prediction Accuracy — {hold}-Day Hold-Out  "
               f"(MAE {mae_str} · RMSE {rmse_str} · MAPE {mape_str})<br>"
               f"<sup>Trained on {train} days, tested on following {hold} days "
               f"using actual weather (best-case accuracy ceiling)</sup>"),
        yaxis_title="Price ex-VAT (p/kWh)",
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
    peak_mae = hh_bt_metrics.get("peak_mae", 0)
    op_mae   = hh_bt_metrics.get("offpeak_mae", 0)
    hold     = hh_bt_metrics.get("holdout_days", 30)

    fig.update_layout(
        template=_TEMPLATE,
        title=(f"Half-Hourly Prediction Accuracy — {hold}-Day Hold-Out  "
               f"(MAE {mae:.2f}p · Peak MAE {peak_mae:.2f}p · "
               f"Off-peak MAE {op_mae:.2f}p)"),
        yaxis_title="Price ex-VAT (p/kWh)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
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
    colours = [
        "#2ecc71" if v["r"] < -0.4 else ("#f39c12" if v["r"] < 0 else "#e74c3c")
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
            fill_color=[["white"] * len(labels), colours,
                        ["white"] * len(labels), ["white"] * len(labels)],
            align="left", font=dict(size=12), height=32,
        ),
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Pearson Correlations with Ex-VAT Price",
        height=max(200, 60 + 32 * len(labels)),
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
        height=max(200, 60 + 32 * len(cell_metrics)),
        margin=dict(t=60, b=10, l=0, r=0),
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
) -> Path:
    """Build and write the HTML dashboard. Returns the path."""

    hist_mean = df_daily["avg_price_ex_vat"].mean()
    updated   = datetime.now().strftime("%d %b %Y %H:%M")

    min_day = df_daily.loc[df_daily["avg_price_ex_vat"].idxmin(), "date"].strftime("%d %b")
    max_day = df_daily.loc[df_daily["avg_price_ex_vat"].idxmax(), "date"].strftime("%d %b")

    # Use out-of-sample MAE for stat cards — more meaningful than in-sample R²
    daily_mae_str = (f"{backtest_metrics['mae']:.2f}p"
                     if backtest_metrics else "n/a")
    hh_mae_str    = (f"{hh_backtest_metrics['mae']:.2f}p"
                     if hh_backtest_metrics else "n/a")

    # Today's forecast headline
    today_slots = hh_pred[hh_pred["datetime_local"].dt.date == hh_pred["datetime_local"].dt.date.min()]
    peak_slots    = today_slots[today_slots["predicted_ex_vat"] == today_slots[
        today_slots["datetime_local"].dt.hour.between(16, 18)]["predicted_ex_vat"].max()] if not today_slots.empty else None
    offpeak_mean  = today_slots[~today_slots["datetime_local"].dt.hour.between(16, 18)]["predicted_ex_vat"].mean() if not today_slots.empty else None

    def _div(fig, full_width=True) -> str:
        cls = "chart-full" if full_width else "chart-half"
        inner = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={"displayModeBar": False})
        return f'<div class="{cls}">{inner}</div>'

    def _section(title: str, subtitle: str = "") -> str:
        sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
        return f'<div class="section-header"><h2>{title}</h2>{sub}</div>'

    fig_forecast    = _fig_halfhourly_forecast(hh_pred, hist_mean)
    fig_history     = _fig_history(df_daily)
    fig_commodity   = _fig_commodity(df_daily)
    fig_demand      = _fig_demand_solar(df_daily)
    fig_generation  = _fig_generation_mix(df_daily)
    fig_scatter     = _fig_scatter(df_daily)
    fig_corr        = _corr_table(correlations)
    fig_model       = _model_table(r2_daily, r2_hh, model, feature_cols)

    bt_df      = backtest_df      if backtest_df      is not None else pd.DataFrame()
    bt_metrics = backtest_metrics if backtest_metrics is not None else {}
    fig_backtest    = _fig_backtest(bt_df, bt_metrics, verifiable_df)
    fig_hh_backtest = _fig_hh_backtest(hh_backtest_df, hh_backtest_metrics or {})

    offpeak_str = f"{offpeak_mean:.2f}p" if offpeak_mean is not None else "—"

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
  </style>
</head>
<body>

<header>
  <h1>⚡ UK Electricity Price Analysis — Agile Tariff Region N</h1>
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
    <div class="sub">ex-VAT per kWh</div>
  </div>
  <div class="stat-card">
    <div class="label">Incl. VAT</div>
    <div class="value">{df_daily["avg_price_inc_vat"].mean():.2f}p</div>
    <div class="sub">per kWh</div>
  </div>
  <div class="stat-card">
    <div class="label">Today — Off-Peak Forecast</div>
    <div class="value">{offpeak_str}</div>
    <div class="sub">predicted ex-VAT avg</div>
  </div>
  <div class="stat-card">
    <div class="label">Cheapest Day (12 months)</div>
    <div class="value">{df_daily["avg_price_ex_vat"].min():.2f}p</div>
    <div class="sub">{min_day}</div>
  </div>
  <div class="stat-card">
    <div class="label">Most Expensive Day</div>
    <div class="value">{df_daily["avg_price_ex_vat"].max():.2f}p</div>
    <div class="sub">{max_day}</div>
  </div>
  <div class="stat-card">
    <div class="label">Forecast Accuracy (MAE)</div>
    <div class="value">{daily_mae_str}</div>
    <div class="sub">daily avg · 30-day hold-out</div>
  </div>
  <div class="stat-card">
    <div class="label">Half-Hourly Accuracy</div>
    <div class="value">{hh_mae_str}</div>
    <div class="sub">per slot · 30-day hold-out</div>
  </div>
</div>

<div id="forecast">
  {_section("7-Day Forecast",
            "Predicted half-hourly Agile prices for the next 7 days. "
            "Shaded bands = peak rate period (16:00–19:00). "
            "Based on weather forecast + current gas prices + typical demand for each time slot.")}
</div>
<div class="charts">
  {_div(fig_forecast)}
</div>

<div id="drivers">
  {_section("Price Drivers",
            "Gas prices set the floor for UK electricity costs. "
            "High demand pushes prices up; high wind and solar generation pushes them down.")}
</div>
<div class="charts">
  {_div(fig_commodity)}
  {_div(fig_demand)}
  {_div(fig_generation)}
</div>

<div id="history">
  {_section("12-Month Price History",
            "Daily average Agile tariff price over the past 12 months. "
            "Dotted line shows estimated wholesale cost (ex-VAT ÷ distribution multiplier).")}
</div>
<div class="charts">
  {_div(fig_history)}
</div>

<div id="accuracy">
  {_section("Model Accuracy",
            "Out-of-sample hold-out test: model trained on older data, "
            "then tested on the most recent 30 days it has never seen. "
            "Actual weather is used as a perfect-forecast proxy to show the accuracy ceiling.")}
</div>
<div class="charts">
  {_div(fig_backtest)}
  {_div(fig_hh_backtest)}
</div>

<div id="model">
  {_section("Model Detail",
            "Correlations and regression coefficients for the underlying statistical models. "
            "Coefficients are standardised (β) so their magnitude reflects relative importance.")}
</div>
<div class="charts">
  {_div(fig_corr, full_width=False)}
  {_div(fig_model, full_width=False)}
  {_div(fig_scatter)}
</div>

<footer>
  Octopus Agile AGILE-24-10-01 · Region N (Southern Scotland) ·
  Weather: Open-Meteo UK average (6 sites: Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff) ·
  Solar: Sheffield Solar PV_Live API (GB national generation) ·
  Demand: Elexon BMRS INDO (GB Initial National Demand Outturn) ·
  Generation mix: Elexon BMRS FUELHH (wind, gas, nuclear, interconnectors) ·
  Commodity: Yahoo Finance (TTF gas, Brent crude) ·
  Network charges retained in price; VAT removed
</footer>

</body>
</html>"""

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH
