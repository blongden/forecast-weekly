"""
Chart generation — saves PNGs to the charts/ directory.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats

from app.config import CHARTS_DIR
from app.analysis import WEATHER_VARS

COLOURS = ["#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
PRICE_COLOUR      = "#e74c3c"
WHOLESALE_COLOUR  = "#c0392b"
PRICE_LABEL       = "Price ex-VAT (p/kWh)"


def _apply_date_fmt(ax, df):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_time_series(df, title_suffix=""):
    n = len(WEATHER_VARS) + 1
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    fig.suptitle(
        f"UK Electricity Wholesale Price & Weather — Scotland Central Belt{title_suffix}",
        fontsize=13, fontweight="bold",
    )

    ax0 = axes[0]
    ax0.plot(df["date"], df["avg_epex_p_kwh"], color=PRICE_COLOUR,
             linewidth=1.2, label="EPEX wholesale (p/kWh)")
    ax0.fill_between(df["date"], df["avg_epex_p_kwh"], alpha=0.12, color=PRICE_COLOUR)
    mean_ex = df["avg_epex_p_kwh"].mean()
    ax0.axhline(mean_ex, color=PRICE_COLOUR, linestyle="--", linewidth=0.8, alpha=0.7,
                label=f"Mean EPEX = {mean_ex:.2f} p/kWh")
    ax0.set_ylabel(PRICE_LABEL)
    ax0.set_title("EPEX Wholesale Price  (day-ahead, before network charges)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    for i, ((var, label), colour) in enumerate(zip(WEATHER_VARS.items(), COLOURS)):
        ax = axes[i + 1]
        ax.plot(df["date"], df[var], color=colour, linewidth=1.0)
        ax.fill_between(df["date"], df[var], alpha=0.15, color=colour)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    _apply_date_fmt(axes[-1], df)
    plt.tight_layout()
    path = CHARTS_DIR / "time_series.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scatter(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "EPEX Wholesale Price vs Weather Variables — Scatter Plots with Trend Lines",
        fontsize=13, fontweight="bold",
    )

    for ax, (var, label), colour in zip(axes.flat, WEATHER_VARS.items(), COLOURS):
        x = df[var].values
        y = df["avg_epex_p_kwh"].values
        ax.scatter(x, y, alpha=0.35, s=18, color=colour, edgecolors="none")

        m, b, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
                label=f"r = {r:+.3f}")

        ax.set_xlabel(label)
        ax.set_ylabel(PRICE_LABEL)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "scatter_plots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_forecast(predictions, historical_mean: float):
    fig, ax = plt.subplots(figsize=(9, 5))
    dates   = predictions["date"].dt.strftime("%a\n%d %b")
    vals    = predictions["predicted_epex_p_kwh"].values
    colours = ["#e74c3c" if v > historical_mean else "#2ecc71" for v in vals]

    bars = ax.bar(dates, vals, color=colours, edgecolor="white",
                  linewidth=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}p",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(historical_mean, color="grey", linestyle="--", linewidth=1,
               label=f"12-month avg EPEX = {historical_mean:.2f}p")
    ax.set_title(
        "Predicted EPEX Wholesale Price — Next 7 Days\n"
        "(Ridge Regression on Weather + Gas + Wind; before network charges and margin)",
        fontsize=11,
    )
    ax.set_ylabel("Predicted EPEX Wholesale (p/kWh)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = CHARTS_DIR / "forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
