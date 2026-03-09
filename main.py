#!/usr/bin/env python3
"""
Energy Analysis App — main entry point.

Usage
-----
    python main.py           # full run: update DB + analyse + charts
    python main.py update    # only fetch new data (no charts/analysis)
    python main.py analyse   # only run analysis on existing DB data
    python main.py status    # show what's in the DB
"""
import sys
from datetime import date, timedelta

from app import db, octopus, weather, gas, analysis, charts, dashboard, pvlive, demand, supply, midprice
from app.config import WIND_SITES, UK_WEATHER_SITES


HISTORY_DAYS = 365  # how many days back to maintain


def cmd_update(verbose=True) -> None:
    """Fetch any missing price and weather data into the DB."""
    today      = date.today()
    start_date = today - timedelta(days=HISTORY_DAYS)

    # ── Prices ────────────────────────────────────────────────────────────────
    gaps = octopus.missing_price_ranges(start_date, today)
    if gaps:
        for gap_start, gap_end in gaps:
            if verbose:
                print(f"[Octopus]  Fetching prices {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = octopus.fetch_prices(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Octopus]  Prices up to date — nothing to fetch.")

    # ── UK weather sites (temperature, solar, precipitation) ──────────────────
    for site_id, site_info in UK_WEATHER_SITES.items():
        gaps = weather.missing_uk_site_ranges(site_id, start_date, today)
        if gaps:
            for gap_start, gap_end in gaps:
                if verbose:
                    print(f"[Weather/{site_id:<10}] Fetching {gap_start} → {gap_end} …",
                          end="", flush=True)
                n = weather.fetch_uk_site_historical(
                    site_id, site_info["lat"], site_info["lon"], gap_start, gap_end)
                if verbose:
                    print(f" {n} records stored.")
        else:
            if verbose:
                print(f"[Weather/{site_id:<10}] Up to date — nothing to fetch.")

    # ── Solar generation (Sheffield Solar PV_Live) ────────────────────────────
    s_gaps = pvlive.missing_solar_ranges(start_date, today)
    if s_gaps:
        for gap_start, gap_end in s_gaps:
            if verbose:
                print(f"[Solar]    Fetching GB solar generation {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = pvlive.fetch_historical(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Solar]    Solar generation up to date — nothing to fetch.")

    # ── Commodity prices (Brent crude + TTF gas) ──────────────────────────────
    g_gaps = gas.missing_commodity_ranges(start_date, today)
    if g_gaps:
        for gap_start, gap_end in g_gaps:
            if verbose:
                print(f"[Gas/Oil]  Fetching commodity prices {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = gas.fetch_commodity(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Gas/Oil]  Commodity prices up to date — nothing to fetch.")

    # ── GB demand (Elexon BMRS INDO) ─────────────────────────────────────────
    d_gaps = demand.missing_demand_ranges(start_date, today)
    if d_gaps:
        for gap_start, gap_end in d_gaps:
            if verbose:
                print(f"[Demand]   Fetching GB demand {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = demand.fetch_demand(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Demand]   GB demand up to date — nothing to fetch.")

    # ── GB generation mix (Elexon BMRS FUELHH) ───────────────────────────────
    sup_gaps = supply.missing_supply_ranges(start_date, today)
    if sup_gaps:
        for gap_start, gap_end in sup_gaps:
            if verbose:
                print(f"[Supply]   Fetching GB generation mix {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = supply.fetch_supply(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Supply]   GB generation mix up to date — nothing to fetch.")

    # ── EPEX SPOT GB day-ahead prices (Elexon BMRS MID / APXMIDP) ───────────────
    mp_gaps = midprice.missing_midprice_ranges(start_date, today)
    if mp_gaps:
        for gap_start, gap_end in mp_gaps:
            if verbose:
                print(f"[MidPrice] Fetching EPEX day-ahead prices {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = midprice.fetch_midprice(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[MidPrice] EPEX day-ahead prices up to date — nothing to fetch.")

    # ── Offshore wind site weather (100m wind speed) ──────────────────────────
    for site_id, site_info in WIND_SITES.items():
        gaps = weather.missing_wind_site_ranges(site_id, start_date, today)
        if gaps:
            for gap_start, gap_end in gaps:
                if verbose:
                    print(f"[Wind/{site_id:<12}] Fetching {gap_start} → {gap_end} …",
                          end="", flush=True)
                n = weather.fetch_wind_site_historical(
                    site_id, site_info["lat"], site_info["lon"], gap_start, gap_end)
                if verbose:
                    print(f" {n} records stored.")
        else:
            if verbose:
                print(f"[Wind/{site_id:<12}] Up to date — nothing to fetch.")


def cmd_analyse() -> None:
    """Run analysis, generate charts, dashboard, and print summary."""
    today      = date.today()
    start_date = today - timedelta(days=HISTORY_DAYS)

    print("\n[Analysis] Loading daily data from DB …")
    df = analysis.load_daily_df(start_date, today)

    if len(df) < 30:
        print("  Not enough data for meaningful analysis (need ≥ 30 days).")
        print("  Run:  python main.py update")
        return

    correlations             = analysis.compute_correlations(df)
    model, scaler, r2, fcols = analysis.fit_model(df)

    # Latest commodity rolling averages — held constant across forecast horizon
    latest_commodity = {
        col: df[col].dropna().iloc[-1] if col in df.columns and df[col].notna().any() else None
        for col in analysis.COMMODITY_FEATURES
    }

    print("[Analysis] Building half-hourly model …", end="", flush=True)
    df_hh = analysis.build_halfhourly_df(start_date, today)
    hh_model, hh_scaler, r2_hh, hh_fcols = analysis.fit_halfhourly_model(df_hh)
    print(f" R² = {r2_hh:.4f}")

    print("[Forecast] Fetching 7-day UK weather forecast …", end="", flush=True)
    fc_hourly = weather.fetch_uk_avg_forecast(days=7)
    print(" done.")

    # Fetch 7-day wind forecasts for each offshore site
    site_forecasts: dict = {}
    for site_id, site_info in WIND_SITES.items():
        try:
            site_forecasts[site_id] = weather.fetch_wind_site_forecast(
                site_id, site_info["lat"], site_info["lon"], days=7)
        except Exception as e:
            print(f"  [Wind/{site_id}] Forecast failed: {e}")

    fc_daily    = weather.daily_from_hourly(fc_hourly)
    predictions = analysis.predict_from_forecast(fc_daily, model, scaler, fcols,
                                                 latest_commodity, site_forecasts,
                                                 df_historical=df)
    hh_pred     = analysis.predict_halfhourly_forecast(fc_hourly, hh_model, hh_scaler,
                                                       hh_fcols, latest_commodity,
                                                       site_forecasts, df_historical=df_hh)

    analysis.print_summary(df, correlations, r2, model, scaler, fcols, predictions, r2_hh)

    # ── Store today's predictions in DB ───────────────────────────────────────
    pred_rows = [
        {
            "date":                    str(row["date"].date()),
            "predicted_ex_vat_p_kwh":  row["predicted_ex_vat_p_kwh"],
            "predicted_inc_vat_p_kwh": row["predicted_inc_vat_p_kwh"],
        }
        for _, row in predictions.iterrows()
    ]
    db.upsert_daily_predictions(today, pred_rows)

    # ── Backtest (out-of-sample hold-out) ─────────────────────────────────────
    print("[Backtest] Running 30-day hold-out test …", end="", flush=True)
    backtest_df, backtest_metrics = analysis.run_backtest(df, holdout_days=30)
    if backtest_metrics:
        print(f" MAE={backtest_metrics['mae']:.2f}p  "
              f"RMSE={backtest_metrics['rmse']:.2f}p  "
              f"MAPE={backtest_metrics['mape']:.1f}%")
    else:
        print(" not enough data.")

    print("[Backtest] Running half-hourly hold-out test …", end="", flush=True)
    hh_backtest_df, hh_backtest_metrics = analysis.run_halfhourly_backtest(df_hh, holdout_days=30)
    if hh_backtest_metrics:
        print(f" MAE={hh_backtest_metrics['mae']:.2f}p  "
              f"Peak MAE={hh_backtest_metrics['peak_mae']:.2f}p  "
              f"Off-peak MAE={hh_backtest_metrics['offpeak_mae']:.2f}p")
    else:
        print(" not enough data.")

    # ── Stored predictions vs actuals ─────────────────────────────────────────
    verifiable = db.get_verifiable_predictions(today)

    # ── PNG charts ────────────────────────────────────────────────────────────
    print("\n[Charts]   Saving PNGs …")
    ts = charts.plot_time_series(df)
    sc = charts.plot_scatter(df)
    fc = charts.plot_forecast(predictions, df["avg_price_ex_vat"].mean())
    print(f"  {ts}")
    print(f"  {sc}")
    print(f"  {fc}")

    # ── HTML dashboard ────────────────────────────────────────────────────────
    print("[Dashboard] Generating …", end="", flush=True)
    dash_path = dashboard.generate(
        df, hh_pred, correlations, r2, r2_hh, model, fcols,
        backtest_df=backtest_df,
        backtest_metrics=backtest_metrics,
        verifiable_df=verifiable,
        hh_backtest_df=hh_backtest_df,
        hh_backtest_metrics=hh_backtest_metrics,
    )
    print(f" {dash_path}")


def cmd_status() -> None:
    """Show what data is currently stored in the DB."""
    p_min,  p_max  = db.get_price_date_range()
    g_min,  g_max  = db.get_commodity_date_range()
    s_min,  s_max  = db.get_solar_date_range()
    d_min,  d_max  = db.get_demand_date_range()
    sup_min, sup_max = db.get_generation_date_range()

    # UK weather sites — show range for first site as representative
    first_site = next(iter(UK_WEATHER_SITES))
    uw_min, uw_max = db.get_uk_site_date_range(first_site)

    print("\n[DB Status]")
    print(f"  Database path   : {db.DB_PATH}")
    print(f"  Prices stored   : {f'{p_min[:10]} → {p_max[:10]}' if p_min else '(none)'}")
    print(f"  UK weather      : {f'{uw_min[:10]} → {uw_max[:10]}' if uw_min else '(none)'}")
    print(f"  Solar (PV_Live) : {f'{s_min[:10]} → {s_max[:10]}' if s_min else '(none)'}")
    print(f"  Demand (BMRS)   : {f'{d_min[:10]} → {d_max[:10]}' if d_min else '(none)'}")
    print(f"  Generation mix  : {f'{sup_min[:10]} → {sup_max[:10]}' if sup_min else '(none)'}")
    print(f"  Gas/Oil stored  : {f'{g_min} → {g_max}' if g_min else '(none)'}")


def main() -> None:
    db.init_db()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "update":
        cmd_update()
    elif cmd == "analyse":
        cmd_analyse()
    elif cmd == "status":
        cmd_status()
    elif cmd == "all":
        cmd_update()
        cmd_analyse()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
