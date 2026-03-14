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
import os
import sys
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from app import db, weather, gas, analysis, charts, dashboard, pvlive, demand, supply, midprice, simulation, octopus, storage, eia
from app.config import WIND_SITES, UK_WEATHER_SITES, PUBLIC_MODE


HISTORY_DAYS = 365  # how many days back to maintain


def cmd_update(verbose=True) -> None:
    """Fetch any missing price and weather data into the DB."""
    today      = date.today()
    start_date = today - timedelta(days=HISTORY_DAYS)

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

    # ── Commodity prices (Brent crude + TTF gas + GBP/USD + DXY) ─────────────
    # If currency columns were just migrated (all NULL), force a full re-fetch
    if db.commodity_needs_currency_data():
        if verbose:
            print("[Gas/Oil]  Currency columns empty — forcing full re-fetch …")
        g_gaps = [(start_date, today)]
    else:
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

    # ── Gas storage levels (GIE AGSI+) ──────────────────────────────────────
    st_gaps = storage.missing_storage_ranges(start_date, today)
    if st_gaps:
        for gap_start, gap_end in st_gaps:
            if verbose:
                print(f"[Storage]  Fetching gas storage levels {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = storage.fetch_gas_storage(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Storage]  Gas storage levels up to date — nothing to fetch.")

    # ── US crude oil inventory (EIA) ──────────────────────────────────────────
    eia_gaps = eia.missing_oil_ranges(start_date, today)
    if eia_gaps:
        for gap_start, gap_end in eia_gaps:
            if verbose:
                print(f"[EIA]      Fetching US crude stocks {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = eia.fetch_oil_inventory(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[EIA]      US crude stocks up to date — nothing to fetch.")

    # ── ENTSO-E scheduled exchanges & generation unavailability ──────────────
    from app import entsoe
    ex_gap = entsoe.missing_exchanges_range(start_date, today)
    if ex_gap:
        if verbose:
            print(f"[ENTSO-E]  Fetching scheduled exchanges {ex_gap[0]} → {ex_gap[1]} …",
                  end="", flush=True)
        n = entsoe.fetch_scheduled_exchanges(ex_gap[0], ex_gap[1])
        if verbose:
            print(f" {n} records stored.")
    else:
        if verbose:
            print("[ENTSO-E]  Scheduled exchanges up to date — nothing to fetch.")

    ua_gap = entsoe.missing_unavailability_range(start_date, today)
    if ua_gap:
        if verbose:
            print(f"[ENTSO-E]  Fetching generation unavailability {ua_gap[0]} → {ua_gap[1]} …",
                  end="", flush=True)
        n = entsoe.fetch_unavailability(ua_gap[0], ua_gap[1])
        if verbose:
            print(f" {n} records stored.")
    else:
        if verbose:
            print("[ENTSO-E]  Generation unavailability up to date — nothing to fetch.")

    # ── System prices (Elexon BMRS cash-out) ──────────────────────────────────
    from app import sysprice
    sp_gap = sysprice.missing_sysprice_range(start_date, today)
    if sp_gap:
        if verbose:
            print(f"[SysPrice] Fetching system prices {sp_gap[0]} → {sp_gap[1]} …",
                  end="", flush=True)
        n = sysprice.fetch_system_prices(sp_gap[0], sp_gap[1])
        if verbose:
            print(f" {n} records stored.")
    else:
        if verbose:
            print("[SysPrice] System prices up to date — nothing to fetch.")

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

    # ── Octopus Agile prices (comparison only — not used for model training) ────
    agile_gaps = octopus.missing_price_ranges(start_date, today)
    if agile_gaps:
        for gap_start, gap_end in agile_gaps:
            if verbose:
                print(f"[Agile]    Fetching Octopus Agile prices {gap_start} → {gap_end} …",
                      end="", flush=True)
            n = octopus.fetch_prices(gap_start, gap_end)
            if verbose:
                print(f" {n} records stored.")
    else:
        if verbose:
            print("[Agile]    Octopus Agile prices up to date — nothing to fetch.")

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

    correlations = analysis.compute_correlations(df)

    # Tune LightGBM hyperparameters via walk-forward CV
    print("[Tuning]   Optimising LightGBM hyperparameters (Optuna) …", end="", flush=True)
    tuned_params = analysis.tune_lgbm_params(df, n_trials=150, n_folds=10)
    if tuned_params:
        print(f" done (n_est={tuned_params.get('n_estimators')}, "
              f"lr={tuned_params.get('learning_rate', 0):.3f}, "
              f"leaves={tuned_params.get('num_leaves')})")
    else:
        print(" skipped (not enough data)")

    # Fit Ridge + LightGBM ensemble with walk-forward optimised blend weight
    print("[Analysis] Fitting Ridge + LightGBM ensemble …", end="", flush=True)
    ensemble = analysis.fit_ensemble(df, lgbm_params=tuned_params or None)
    ridge_info = ensemble["ridge"]
    model, scaler, fcols = ridge_info["model"], ridge_info["scaler"], ridge_info["feature_cols"]
    r2 = ridge_info["r2"]
    print(f" Ridge α={model.alpha_:.4g} R²={r2:.4f}  "
          f"LightGBM R²={ensemble['lgbm']['r2']:.4f}  "
          f"blend={ensemble['blend_weight']:.1f}R/{1-ensemble['blend_weight']:.1f}L")

    # Latest commodity/inventory/lag values — held constant across forecast horizon
    latest_commodity = {
        col: df[col].dropna().iloc[-1] if col in df.columns and df[col].notna().any() else None
        for col in (list(analysis.COMMODITY_FEATURES) +
                    list(analysis.INVENTORY_FEATURES) +
                    list(analysis.LAG_ROLLING_FEATURES) +
                    list(analysis.ENTSOE_FEATURES) +
                    list(analysis.SYSTEM_PRICE_FEATURES) +
                    list(analysis.RAMP_FEATURES))
    }

    print("[Analysis] Building half-hourly ensemble …", end="", flush=True)
    df_hh = analysis.build_halfhourly_df(start_date, today)
    hh_ensemble = analysis.fit_halfhourly_ensemble(df_hh)
    hh_ridge = hh_ensemble["ridge"]
    hh_model, hh_scaler, hh_fcols = hh_ridge["model"], hh_ridge["scaler"], hh_ridge["feature_cols"]
    r2_hh = hh_ridge["r2"]
    print(f" Ridge R²={r2_hh:.4f}  "
          f"LightGBM R²={hh_ensemble['lgbm']['r2']:.4f}  "
          f"blend={hh_ensemble['blend_weight']:.1f}R/{1-hh_ensemble['blend_weight']:.1f}L")

    # Hourly price profile for dashboard (mean/std by hour + weekday flag)
    _hh = df_hh.assign(
        hour=df_hh["datetime_local"].dt.hour,
        is_weekday=df_hh["datetime_local"].dt.dayofweek < 5,
    )
    hh_hourly_profile = (
        _hh.groupby(["hour", "is_weekday"])["epex_price_p_kwh"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

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

    # ── Archive today's weather + wind forecast for future lead-time accuracy tests ──
    wx_archive_rows = []
    for _, row in fc_daily.iterrows():
        target = row["date"].date() if hasattr(row["date"], "date") else date.fromisoformat(str(row["date"])[:10])
        if target < today:
            continue
        wx_archive_rows.append({
            "fetch_date":          str(today),
            "target_date":         str(target),
            "lead_days":           (target - today).days,
            "temperature_2m":      row.get("temperature_2m"),
            "shortwave_radiation": row.get("shortwave_radiation"),
            "precipitation":       row.get("precipitation"),
        })
    db.upsert_weather_forecast_archive(wx_archive_rows)

    wind_archive_rows = []
    for site_id, site_df in site_forecasts.items():
        if site_df is None or site_df.empty:
            continue
        site_df = site_df.copy()
        site_df["_date"] = site_df["datetime"].dt.date
        for d, grp in site_df[site_df["_date"] >= today].groupby("_date"):
            wind_archive_rows.append({
                "fetch_date":  str(today),
                "target_date": str(d),
                "site_id":     site_id,
                "wind_speed":  float(grp["wind_speed"].mean()),
            })
    db.upsert_wind_site_forecast_archive(wind_archive_rows)

    predictions = analysis.predict_ensemble(fc_daily, ensemble,
                                             latest_commodity, site_forecasts,
                                             df_historical=df)
    hh_pred     = analysis.predict_halfhourly_ensemble(fc_hourly, hh_ensemble,
                                                       latest_commodity, site_forecasts,
                                                       df_historical=df_hh)

    analysis.print_summary(df, correlations, r2, model, scaler, fcols, predictions, r2_hh)

    # ── Store today's predictions in DB ───────────────────────────────────────
    pred_rows = [
        {
            "date":                str(row["date"].date()),
            "predicted_epex_p_kwh": row["predicted_epex_p_kwh"],
        }
        for _, row in predictions.iterrows()
    ]
    db.upsert_daily_predictions(today, pred_rows)

    # ── Backtest (out-of-sample hold-out) ─────────────────────────────────────
    print("[Backtest] Running 30-day hold-out test …", end="", flush=True)
    backtest_df, backtest_metrics = analysis.run_backtest(df, holdout_days=30)
    if backtest_metrics:
        print(f" MAE={backtest_metrics['mae']:.2f}p  "
              f"R²={backtest_metrics['r2']:.3f}  "
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

    # ── Archived lead-time accuracy backtest ──────────────────────────────────
    print("[Backtest] Running archived lead-time backtest …", end="", flush=True)
    leadtime_detail_df, leadtime_metrics = analysis.run_archived_leadtime_backtest(df)
    if leadtime_metrics:
        leads = sorted(leadtime_metrics.keys())
        print(" " + "  ".join(f"D+{l} MAE={leadtime_metrics[l]['mae']:.2f}p" for l in leads))
    else:
        print(" no archive data yet.")

    # ── Walk-forward cross-validation ────────────────────────────────────────
    print("[Backtest] Running walk-forward CV …", end="", flush=True)
    wfcv_detail, wfcv_metrics = analysis.run_walkforward_cv(df, n_folds=10, min_train_days=120,
                                                              lgbm_params=tuned_params or None)
    if wfcv_metrics:
        print(f" Ridge MAE={wfcv_metrics['mae_ridge']:.2f}p  "
              f"LightGBM MAE={wfcv_metrics['mae_lgbm']:.2f}p  "
              f"Ensemble MAE={wfcv_metrics['mae_ensemble']:.2f}p")
    else:
        print(" not enough data.")

    # ── Stored predictions vs actuals ─────────────────────────────────────────
    verifiable = db.get_verifiable_predictions(today)

    # ── Tariff design (first 3 days only — forecast reliability window) ───────
    if not PUBLIC_MODE:
        # Flat margin variant (SUPPLIER_MARGIN p/kWh adder)
        daily_tariffs_3 = analysis.design_daily_tariffs(hh_pred, days=3, slots="3",
                                                         margin_mode="flat")
        daily_tariffs_4 = analysis.design_daily_tariffs(hh_pred, days=3, slots="4",
                                                         margin_mode="flat")
        # Multiplier variant (wholesale × SUPPLIER_MULTIPLIER, like Octopus Agile)
        daily_tariffs_3_mult = analysis.design_daily_tariffs(hh_pred, days=3, slots="3",
                                                              margin_mode="multiplier")
        daily_tariffs_4_mult = analysis.design_daily_tariffs(hh_pred, days=3, slots="4",
                                                              margin_mode="multiplier")

        # ── Agile price history for tariff comparison chart (last 60 days) ─────────
        price_hist_rows = db.get_halfhourly_prices(today - timedelta(days=60), today)
        price_hist = pd.DataFrame(
            price_hist_rows,
            columns=["datetime", "price_ex_vat", "price_inc_vat", "wholesale_price", "is_peak"],
        ) if price_hist_rows else pd.DataFrame()

        # ── Customer behaviour simulation ─────────────────────────────────────────
        print("[Simulation] Running customer behaviour scenarios …", end="", flush=True)
        sim_df = simulation.run_simulation(price_hist)
        if not sim_df.empty:
            r0 = sim_df[sim_df["scenario"] == "no_shift"].iloc[0]
            rev = sim_df[sim_df["scenario"] == "ev_household"].iloc[0]
            print(f" no-shift: £{r0['cust_bill_ours_annual_gbp']:.0f}/yr  "
                  f"EV household: £{rev['cust_bill_ours_annual_gbp']:.0f}/yr  "
                  f"(Agile: £{r0['cust_bill_agile_annual_gbp']:.0f}/yr)")
        else:
            print(" skipped — no history data.")
    else:
        daily_tariffs_3 = None
        daily_tariffs_4 = None
        daily_tariffs_3_mult = None
        daily_tariffs_4_mult = None
        price_hist = pd.DataFrame()
        sim_df = pd.DataFrame()

    # ── PNG charts ────────────────────────────────────────────────────────────
    print("\n[Charts]   Saving PNGs …")
    ts = charts.plot_time_series(df)
    sc = charts.plot_scatter(df)
    fc = charts.plot_forecast(predictions, df["avg_epex_p_kwh"].mean())
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
        daily_tariffs_3=daily_tariffs_3,
        daily_tariffs_4=daily_tariffs_4,
        daily_tariffs_3_mult=daily_tariffs_3_mult,
        daily_tariffs_4_mult=daily_tariffs_4_mult,
        daily_predictions=predictions,
        price_hist=price_hist,
        leadtime_detail_df=leadtime_detail_df,
        leadtime_metrics=leadtime_metrics,
        hh_hourly_profile=hh_hourly_profile,
        sim_df=sim_df,
        ensemble=ensemble,
        hh_ensemble=hh_ensemble,
        wfcv_detail=wfcv_detail,
        wfcv_metrics=wfcv_metrics,
    )
    print(f" {dash_path}")

    # ── S3 upload (optional — only runs if DASHBOARD_BUCKET env var is set) ──
    bucket = os.environ.get("DASHBOARD_BUCKET")
    if bucket:
        try:
            import boto3
            from app.config import CHARTS_DIR
            s3 = boto3.client("s3")
            # Upload dashboard HTML
            s3.upload_file(
                str(dash_path), bucket, "index.html",
                ExtraArgs={"ContentType": "text/html", "CacheControl": "max-age=300"},
            )
            # Upload chart PNGs
            chart_count = 0
            for png in CHARTS_DIR.glob("*.png"):
                s3.upload_file(
                    str(png), bucket, f"charts/{png.name}",
                    ExtraArgs={"ContentType": "image/png", "CacheControl": "max-age=3600"},
                )
                chart_count += 1
            # Invalidate CloudFront cache
            cf_dist = os.environ.get("CLOUDFRONT_DISTRIBUTION_ID")
            if cf_dist:
                cf = boto3.client("cloudfront")
                cf.create_invalidation(
                    DistributionId=cf_dist,
                    InvalidationBatch={
                        "Paths": {"Quantity": 1, "Items": ["/*"]},
                        "CallerReference": str(int(datetime.now().timestamp())),
                    },
                )
            print(f"[S3] Uploaded index.html + {chart_count} charts → s3://{bucket}/")
        except Exception as exc:
            print(f"[S3] Upload failed (non-fatal): {exc}")


def cmd_status() -> None:
    """Show what data is currently stored in the DB."""
    p_min,  p_max  = db.get_price_date_range()
    g_min,  g_max  = db.get_commodity_date_range()
    s_min,  s_max  = db.get_solar_date_range()
    d_min,  d_max  = db.get_demand_date_range()
    sup_min, sup_max = db.get_generation_date_range()
    st_min, st_max = db.get_gas_storage_date_range()
    eia_min, eia_max = db.get_oil_inventory_date_range()

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
    print(f"  Gas storage     : {f'{st_min} → {st_max}' if st_min else '(none — set GIE_API_KEY)'}")
    print(f"  Oil inventory   : {f'{eia_min} → {eia_max}' if eia_min else '(none — set EIA_API_KEY)'}")
    ex_min, ex_max = db.get_entsoe_exchanges_date_range()
    ua_min, ua_max = db.get_entsoe_unavailability_date_range()
    print(f"  ENTSO-E flows   : {f'{ex_min} → {ex_max}' if ex_min else '(none — set ENTSOE_API_KEY)'}")
    print(f"  ENTSO-E outages : {f'{ua_min} → {ua_max}' if ua_min else '(none — set ENTSOE_API_KEY)'}")


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
