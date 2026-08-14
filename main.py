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
import time
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from app import db, weather, gas, analysis, charts, dashboard, pvlive, demand, supply, midprice, octopus, storage, eia
from app.config import WIND_SITES, UK_WEATHER_SITES


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
    # Fetch up to tomorrow — D+1 prices are available after ~12:00 CET auction
    tomorrow = today + timedelta(days=1)
    mp_gaps = midprice.missing_midprice_ranges(start_date, tomorrow)
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

    # ── D+1 actual EPEX prices ────────────────────────────────────────────────
    tomorrow = today + timedelta(days=1)
    tomorrow_hh_rows = db.get_halfhourly_midprice(tomorrow, tomorrow)
    tomorrow_avg_epex = None
    if tomorrow_hh_rows:
        tomorrow_avg_epex = sum(r[1] for r in tomorrow_hh_rows) / len(tomorrow_hh_rows) * 0.1  # £/MWh → p/kWh
        print(f"[D+1]      Tomorrow's EPEX actual: {tomorrow_avg_epex:.2f}p/kWh avg ({len(tomorrow_hh_rows)} slots)")
    else:
        print("[D+1]      No EPEX prices available for tomorrow — all days will be forecasted.")

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

    # ── Split forecast: D+2+ model predictions only (D+1 uses actuals) ─────
    if tomorrow_avg_epex is not None:
        # Filter weather forecast to D+2+ for model prediction
        fc_daily_pred = fc_daily[fc_daily["date"].apply(
            lambda d: (d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])) > tomorrow
        )].copy()
        fc_hourly_pred = fc_hourly[fc_hourly["datetime"].dt.date > tomorrow].copy()

        # Use tomorrow's actual daily avg as lag-1 for D+2+ (instead of today's)
        latest_commodity["epex_lag1_gbp_mwh"] = tomorrow_avg_epex / 0.1  # p/kWh → £/MWh

        # Append tomorrow's actual HH prices to historical HH df so price_lag1_slot is real for D+2
        from zoneinfo import ZoneInfo
        _london = ZoneInfo("Europe/London")
        tomorrow_hh_for_hist = []
        for row in tomorrow_hh_rows:
            dt_utc = pd.Timestamp(row[0]).tz_localize("UTC")
            dt_local = dt_utc.tz_convert(_london)
            tomorrow_hh_for_hist.append({
                "datetime_local": dt_local,
                "epex_price_p_kwh": row[1] * 0.1,  # £/MWh → p/kWh
            })
        if tomorrow_hh_for_hist:
            df_hh_extended = pd.concat([df_hh, pd.DataFrame(tomorrow_hh_for_hist)], ignore_index=True)
        else:
            df_hh_extended = df_hh

        # Model predictions for D+2+ — use recursive lag so each day's prediction
        # feeds the next day's epex_lag1, rather than holding a stale constant.
        if not fc_daily_pred.empty:
            predictions_d2plus = analysis.predict_ensemble(fc_daily_pred, ensemble,
                                                            latest_commodity, site_forecasts,
                                                            df_historical=df,
                                                            recursive_lag=True)
        else:
            predictions_d2plus = pd.DataFrame()
        if not fc_hourly_pred.empty:
            hh_pred_d2plus = analysis.predict_halfhourly_ensemble(fc_hourly_pred, hh_ensemble,
                                                                    latest_commodity, site_forecasts,
                                                                    df_historical=df_hh_extended)
        else:
            hh_pred_d2plus = pd.DataFrame()

        # Build D+1 actual rows
        d1_daily = pd.DataFrame([{
            "date": pd.Timestamp(tomorrow),
            "predicted_epex_p_kwh": tomorrow_avg_epex,
            "pred_ridge": tomorrow_avg_epex,
            "pred_lgbm": tomorrow_avg_epex,
            "pred_q10": tomorrow_avg_epex,
            "pred_q90": tomorrow_avg_epex,
            "is_actual": True,
        }])

        d1_hh_rows = []
        for row in tomorrow_hh_rows:
            dt_utc = pd.Timestamp(row[0]).tz_localize("UTC")
            dt_local = dt_utc.tz_convert(_london)
            price_p = row[1] * 0.1  # £/MWh → p/kWh
            d1_hh_rows.append({
                "datetime_local": dt_local,
                "predicted_epex_p_kwh": price_p,
                "pred_ridge": price_p,
                "pred_lgbm": price_p,
                "pred_q10": price_p,
                "pred_q90": price_p,
                "is_actual": True,
            })
        d1_hh = pd.DataFrame(d1_hh_rows)

        # Concat D+1 actuals + D+2+ forecasts
        if not predictions_d2plus.empty:
            if "is_actual" not in predictions_d2plus.columns:
                predictions_d2plus["is_actual"] = False
            predictions = pd.concat([d1_daily, predictions_d2plus], ignore_index=True)
        else:
            predictions = d1_daily
        if not hh_pred_d2plus.empty:
            if "is_actual" not in hh_pred_d2plus.columns:
                hh_pred_d2plus["is_actual"] = False
            hh_pred = pd.concat([d1_hh, hh_pred_d2plus], ignore_index=True)
        else:
            hh_pred = d1_hh
    else:
        # No tomorrow prices — forecast all days with recursive lag
        predictions = analysis.predict_ensemble(fc_daily, ensemble,
                                                 latest_commodity, site_forecasts,
                                                 df_historical=df,
                                                 recursive_lag=True)
        hh_pred     = analysis.predict_halfhourly_ensemble(fc_hourly, hh_ensemble,
                                                           latest_commodity, site_forecasts,
                                                           df_historical=df_hh)
        if "is_actual" not in predictions.columns:
            predictions["is_actual"] = False
        if "is_actual" not in hh_pred.columns:
            hh_pred["is_actual"] = False

    analysis.print_summary(df, correlations, r2, model, scaler, fcols, predictions, r2_hh)

    # ── Store today's predictions in DB ───────────────────────────────────────
    pred_rows = [
        {
            "date":                str(row["date"].date() if hasattr(row["date"], "date") else row["date"]),
            "predicted_epex_p_kwh": row["predicted_epex_p_kwh"],
            "is_actual":           int(bool(row.get("is_actual", False))),
        }
        for _, row in predictions.iterrows()
    ]
    db.upsert_daily_predictions(today, pred_rows)

    # ── Store half-hourly predictions in SQLite ──────────────────────────────
    if not hh_pred.empty and "datetime_local" in hh_pred.columns:
        import pandas as _pd
        from zoneinfo import ZoneInfo as _ZI
        _london = _ZI("Europe/London")
        hh_rows = []
        for _, row in hh_pred.iterrows():
            dt = _pd.Timestamp(row["datetime_local"])
            if dt.tzinfo is None:
                dt = dt.tz_localize(_london)
            slot_utc = dt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            hh_rows.append({
                "datetime_utc":         slot_utc,
                "predicted_epex_p_kwh": row["predicted_epex_p_kwh"],
                "pred_q10":             row.get("pred_q10"),
                "pred_q90":             row.get("pred_q90"),
                "is_actual":            int(bool(row.get("is_actual", False))),
            })
        n = db.upsert_halfhourly_predictions(today, hh_rows)
        print(f"[DB] Stored {n} half-hourly predictions in SQLite.")

    # ── Week-ahead LLM summary ─────────────────────────────────────────────────
    try:
        from app.summary import generate_week_summary
        print("[Summary] Generating week-ahead narrative …", end="", flush=True)
        summary = generate_week_summary(predictions, hh_pred, df["avg_epex_p_kwh"].mean())
        if summary:
            db.upsert_forecast_summary(today, summary["week_summary"], summary.get("days", []))
            print(f" done.")
            print(f"  {summary['week_summary']}")
        else:
            print(" skipped (no OPENAI_API_KEY or call failed).")
    except Exception as e:
        print(f" failed: {e}")

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

    # ── Bias correction from recent prediction errors ─────────────────────────
    bias_by_lead = analysis.compute_bias_by_lead(verifiable, window_days=30)
    if bias_by_lead:
        leads = sorted(bias_by_lead.keys())
        print("[Bias]     " + "  ".join(f"D+{l} {bias_by_lead[l]:+.2f}p" for l in leads))
        predictions = analysis.apply_bias_correction(predictions, bias_by_lead, today)
        hh_pred     = analysis.apply_bias_correction_hh(hh_pred, bias_by_lead, today)

    # ── Widen uncertainty bands for longer lead days ──────────────────────────
    if leadtime_metrics:
        predictions = analysis.scale_intervals_by_leadtime(predictions, leadtime_metrics, today)
        hh_pred     = analysis.scale_hh_intervals_by_leadtime(hh_pred, leadtime_metrics, today)

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
        daily_predictions=predictions,
        leadtime_detail_df=leadtime_detail_df,
        leadtime_metrics=leadtime_metrics,
        hh_hourly_profile=hh_hourly_profile,
        ensemble=ensemble,
        hh_ensemble=hh_ensemble,
        wfcv_detail=wfcv_detail,
        wfcv_metrics=wfcv_metrics,
        forecast_summary=db.get_forecast_summary(today),
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

        # Retry loop: wait for tomorrow's EPEX prices if not yet available
        today = date.today()
        tomorrow = today + timedelta(days=1)
        deadline = datetime(today.year, today.month, today.day, 16, 0, tzinfo=timezone.utc)
        while not db.has_complete_midprice(tomorrow, min_slots=46):
            if datetime.now(timezone.utc) >= deadline:
                print("ERROR: Tomorrow's EPEX prices not available by 16:00 UTC deadline")
                sys.exit(1)
            print(f"[Wait] Tomorrow's EPEX prices not yet available, retrying in 5 min …")
            time.sleep(300)
            midprice.fetch_midprice(tomorrow, tomorrow)

        cmd_analyse()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
