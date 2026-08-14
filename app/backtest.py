"""
Backtesting: hold-out tests, walk-forward cross-validation, and
archived lead-time accuracy analysis.
"""
import warnings

import numpy as np
import pandas as pd

from app.features import (
    COMMODITY_FEATURES, INVENTORY_FEATURES, LAG_ROLLING_FEATURES, WIND_SITES,
    _time_features,
)
from app.models import (
    fit_model, fit_lgbm_model, fit_halfhourly_model, fit_halfhourly_lgbm,
    predict_from_forecast,
)

from datetime import date, timedelta


def run_walkforward_cv(df: pd.DataFrame, n_folds: int = 5,
                        min_train_days: int = 120,
                        model_type: str = "ensemble",
                        lgbm_params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Expanding-window walk-forward cross-validation.

    Returns (detail_df, summary_metrics).
      detail_df columns: date, actual, pred_ridge, pred_lgbm, predicted, fold
      summary_metrics: {mae_ridge, mae_lgbm, mae_ensemble, rmse, mape, r2,
                        per_fold: [{fold, mae_ridge, mae_lgbm, mae_ensemble, n}]}
    """
    if len(df) < min_train_days + n_folds * 5:
        return pd.DataFrame(), {}

    test_size = (len(df) - min_train_days) // n_folds
    if test_size < 5:
        return pd.DataFrame(), {}

    all_rows = []
    for fold in range(n_folds):
        cutoff = min_train_days + fold * test_size
        end = min(cutoff + test_size, len(df))
        if cutoff >= len(df) - 3:
            break

        train = df.iloc[:cutoff].copy()
        test = df.iloc[cutoff:end].copy()
        if len(test) < 3:
            break

        try:
            r_model, r_scaler, _, r_fcols = fit_model(train)
            l_model, l_fcols, _ = fit_lgbm_model(train, params_override=lgbm_params)
        except Exception:
            continue

        r_cols = [c for c in r_fcols if c in test.columns]
        l_cols = [c for c in l_fcols if c in test.columns]
        test_clean = test.dropna(subset=list(set(r_cols + l_cols + ["avg_epex_p_kwh"])))
        if len(test_clean) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_r = test_clean[r_cols].values
            pred_r = r_model.predict(r_scaler.transform(X_r))
            pred_l = l_model.predict(test_clean[l_cols].values)

        for i, (_, row) in enumerate(test_clean.iterrows()):
            all_rows.append({
                "date":     row["date"],
                "actual":   row["avg_epex_p_kwh"],
                "pred_ridge": pred_r[i],
                "pred_lgbm":  pred_l[i],
                "fold":     fold,
            })

    if not all_rows:
        return pd.DataFrame(), {}

    detail = pd.DataFrame(all_rows)

    # Find optimal blend weight from fold data (5% steps for finer calibration)
    best_w, best_mae = 0.5, float("inf")
    for w_int in range(21):
        w = w_int / 20
        blended = w * detail["pred_ridge"] + (1 - w) * detail["pred_lgbm"]
        mae = (blended - detail["actual"]).abs().mean()
        if mae < best_mae:
            best_w, best_mae = w, mae

    detail["predicted"] = best_w * detail["pred_ridge"] + (1 - best_w) * detail["pred_lgbm"]
    detail["error"] = detail["predicted"] - detail["actual"]

    # Per-fold metrics
    per_fold = []
    for fold_id in detail["fold"].unique():
        f = detail[detail["fold"] == fold_id]
        per_fold.append({
            "fold":         int(fold_id),
            "mae_ridge":    (f["pred_ridge"] - f["actual"]).abs().mean(),
            "mae_lgbm":     (f["pred_lgbm"] - f["actual"]).abs().mean(),
            "mae_ensemble": f["error"].abs().mean(),
            "n":            len(f),
        })

    mae_r = (detail["pred_ridge"] - detail["actual"]).abs().mean()
    mae_l = (detail["pred_lgbm"] - detail["actual"]).abs().mean()
    mae_e = detail["error"].abs().mean()
    rmse = np.sqrt((detail["error"] ** 2).mean())
    mape = (detail["error"].abs() / detail["actual"].abs()).mean() * 100
    ss_res = (detail["error"] ** 2).sum()
    ss_tot = ((detail["actual"] - detail["actual"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    summary = {
        "mae_ridge":    mae_r,
        "mae_lgbm":     mae_l,
        "mae_ensemble":  mae_e,
        "rmse":         rmse,
        "mape":         mape,
        "r2":           r2,
        "blend_weight": best_w,
        "n_folds":      len(per_fold),
        "per_fold":     per_fold,
    }
    return detail, summary


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

    # Latest commodity/inventory/lag from training data only (as would be available on cutoff date)
    latest_commodity_bt = {
        col: train[col].dropna().iloc[-1]
        if col in train.columns and train[col].notna().any() else None
        for col in list(COMMODITY_FEATURES) + list(INVENTORY_FEATURES) + list(LAG_ROLLING_FEATURES)
    }

    # Build test feature set — include all columns the model needs
    base_cols = ["date", "temperature_2m", "shortwave_radiation", "precipitation"]
    extra_cols = [c for c in fcols_bt if c in test.columns and c not in base_cols]
    test_fc = test[base_cols + extra_cols].copy()

    preds = predict_from_forecast(test_fc, model_bt, scaler_bt, fcols_bt,
                                  latest_commodity_bt)

    result = pd.merge(
        test[["date", "avg_epex_p_kwh"]],
        preds[["date", "predicted_epex_p_kwh"]],
        on="date",
    )
    result["error"] = result["predicted_epex_p_kwh"] - result["avg_epex_p_kwh"]

    mae  = result["error"].abs().mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    # Mean absolute percentage error
    mape = (result["error"].abs() / result["avg_epex_p_kwh"].abs()).mean() * 100

    ss_res = (result["error"] ** 2).sum()
    ss_tot = ((result["avg_epex_p_kwh"] - result["avg_epex_p_kwh"].mean()) ** 2).sum()
    r2_holdout = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    metrics = {
        "r2_train":    r2_train,
        "r2":          r2_holdout,
        "mae":         mae,
        "rmse":        rmse,
        "mape":        mape,
        "holdout_days": holdout_days,
        "train_days":  len(train),
    }
    return result, metrics


def run_archived_leadtime_backtest(
    df: pd.DataFrame,
    max_lead: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """
    Lead-time accuracy backtest using the archived weather forecasts.

    For each (fetch_date, target_date) pair in the archive:
      - Substitute the archived weather forecast (temp, solar, precip, wind sites)
        into the target day's feature row — replacing actuals with what the model
        would actually have seen at that lead time.
      - Run the daily model and compare to the actual EPEX price.

    Unlike run_backtest(), which uses actual weather and is a best-case ceiling,
    this test shows true real-world accuracy degradation with forecast lead time.

    Returns (detail_df, metrics_by_lead_dict).
      detail_df columns: fetch_date, target_date, lead_days, actual, predicted, error
      metrics_by_lead_dict keys: 1..max_lead, values: {mae, rmse, mape, n}
    """
    from app import db as _db

    # Need at least 30 days of training data before the archive started
    if df.empty:
        return pd.DataFrame(), {}

    fetch_dates = _db.get_forecast_archive_fetch_dates()
    if not fetch_dates:
        return pd.DataFrame(), {}

    # Index df by date string for fast lookup
    df_indexed = df.copy()
    df_indexed["date_str"] = df_indexed["date"].dt.strftime("%Y-%m-%d")
    df_indexed = df_indexed.set_index("date_str")

    rows = []
    for fetch_date_str in fetch_dates:
        fetch_date = date.fromisoformat(fetch_date_str)

        # Train on data up to and including the fetch date
        train = df[df["date"].dt.date <= fetch_date].copy()
        if len(train) < 30:
            continue

        model_bt, scaler_bt, _, fcols_bt = fit_model(train)
        latest_commodity_bt = {
            col: train[col].dropna().iloc[-1]
            if col in train.columns and train[col].notna().any() else None
            for col in list(COMMODITY_FEATURES) + list(INVENTORY_FEATURES) + list(LAG_ROLLING_FEATURES)
        }

        # Get archived weather forecasts made on this fetch_date
        target_min = fetch_date + timedelta(days=1)
        target_max = fetch_date + timedelta(days=max_lead)
        wx_rows = _db.get_weather_forecast_archive(target_min, target_max)
        wind_rows = _db.get_wind_site_forecast_archive(target_min, target_max)

        # Filter to this fetch_date only
        wx_for_date = {
            r[1]: {"temperature_2m": r[3], "shortwave_radiation": r[4], "precipitation": r[5]}
            for r in wx_rows if r[0] == fetch_date_str
        }
        wind_for_date: dict[str, dict[str, float]] = {}
        for r in wind_rows:
            if r[0] != fetch_date_str:
                continue
            wind_for_date.setdefault(r[1], {})[r[2]] = r[3]  # target_date → {site_id: speed}

        for target_date_str, wx in wx_for_date.items():
            target_date = date.fromisoformat(target_date_str)
            lead = (target_date - fetch_date).days

            # Get actual price for the target date
            if target_date_str not in df_indexed.index:
                continue
            actual_row = df_indexed.loc[target_date_str]
            actual_p = float(actual_row["avg_epex_p_kwh"])

            # Build forecast row: start from actual historical row, substitute forecast weather
            fc_row = actual_row.to_frame().T.copy()
            fc_row["temperature_2m"]      = wx["temperature_2m"]
            fc_row["shortwave_radiation"] = wx["shortwave_radiation"]
            fc_row["precipitation"]       = wx["precipitation"]
            # Derived: heating_dd from archived forecast temperature
            if wx["temperature_2m"] is not None:
                fc_row["heating_dd"] = max(0.0, 15.5 - wx["temperature_2m"])
            # Wind sites: substitute archived wind speed per site
            site_wind = wind_for_date.get(target_date_str, {})
            for site_id in WIND_SITES:
                col = f"wind_{site_id}"
                if col in fc_row.columns and site_id in site_wind:
                    fc_row[col] = site_wind[site_id]
            # Null out solar_gw so predict_from_forecast re-estimates it from radiation
            if "solar_gw" in fc_row.columns:
                fc_row["solar_gw"] = np.nan
            # Use actual lag for D+1 (always known); for D+2+ use the latest known price
            # (same approach as the real forecast pipeline)
            if "epex_lag1_gbp_mwh" in fc_row.columns:
                if lead == 1:
                    pass  # keep actual lag — it is truly known for D+1
                else:
                    # Latest EPEX known at fetch_date
                    known = train["epex_lag1_gbp_mwh"].dropna()
                    fc_row["epex_lag1_gbp_mwh"] = float(known.iloc[-1]) if len(known) else np.nan

            pred_df = predict_from_forecast(
                fc_row, model_bt, scaler_bt, fcols_bt, latest_commodity_bt,
                df_historical=train,
            )
            if pred_df.empty:
                continue
            predicted = float(pred_df.iloc[0]["predicted_epex_p_kwh"])
            rows.append({
                "fetch_date":  fetch_date_str,
                "target_date": target_date_str,
                "lead_days":   lead,
                "actual":      actual_p,
                "predicted":   predicted,
                "error":       predicted - actual_p,
            })

    if not rows:
        return pd.DataFrame(), {}

    detail_df = pd.DataFrame(rows)
    metrics_by_lead: dict = {}
    for lead in range(1, max_lead + 1):
        sub = detail_df[detail_df["lead_days"] == lead]
        if len(sub) < 3:
            continue
        errs = sub["error"].values
        metrics_by_lead[lead] = {
            "mae":  float(np.abs(errs).mean()),
            "rmse": float(np.sqrt((errs ** 2).mean())),
            "mape": float((np.abs(errs) / sub["actual"].abs()).mean() * 100),
            "n":    len(sub),
        }
    return detail_df, metrics_by_lead


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

    # Fit Ridge + LightGBM ensemble on training data
    r_model, r_scaler, r2_train, r_fcols = fit_halfhourly_model(train)
    try:
        l_model, l_fcols, _ = fit_halfhourly_lgbm(train)
        has_lgbm = True
    except Exception:
        has_lgbm = False

    r_cols_avail = [c for c in r_fcols if c in test.columns]
    test_clean = test.dropna(subset=r_cols_avail + ["epex_price_p_kwh"]).copy()
    X_r = r_scaler.transform(test_clean[r_cols_avail].fillna(0).values)
    pred_r = r_model.predict(X_r)

    if has_lgbm:
        l_cols_avail = [c for c in l_fcols if c in test_clean.columns]
        pred_l = l_model.predict(test_clean[l_cols_avail].fillna(0).values)
        # Simple 50/50 blend for backtest (no CV overhead here)
        test_clean["predicted"] = 0.5 * pred_r + 0.5 * pred_l
    else:
        test_clean["predicted"] = pred_r

    result = test_clean[["datetime_local", "epex_price_p_kwh", "predicted", "is_peak"]].copy()
    result.columns = ["datetime_local", "actual", "predicted", "is_peak"]
    result["error"] = result["predicted"] - result["actual"]

    mae  = result["error"].abs().mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    mape = (result["error"].abs() / result["actual"].abs()).mean() * 100

    ss_res = (result["error"] ** 2).sum()
    ss_tot = ((result["actual"] - result["actual"].mean()) ** 2).sum()
    r2_holdout = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Peak vs off-peak accuracy split
    peak_mae    = result[result["is_peak"] == 1]["error"].abs().mean()
    offpeak_mae = result[result["is_peak"] == 0]["error"].abs().mean()

    metrics = {
        "r2_train":     r2_train,
        "r2":           r2_holdout,
        "mae":          mae,
        "rmse":         rmse,
        "mape":         mape,
        "peak_mae":     peak_mae,
        "offpeak_mae":  offpeak_mae,
        "holdout_days": holdout_days,
    }
    return result, metrics
