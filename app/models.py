"""
Model fitting, prediction, and ensemble blending.
Daily model: Ridge + LightGBM ensemble for daily avg EPEX price.
Half-hourly model: Ridge + LightGBM ensemble for per-slot EPEX price.
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from app.features import (
    DAILY_FEATURES, HH_FEATURES, COMMODITY_FEATURES, INVENTORY_FEATURES,
    LAG_ROLLING_FEATURES, ENTSOE_FEATURES, SYSTEM_PRICE_FEATURES,
    RAMP_FEATURES, ALL_FEATURE_LABELS,
    _uk_holidays, estimate_solar_from_radiation, estimate_wind_gen_from_speed,
    build_demand_profile, build_imports_profile, _time_features, WIND_SITES,
)


# ── Daily model ───────────────────────────────────────────────────────────────

def fit_model(df: pd.DataFrame):
    """
    Fit a multiple linear regression of weather → daily EPEX wholesale price (p/kWh).
    Returns (model, scaler, r2, feature_cols).
    """
    # Include features present with at least some data; dropna below removes sparse rows
    feature_cols = [c for c in DAILY_FEATURES if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols)
    X = df_model[feature_cols].values
    y = df_model["avg_epex_p_kwh"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model.fit(X_scaled, y)

    y_pred  = model.predict(X_scaled)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return model, scaler, r2, feature_cols


# ── LightGBM model ──────────────────────────────────────────────────────────

# Conservative hyperparameters for ~364 daily rows — prioritise generalisation.
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)


def tune_lgbm_params(df: pd.DataFrame, n_trials: int = 80,
                      min_train_days: int = 120, n_folds: int = 5,
                      target_col: str = "avg_epex_p_kwh",
                      feature_list: list | None = None) -> dict:
    """
    Use Optuna to find LightGBM hyperparameters that minimise walk-forward CV MAE.
    Returns the best params dict (ready to merge with LGBM_PARAMS).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    features = feature_list or DAILY_FEATURES
    feature_cols = [c for c in features if c in df.columns and df[c].notna().any()]
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df_clean) < min_train_days + n_folds * 5:
        return {}

    test_size = (len(df_clean) - min_train_days) // n_folds

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=42,
            verbosity=-1,
        )
        maes = []
        for fold in range(n_folds):
            cutoff = min_train_days + fold * test_size
            end = min(cutoff + test_size, len(df_clean))
            if cutoff >= len(df_clean) - 3:
                break
            train = df_clean.iloc[:cutoff]
            test = df_clean.iloc[cutoff:end]
            if len(test) < 3:
                break
            X_train = train[feature_cols].values
            y_train = train[target_col].values
            X_test = test[feature_cols].values
            y_test = test[target_col].values
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            maes.append(np.mean(np.abs(pred - y_test)))
        return np.mean(maes) if maes else float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def fit_lgbm_model(df: pd.DataFrame, target_col: str = "avg_epex_p_kwh",
                    feature_list: list | None = None,
                    params_override: dict | None = None):
    """
    Fit a LightGBM regressor on daily data.
    Returns (model, feature_cols, r2).
    """
    features = feature_list or DAILY_FEATURES
    feature_cols = [c for c in features if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols + [target_col])
    X = df_model[feature_cols].values
    y = df_model[target_col].values

    params = {**LGBM_PARAMS, **(params_override or {})}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return model, feature_cols, r2


def fit_lgbm_quantile(df: pd.DataFrame, feature_cols: list[str], quantile: float,
                       target_col: str = "avg_epex_p_kwh"):
    """Fit a LightGBM quantile regression model. Returns the fitted model."""
    df_model = df.dropna(subset=feature_cols + [target_col])
    X = df_model[feature_cols].values
    y = df_model[target_col].values

    params = {**LGBM_PARAMS, "objective": "quantile", "alpha": quantile}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def fit_ensemble(df: pd.DataFrame, lgbm_params: dict | None = None) -> dict:
    """
    Fit Ridge + LightGBM + quantile models and determine optimal blend weight
    via internal cross-validation.

    Returns dict with all fitted components:
      ridge: {model, scaler, feature_cols}
      lgbm:  {model, feature_cols}
      lgbm_q10, lgbm_q90: {model, feature_cols}
      blend_weight: float (Ridge weight; LightGBM = 1 - weight)
      r2_ridge, r2_lgbm: in-sample R² for each
    """
    # Fit both models on full data
    ridge_model, ridge_scaler, r2_ridge, ridge_fcols = fit_model(df)
    lgbm_model, lgbm_fcols, r2_lgbm = fit_lgbm_model(df, params_override=lgbm_params)

    # Quantile models for prediction intervals
    lgbm_q10 = fit_lgbm_quantile(df, lgbm_fcols, quantile=0.10)
    lgbm_q90 = fit_lgbm_quantile(df, lgbm_fcols, quantile=0.90)

    # Determine blend weight via walk-forward on the last 120 days
    blend_weight = _find_blend_weight(df, n_folds=4, min_train_days=120)

    return {
        "ridge":      {"model": ridge_model, "scaler": ridge_scaler, "feature_cols": ridge_fcols, "r2": r2_ridge},
        "lgbm":       {"model": lgbm_model, "feature_cols": lgbm_fcols, "r2": r2_lgbm},
        "lgbm_q10":   {"model": lgbm_q10, "feature_cols": lgbm_fcols},
        "lgbm_q90":   {"model": lgbm_q90, "feature_cols": lgbm_fcols},
        "blend_weight": blend_weight,
        "r2_ridge":   r2_ridge,
        "r2_lgbm":    r2_lgbm,
    }


def _find_blend_weight(df: pd.DataFrame, n_folds: int = 4,
                        min_train_days: int = 120) -> float:
    """
    Find the optimal Ridge/LightGBM blend weight via walk-forward CV.
    Returns the Ridge weight (0.0 = pure LightGBM, 1.0 = pure Ridge).
    """
    if len(df) < min_train_days + n_folds * 10:
        return 0.3  # default: lean toward LightGBM

    test_size = (len(df) - min_train_days) // n_folds
    if test_size < 5:
        return 0.3

    weight_scores: dict[float, list] = {w / 10: [] for w in range(11)}

    for fold in range(n_folds):
        cutoff = min_train_days + fold * test_size
        if cutoff >= len(df) - 5:
            break
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:cutoff + test_size]
        if len(test) < 3:
            break

        try:
            r_model, r_scaler, _, r_fcols = fit_model(train)
            l_model, l_fcols, _ = fit_lgbm_model(train)
        except Exception:
            continue

        # Predict on test set
        r_cols = [c for c in r_fcols if c in test.columns]
        l_cols = [c for c in l_fcols if c in test.columns]
        test_clean = test.dropna(subset=list(set(r_cols + l_cols + ["avg_epex_p_kwh"])))
        if len(test_clean) < 3:
            continue

        X_r = test_clean[r_cols].values
        X_l = test_clean[l_cols].values
        y_true = test_clean["avg_epex_p_kwh"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_r = r_model.predict(r_scaler.transform(X_r))
            pred_l = l_model.predict(X_l)

        for w_int in range(11):
            w = w_int / 10
            blended = w * pred_r + (1 - w) * pred_l
            mae = np.mean(np.abs(y_true - blended))
            weight_scores[w].append(mae)

    # Pick weight with lowest mean MAE
    best_w, best_mae = 0.3, float("inf")
    for w, scores in weight_scores.items():
        if scores:
            mean_mae = np.mean(scores)
            if mean_mae < best_mae:
                best_w, best_mae = w, mean_mae

    return best_w


def predict_ensemble(forecast_df: pd.DataFrame, ensemble: dict,
                      latest_commodity: dict | None = None,
                      site_forecasts: dict | None = None,
                      df_historical: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Predict using the Ridge + LightGBM ensemble. Returns a DataFrame with
    predicted_epex_p_kwh (blended), pred_ridge, pred_lgbm, pred_q10, pred_q90.
    """
    w = ensemble["blend_weight"]
    ridge = ensemble["ridge"]
    lgbm = ensemble["lgbm"]
    q10_model = ensemble["lgbm_q10"]["model"]
    q90_model = ensemble["lgbm_q90"]["model"]

    # Get Ridge predictions
    ridge_pred = predict_from_forecast(
        forecast_df, ridge["model"], ridge["scaler"], ridge["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
    )

    # Get LightGBM predictions — reuse the feature-prepared df from Ridge path
    lgbm_pred = predict_from_forecast(
        forecast_df, lgbm["model"], None, lgbm["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )

    if ridge_pred.empty or lgbm_pred.empty:
        return ridge_pred if not ridge_pred.empty else lgbm_pred

    result = ridge_pred[["date"]].copy()
    result["pred_ridge"] = ridge_pred["predicted_epex_p_kwh"].values
    result["pred_lgbm"] = lgbm_pred["predicted_epex_p_kwh"].values
    result["predicted_epex_p_kwh"] = w * result["pred_ridge"] + (1 - w) * result["pred_lgbm"]

    # Quantile predictions
    q10_pred = predict_from_forecast(
        forecast_df, q10_model, None, ensemble["lgbm_q10"]["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )
    q90_pred = predict_from_forecast(
        forecast_df, q90_model, None, ensemble["lgbm_q90"]["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )
    if not q10_pred.empty:
        result["pred_q10"] = q10_pred["predicted_epex_p_kwh"].values
    if not q90_pred.empty:
        result["pred_q90"] = q90_pred["predicted_epex_p_kwh"].values

    return result


def predict_from_forecast(forecast_df: pd.DataFrame,
                          model,
                          scaler: StandardScaler | None,
                          feature_cols: list[str],
                          latest_commodity: dict | None = None,
                          site_forecasts: dict | None = None,
                          df_historical: pd.DataFrame | None = None,
                          use_scaler: bool = True) -> pd.DataFrame:
    """
    Apply trained model to forecast weather.
    latest_commodity: dict of {feature_col: value} for commodity rolling averages —
    the most recent known values are held constant across the forecast horizon.
    site_forecasts: dict of {site_id: DataFrame} with wind speed forecasts per site.
    """
    fc = forecast_df.copy()
    for col in COMMODITY_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)
    for col in INVENTORY_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)
    for col in LAG_ROLLING_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)
    for col in ENTSOE_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)
    for col in SYSTEM_PRICE_FEATURES:
        if col in feature_cols:
            fc[col] = (latest_commodity or {}).get(col, np.nan)
    for col in RAMP_FEATURES:
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

    # Estimate wind_gen_mw for forecast from wind speed (polynomial model on history)
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

    # heating_dd: derived from forecast temperature — no estimation needed
    if "heating_dd" in feature_cols and "heating_dd" not in fc.columns:
        fc["heating_dd"] = (15.5 - fc["temperature_2m"]).clip(lower=0.0)

    # Calendar features: deterministic from date
    fc["date"] = pd.to_datetime(fc["date"])
    if "is_bank_holiday" in feature_cols:
        bh = _uk_holidays()
        fc["is_bank_holiday"] = fc["date"].apply(lambda d: 1 if d.date() in bh else 0)
    if "is_weekend" in feature_cols:
        fc["is_weekend"] = (fc["date"].dt.dayofweek >= 5).astype(int)

    # epex_lag1_gbp_mwh: use most recent known EPEX price as a constant estimate.
    # Day-ahead prices are autocorrelated; the most recent known price is a better
    # prior than zero for the 7-day horizon.
    if "epex_lag1_gbp_mwh" in feature_cols:
        if "epex_lag1_gbp_mwh" not in fc.columns or fc["epex_lag1_gbp_mwh"].isna().all():
            if df_historical is not None and "epex_lag1_gbp_mwh" in df_historical.columns:
                last_known = df_historical["epex_lag1_gbp_mwh"].dropna()
                epex_const = float(last_known.iloc[-1]) if len(last_known) > 0 else np.nan
            else:
                epex_const = np.nan
            fc["epex_lag1_gbp_mwh"] = epex_const

    # Interaction terms using mean of wind-farm sites as the wind signal
    wind_site_cols = [c for c in fc.columns if c.startswith("wind_") and c in feature_cols]
    uk_avg_wind = fc[wind_site_cols].mean(axis=1) if wind_site_cols else pd.Series(0.0, index=fc.index)
    solar_signal = fc["solar_gw"] if "solar_gw" in fc.columns else fc.get("shortwave_radiation", pd.Series(0.0, index=fc.index))
    fc["temp_x_wind"]  = fc["temperature_2m"] * uk_avg_wind
    fc["wind_x_solar"] = uk_avg_wind * solar_signal.fillna(0.0)

    missing = fc[feature_cols].isna().any()
    if missing.any():
        import warnings
        warnings.warn(
            f"predict_from_forecast: NaN in features {missing[missing].index.tolist()}; "
            "filling with 0.0 — predictions may be inaccurate",
            RuntimeWarning, stacklevel=2,
        )
    X_fc = fc[feature_cols].fillna(0.0).values
    if use_scaler and scaler is not None:
        X_fc = scaler.transform(X_fc)
    preds = model.predict(X_fc)

    result = forecast_df[["date"]].copy()
    result["predicted_epex_p_kwh"] = preds
    return result


# ── Half-hourly model ──────────────────────────────────────────────────────────

def fit_halfhourly_model(df: pd.DataFrame):
    """
    Train a Ridge regression on half-hourly data.
    Returns (model, scaler, r2, feature_cols).
    """
    feature_cols = [c for c in HH_FEATURES if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols + ["epex_price_p_kwh"])
    X = df_model[feature_cols].values
    y = df_model["epex_price_p_kwh"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return model, scaler, r2, feature_cols


# HH LightGBM params — more data (~17k rows), so can afford slightly more capacity.
LGBM_HH_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)


def fit_halfhourly_lgbm(df: pd.DataFrame):
    """
    Fit a LightGBM regressor on half-hourly data.
    Returns (model, feature_cols, r2).
    """
    feature_cols = [c for c in HH_FEATURES if c in df.columns and df[c].notna().any()]
    df_model = df.dropna(subset=feature_cols + ["epex_price_p_kwh"])
    X = df_model[feature_cols].values
    y = df_model["epex_price_p_kwh"].values

    model = lgb.LGBMRegressor(**LGBM_HH_PARAMS)
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return model, feature_cols, r2


def fit_halfhourly_lgbm_quantile(df: pd.DataFrame, feature_cols: list[str], quantile: float):
    """Fit a LightGBM quantile regression model on half-hourly data."""
    df_model = df.dropna(subset=feature_cols + ["epex_price_p_kwh"])
    X = df_model[feature_cols].values
    y = df_model["epex_price_p_kwh"].values

    params = {**LGBM_HH_PARAMS, "objective": "quantile", "alpha": quantile}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def fit_halfhourly_ensemble(df_hh: pd.DataFrame) -> dict:
    """
    Fit Ridge + LightGBM ensemble on half-hourly data.
    Returns dict with ridge, lgbm, lgbm_q10, lgbm_q90, blend_weight.
    """
    ridge_model, ridge_scaler, r2_ridge, ridge_fcols = fit_halfhourly_model(df_hh)
    lgbm_model, lgbm_fcols, r2_lgbm = fit_halfhourly_lgbm(df_hh)

    lgbm_q10 = fit_halfhourly_lgbm_quantile(df_hh, lgbm_fcols, quantile=0.10)
    lgbm_q90 = fit_halfhourly_lgbm_quantile(df_hh, lgbm_fcols, quantile=0.90)

    blend_weight = _find_hh_blend_weight(df_hh, n_folds=4, min_train_days=120)

    return {
        "ridge":      {"model": ridge_model, "scaler": ridge_scaler, "feature_cols": ridge_fcols, "r2": r2_ridge},
        "lgbm":       {"model": lgbm_model, "feature_cols": lgbm_fcols, "r2": r2_lgbm},
        "lgbm_q10":   {"model": lgbm_q10, "feature_cols": lgbm_fcols},
        "lgbm_q90":   {"model": lgbm_q90, "feature_cols": lgbm_fcols},
        "blend_weight": blend_weight,
    }


def _find_hh_blend_weight(df_hh: pd.DataFrame, n_folds: int = 4,
                            min_train_days: int = 120) -> float:
    """
    Find optimal Ridge/LightGBM blend weight for HH model via walk-forward CV.
    Splits on day boundaries to avoid data leakage.
    """
    if df_hh.empty or "datetime_local" not in df_hh.columns:
        return 0.3

    dates = sorted(df_hh["datetime_local"].dt.date.unique())
    if len(dates) < min_train_days + n_folds * 5:
        return 0.3

    test_days = (len(dates) - min_train_days) // n_folds
    if test_days < 5:
        return 0.3

    weight_scores: dict[float, list] = {w / 10: [] for w in range(11)}

    for fold in range(n_folds):
        cutoff_idx = min_train_days + fold * test_days
        if cutoff_idx >= len(dates) - 3:
            break
        cutoff_date = dates[cutoff_idx]
        end_date = dates[min(cutoff_idx + test_days, len(dates) - 1)]

        train = df_hh[df_hh["datetime_local"].dt.date < cutoff_date].copy()
        test = df_hh[(df_hh["datetime_local"].dt.date >= cutoff_date) &
                     (df_hh["datetime_local"].dt.date <= end_date)].copy()

        if len(train) < 1000 or len(test) < 100:
            continue

        try:
            r_model, r_scaler, _, r_fcols = fit_halfhourly_model(train)
            l_model, l_fcols, _ = fit_halfhourly_lgbm(train)
        except Exception:
            continue

        common_cols = [c for c in set(r_fcols) | set(l_fcols) if c in test.columns]
        test_clean = test.dropna(subset=common_cols + ["epex_price_p_kwh"])
        if len(test_clean) < 50:
            continue

        y_true = test_clean["epex_price_p_kwh"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_cols = [c for c in r_fcols if c in test_clean.columns]
            l_cols = [c for c in l_fcols if c in test_clean.columns]
            pred_r = r_model.predict(r_scaler.transform(test_clean[r_cols].fillna(0).values))
            pred_l = l_model.predict(test_clean[l_cols].fillna(0).values)

        for w_int in range(11):
            w = w_int / 10
            blended = w * pred_r + (1 - w) * pred_l
            mae = np.mean(np.abs(y_true - blended))
            weight_scores[w].append(mae)

    best_w, best_mae = 0.3, float("inf")
    for w, scores in weight_scores.items():
        if scores:
            mean_mae = np.mean(scores)
            if mean_mae < best_mae:
                best_w, best_mae = w, mean_mae

    return best_w


def predict_halfhourly_forecast(
    forecast_hourly_df: pd.DataFrame,
    model,
    scaler: StandardScaler | None,
    feature_cols: list[str] | None = None,
    latest_commodity: dict | None = None,
    site_forecasts: dict | None = None,
    df_historical: pd.DataFrame | None = None,
    use_scaler: bool = True,
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
    need_solar         = "solar_gw"          in cols_needed
    need_demand        = "demand_mw"         in cols_needed
    need_wind_gen      = "wind_gen_mw"       in cols_needed
    need_gas_gen       = "gas_gen_mw"        in cols_needed
    need_nuclear       = "nuclear_mw"        in cols_needed
    need_pumped        = "pumped_storage_mw" in cols_needed
    need_hydro         = "hydro_mw"          in cols_needed
    need_imports       = "imports_mw"        in cols_needed
    need_heating_dd    = "heating_dd"        in cols_needed
    need_bank_holiday  = "is_bank_holiday"   in cols_needed
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

    # Pre-compute gas generation estimates from historical (day_of_week, hour) profile
    gas_gen_profile: dict = {}
    if need_gas_gen and df_historical is not None and "gas_gen_mw" in df_historical.columns:
        d = df_historical.dropna(subset=["gas_gen_mw"]).copy()
        d["_dow"]  = d["datetime_local"].dt.dayofweek
        d["_hour"] = d["datetime_local"].dt.hour
        gas_gen_profile = d.groupby(["_dow", "_hour"])["gas_gen_mw"].mean().to_dict()

    # Nuclear is stable — use historical mean
    nuclear_mean_val: float = np.nan
    if need_nuclear and df_historical is not None and "nuclear_mw" in df_historical.columns:
        nuclear_mean_val = float(df_historical["nuclear_mw"].dropna().mean())

    # Pumped storage and hydro profiles by (day_of_week, hour)
    pumped_profile: dict = {}
    hydro_profile:  dict = {}
    for col, profile_dict, need in [
        ("pumped_storage_mw", pumped_profile, need_pumped),
        ("hydro_mw",          hydro_profile,  need_hydro),
    ]:
        if need and df_historical is not None and col in df_historical.columns:
            d = df_historical.dropna(subset=[col]).copy()
            d["_dow"]  = d["datetime_local"].dt.dayofweek
            d["_hour"] = d["datetime_local"].dt.hour
            profile_dict.update(d.groupby(["_dow", "_hour"])[col].mean().to_dict())

    bh_set = _uk_holidays() if need_bank_holiday else set()

    # Per-slot price lag: yesterday's EPEX price at the same half-hour slot.
    # Build a lookup keyed by (hour, minute) from the most recent 24h of historical data.
    need_price_lag = "price_lag1_slot" in cols_needed
    price_lag_by_slot: dict[tuple, float] = {}
    if need_price_lag and df_historical is not None and "epex_price_p_kwh" in df_historical.columns:
        recent = df_historical.sort_values("datetime_local").tail(48)
        for _, r in recent.iterrows():
            price_lag_by_slot[(r["datetime_local"].hour, r["datetime_local"].minute)] = r["epex_price_p_kwh"]

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
            demand_mw         = demand_profile.get((dow, hour), np.nan) if need_demand else None
            wind_gen_mw       = wind_gen_est_by_hour.get(hour_key, np.nan) if need_wind_gen else None
            gas_gen_mw        = gas_gen_profile.get((dow, hour), np.nan) if need_gas_gen else None
            nuclear_mw        = nuclear_mean_val if need_nuclear else None
            pumped_storage_mw = pumped_profile.get((dow, hour), np.nan) if need_pumped else None
            hydro_mw          = hydro_profile.get((dow, hour), np.nan) if need_hydro else None
            imports_mw        = imports_profile.get((dow, hour), np.nan) if need_imports else None
            heating_dd        = max(0.0, 15.5 - row["temperature_2m"]) if need_heating_dd else None
            is_bh             = (1 if dt_local.date() in bh_set else 0) if need_bank_holiday else None
            is_wkend = 1 if dt_local.weekday() >= 5 else 0
            rec = {
                "datetime_local":      dt_local,
                "is_peak":             is_peak,
                "temperature_2m":      row["temperature_2m"],
                "heating_dd":          heating_dd,
                "shortwave_radiation": row["shortwave_radiation"],
                "precipitation":       row["precipitation"],
                "solar_gw":            solar_gw,
                "demand_mw":           demand_mw,
                "wind_gen_mw":         wind_gen_mw,
                "gas_gen_mw":          gas_gen_mw,
                "nuclear_mw":          nuclear_mw,
                "pumped_storage_mw":   pumped_storage_mw,
                "hydro_mw":            hydro_mw,
                "imports_mw":          imports_mw,
                "is_bank_holiday":     is_bh,
                "is_weekend":          is_wkend,
                "price_lag1_slot":     price_lag_by_slot.get((dt_local.hour, dt_local.minute), np.nan) if need_price_lag else None,
                "net_residual_mw":     (
                    (demand_mw or 0.0)
                    - (wind_gen_mw or 0.0)
                    - ((solar_gw or 0.0) * 1000)
                    - (nuclear_mw or 0.0)
                    - (hydro_mw or 0.0)
                ),
                "temp_x_wind":         row["temperature_2m"] * uk_avg_wind,
                "wind_x_solar":        uk_avg_wind * (solar_signal or 0.0),
                "brent_roll7":          (latest_commodity or {}).get("brent_roll7"),
                "gas_ttf_roll7":        (latest_commodity or {}).get("gas_ttf_roll7"),
                "gbpusd_roll7":         (latest_commodity or {}).get("gbpusd_roll7"),
                "dxy_roll7":            (latest_commodity or {}).get("dxy_roll7"),
                "eu_gas_storage_pct":    (latest_commodity or {}).get("eu_gas_storage_pct"),
                "gb_gas_storage_pct":    (latest_commodity or {}).get("gb_gas_storage_pct"),
                "us_crude_stocks_delta": (latest_commodity or {}).get("us_crude_stocks_delta"),
                "epex_lag7_gbp_mwh":     (latest_commodity or {}).get("epex_lag7_gbp_mwh"),
                "epex_roll7_std":        (latest_commodity or {}).get("epex_roll7_std"),
                "epex_roll7_min":        (latest_commodity or {}).get("epex_roll7_min"),
                "epex_roll7_max":        (latest_commodity or {}).get("epex_roll7_max"),
                "epex_momentum_7":       (latest_commodity or {}).get("epex_momentum_7"),
                "carbon_roll7":             (latest_commodity or {}).get("carbon_roll7"),
                "net_scheduled_imports_mw": (latest_commodity or {}).get("net_scheduled_imports_mw", 0.0),
                "nuclear_unavailable_mw":   (latest_commodity or {}).get("nuclear_unavailable_mw", 0.0),
                "total_unavailable_mw":     (latest_commodity or {}).get("total_unavailable_mw", 0.0),
                "sysprice_lag1_gbp_mwh":    (latest_commodity or {}).get("sysprice_lag1_gbp_mwh"),
                "abs_imbalance_lag1_mw":    (latest_commodity or {}).get("abs_imbalance_lag1_mw"),
                "wind_ramp_mw":             (latest_commodity or {}).get("wind_ramp_mw", 0.0),
                "solar_ramp_gw":            (latest_commodity or {}).get("solar_ramp_gw", 0.0),
                **site_wind_vals,
                **_time_features(dt_local),
            }
            records.append(rec)

    df = pd.DataFrame(records)
    cols = feature_cols if feature_cols is not None else HH_FEATURES
    cols = [c for c in cols if c in df.columns]

    # ── Recursive day-by-day forecasting ──────────────────────────────────────
    # Day 1: uses actual yesterday's prices as price_lag1_slot (already set).
    # Day 2+: uses the previous day's *predicted* prices as the lag.
    df["predicted_epex_p_kwh"] = np.nan
    df["_date"] = df["datetime_local"].dt.date
    forecast_dates = sorted(df["_date"].unique())

    for i, fdate in enumerate(forecast_dates):
        day_mask = df["_date"] == fdate

        if i > 0 and "price_lag1_slot" in cols:
            # Replace price_lag1_slot with yesterday's predictions
            prev_date = forecast_dates[i - 1]
            prev_mask = df["_date"] == prev_date
            prev_preds = df.loc[prev_mask, ["datetime_local", "predicted_epex_p_kwh"]].copy()
            # Build lookup: (hour, minute) → predicted price from previous day
            pred_by_slot = {
                (r["datetime_local"].hour, r["datetime_local"].minute): r["predicted_epex_p_kwh"]
                for _, r in prev_preds.iterrows()
            }
            df.loc[day_mask, "price_lag1_slot"] = df.loc[day_mask, "datetime_local"].apply(
                lambda dt: pred_by_slot.get((dt.hour, dt.minute), np.nan)
            )

        day_X = df.loc[day_mask, cols].fillna(0.0).values
        if use_scaler and scaler is not None:
            day_X = scaler.transform(day_X)
        df.loc[day_mask, "predicted_epex_p_kwh"] = model.predict(day_X)

    df.drop(columns=["_date"], inplace=True)
    return df


def predict_halfhourly_ensemble(
    forecast_hourly_df: pd.DataFrame,
    hh_ensemble: dict,
    latest_commodity: dict | None = None,
    site_forecasts: dict | None = None,
    df_historical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Predict half-hourly prices using the Ridge + LightGBM ensemble.
    Returns a DataFrame with predicted_epex_p_kwh (blended), pred_ridge, pred_lgbm,
    pred_q10, pred_q90, and the standard HH columns.
    """
    w = hh_ensemble["blend_weight"]
    ridge = hh_ensemble["ridge"]
    lgbm = hh_ensemble["lgbm"]

    ridge_pred = predict_halfhourly_forecast(
        forecast_hourly_df, ridge["model"], ridge["scaler"], ridge["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=True,
    )

    lgbm_pred = predict_halfhourly_forecast(
        forecast_hourly_df, lgbm["model"], None, lgbm["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )

    if ridge_pred.empty or lgbm_pred.empty:
        return ridge_pred if not ridge_pred.empty else lgbm_pred

    result = ridge_pred.copy()
    result["pred_ridge"] = ridge_pred["predicted_epex_p_kwh"].values
    result["pred_lgbm"] = lgbm_pred["predicted_epex_p_kwh"].values
    result["predicted_epex_p_kwh"] = w * result["pred_ridge"] + (1 - w) * result["pred_lgbm"]

    # Quantile predictions
    q10_pred = predict_halfhourly_forecast(
        forecast_hourly_df, hh_ensemble["lgbm_q10"]["model"], None,
        hh_ensemble["lgbm_q10"]["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )
    q90_pred = predict_halfhourly_forecast(
        forecast_hourly_df, hh_ensemble["lgbm_q90"]["model"], None,
        hh_ensemble["lgbm_q90"]["feature_cols"],
        latest_commodity, site_forecasts, df_historical=df_historical,
        use_scaler=False,
    )
    if not q10_pred.empty:
        result["pred_q10"] = q10_pred["predicted_epex_p_kwh"].values
    if not q90_pred.empty:
        result["pred_q90"] = q90_pred["predicted_epex_p_kwh"].values

    return result


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, correlations: dict, r2: float,
                  model, scaler, feature_cols: list[str],
                  predictions: pd.DataFrame, r2_hh: float | None = None) -> None:
    """Print a formatted summary to the terminal."""
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Date range        : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Days              : {len(df)}")
    print(f"  Mean EPEX price   : {df['avg_epex_p_kwh'].mean():.2f} p/kWh  "
          f"(wholesale day-ahead, ex-VAT, ex-network charges)")
    print(f"  EPEX range        : {df['avg_epex_p_kwh'].min():.2f} – "
          f"{df['avg_epex_p_kwh'].max():.2f} p/kWh")

    print("\n" + "=" * 65)
    print("PEARSON CORRELATIONS  (EPEX wholesale price vs weather)")
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
        print(f"  {ds:<25s}  EPEX ~{row['predicted_epex_p_kwh']:+.2f} p/kWh")
    print("=" * 65)
