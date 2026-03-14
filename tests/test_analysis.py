"""
Tests for app/analysis.py — covering feature engineering, model fitting,
prediction, and backtest pipeline.

Uses synthetic DataFrames where possible to avoid database dependencies.
Integration tests (marked with @pytest.mark.integration) hit the real SQLite DB.
"""
import math
import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from app.analysis import (
    DAILY_FEATURES,
    HH_FEATURES,
    _time_features,
    build_demand_profile,
    build_imports_profile,
    estimate_solar_from_radiation,
    estimate_wind_gen_from_speed,
    fit_halfhourly_model,
    fit_model,
    predict_from_forecast,
    run_backtest,
    run_halfhourly_backtest,
)

LOCAL_TZ = ZoneInfo("Europe/London")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_daily_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily DataFrame covering all DAILY_FEATURES."""
    rng = np.random.default_rng(seed)
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({
        "date":               pd.to_datetime(dates),
        "avg_epex_p_kwh":     rng.uniform(3, 15, n),
        "temperature_2m":     rng.uniform(0, 20, n),
        "heating_dd":         rng.uniform(0, 15, n),
        "shortwave_radiation": rng.uniform(0, 300, n),
        "precipitation":      rng.uniform(0, 5, n),
        "solar_gw":           rng.uniform(0, 5, n),
        "wind_gen_mw":        rng.uniform(2000, 15000, n),
        "is_bank_holiday":    rng.integers(0, 2, n),
        "is_weekend":         [1 if (date(2025, 1, 1) + timedelta(days=i)).weekday() >= 5 else 0
                               for i in range(n)],
        "epex_lag1_gbp_mwh":  rng.uniform(50, 120, n),
        "gas_ttf_roll7":      rng.uniform(20, 60, n),
        "brent_roll7":        rng.uniform(60, 100, n),
        "temp_x_wind":        rng.uniform(0, 1000, n),
        "wind_x_solar":       rng.uniform(0, 500, n),
        "wind_dogger_bank":   rng.uniform(5, 80, n),
        "wind_hornsea":       rng.uniform(5, 80, n),
        "wind_walney":        rng.uniform(5, 80, n),
        "wind_whitelee":      rng.uniform(5, 80, n),
        "wind_clyde_wind":    rng.uniform(5, 80, n),
        "wind_pen_y_cymoedd": rng.uniform(5, 80, n),
    })
    return df


def _make_hh_df(n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic half-hourly DataFrame covering all HH_FEATURES."""
    rng = np.random.default_rng(seed)
    base = datetime(2025, 1, 1, tzinfo=LOCAL_TZ)
    slots = n_days * 48
    datetimes = [base + timedelta(minutes=30 * i) for i in range(slots)]

    wind_gen = rng.uniform(2000, 14000, slots)
    solar_gw = np.array([max(0.0, rng.normal(3 if 8 <= dt.hour <= 16 else 0, 1))
                          for dt in datetimes])
    demand = rng.uniform(25000, 40000, slots)
    nuclear = rng.uniform(4000, 6000, slots)
    hydro = rng.uniform(100, 800, slots)

    df = pd.DataFrame({
        "datetime_local":     datetimes,
        "epex_price_p_kwh":   rng.uniform(3, 20, slots),
        "is_peak":            [1 if 16 <= dt.hour < 19 else 0 for dt in datetimes],
        "temperature_2m":     rng.uniform(0, 20, slots),
        "heating_dd":         rng.uniform(0, 15, slots),
        "precipitation":      rng.uniform(0, 0.5, slots),
        "solar_gw":           solar_gw,
        "wind_gen_mw":        wind_gen,
        "net_residual_mw":    demand - wind_gen - solar_gw * 1000 - nuclear - hydro,
        "gas_ttf_roll7":      rng.uniform(20, 60, slots),
        "brent_roll7":        rng.uniform(60, 100, slots),
        "price_lag1_slot":    rng.uniform(5, 35, slots),
        "wind_dogger_bank":   rng.uniform(5, 80, slots),
        "wind_hornsea":       rng.uniform(5, 80, slots),
        "wind_walney":        rng.uniform(5, 80, slots),
        "wind_whitelee":      rng.uniform(5, 80, slots),
        "wind_clyde_wind":    rng.uniform(5, 80, slots),
        "wind_pen_y_cymoedd": rng.uniform(5, 80, slots),
        "temp_x_wind":        rng.uniform(0, 1000, slots),
        "wind_x_solar":       rng.uniform(0, 500, slots),
        "is_bank_holiday":    rng.integers(0, 2, slots),
        "is_weekend":         [1 if dt.weekday() >= 5 else 0 for dt in datetimes],
        "hour_sin":           [math.sin(2 * math.pi * dt.hour / 24) for dt in datetimes],
        "hour_cos":           [math.cos(2 * math.pi * dt.hour / 24) for dt in datetimes],
        "doy_sin":            [math.sin(2 * math.pi * dt.timetuple().tm_yday / 365)
                               for dt in datetimes],
        "doy_cos":            [math.cos(2 * math.pi * dt.timetuple().tm_yday / 365)
                               for dt in datetimes],
    })
    return df


# ── _time_features ─────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_midnight_hour_sin_zero(self):
        dt = datetime(2025, 6, 1, 0, 0, tzinfo=LOCAL_TZ)
        tf = _time_features(dt)
        assert abs(tf["hour_sin"]) < 1e-10
        assert abs(tf["hour_cos"] - 1.0) < 1e-10

    def test_noon_hour_sin_zero_cos_minus_one(self):
        dt = datetime(2025, 6, 1, 12, 0, tzinfo=LOCAL_TZ)
        tf = _time_features(dt)
        assert abs(tf["hour_sin"]) < 1e-10
        assert abs(tf["hour_cos"] - (-1.0)) < 1e-10

    def test_unit_circle_constraint(self):
        """sin² + cos² == 1 for any datetime."""
        for hour in [0, 6, 12, 18, 23]:
            dt = datetime(2025, 3, 15, hour, 30, tzinfo=LOCAL_TZ)
            tf = _time_features(dt)
            assert abs(tf["hour_sin"] ** 2 + tf["hour_cos"] ** 2 - 1.0) < 1e-10
            assert abs(tf["doy_sin"]  ** 2 + tf["doy_cos"]  ** 2 - 1.0) < 1e-10

    def test_half_hour_offset(self):
        """00:30 should have a different sin than 00:00."""
        t0 = _time_features(datetime(2025, 1, 1, 0, 0,  tzinfo=LOCAL_TZ))
        t1 = _time_features(datetime(2025, 1, 1, 0, 30, tzinfo=LOCAL_TZ))
        assert t0["hour_sin"] != t1["hour_sin"]

    def test_returns_four_keys(self):
        dt = datetime(2025, 6, 1, 16, 0, tzinfo=LOCAL_TZ)
        tf = _time_features(dt)
        assert set(tf.keys()) == {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}


# ── fit_model (daily Ridge) ────────────────────────────────────────────────────

class TestFitModel:
    def test_returns_four_tuple(self):
        df = _make_daily_df()
        result = fit_model(df)
        assert len(result) == 4

    def test_r2_in_range(self):
        df = _make_daily_df()
        _, _, r2, _ = fit_model(df)
        assert 0.0 <= r2 <= 1.0

    def test_r2_zero_when_all_prices_identical(self):
        df = _make_daily_df()
        df["avg_epex_p_kwh"] = 8.0  # flat price — no variance to explain
        _, _, r2, _ = fit_model(df)
        assert r2 == 0.0  # ss_tot == 0 → guarded fallback

    def test_feature_cols_subset_of_daily_features(self):
        df = _make_daily_df()
        _, _, _, feature_cols = fit_model(df)
        assert all(f in DAILY_FEATURES for f in feature_cols)

    def test_predictions_are_finite(self):
        df = _make_daily_df()
        model, scaler, _, feature_cols = fit_model(df)
        X = df.dropna(subset=feature_cols)[feature_cols].values
        preds = model.predict(scaler.transform(X))
        assert np.all(np.isfinite(preds))

    def test_no_non_forecastable_features(self):
        """Daily model should not include features that are not available at forecast time."""
        excluded = {"demand_mw", "gas_gen_mw", "nuclear_mw", "pumped_storage_mw",
                     "hydro_mw", "imports_mw"}
        assert excluded.isdisjoint(set(DAILY_FEATURES))

    def test_is_weekend_in_daily_features(self):
        assert "is_weekend" in DAILY_FEATURES


# ── fit_halfhourly_model ───────────────────────────────────────────────────────

class TestFitHalfhourlyModel:
    def test_returns_four_tuple(self):
        df = _make_hh_df()
        result = fit_halfhourly_model(df)
        assert len(result) == 4

    def test_r2_in_range(self):
        df = _make_hh_df()
        _, _, r2, _ = fit_halfhourly_model(df)
        assert 0.0 <= r2 <= 1.0

    def test_r2_zero_when_all_prices_identical(self):
        df = _make_hh_df()
        df["epex_price_p_kwh"] = 10.0  # constant target → R² should be 0
        _, _, r2, _ = fit_halfhourly_model(df)
        assert r2 == 0.0

    def test_feature_cols_subset_of_hh_features(self):
        df = _make_hh_df()
        _, _, _, feature_cols = fit_halfhourly_model(df)
        assert all(f in HH_FEATURES for f in feature_cols)

    def test_scaler_is_not_none(self):
        df = _make_hh_df()
        _, scaler, _, _ = fit_halfhourly_model(df)
        assert scaler is not None

    def test_predictions_are_finite(self):
        df = _make_hh_df()
        model, scaler, _, feature_cols = fit_halfhourly_model(df)
        X = df.dropna(subset=feature_cols)[feature_cols].values
        preds = model.predict(scaler.transform(X))
        assert np.all(np.isfinite(preds))

    def test_net_residual_in_hh_features(self):
        assert "net_residual_mw" in HH_FEATURES

    def test_is_weekend_in_hh_features(self):
        assert "is_weekend" in HH_FEATURES


# ── predict_from_forecast (daily) ─────────────────────────────────────────────

class TestPredictFromForecast:
    def _trained(self, n=120):
        df = _make_daily_df(n)
        model, scaler, _, feature_cols = fit_model(df)
        return df, model, scaler, feature_cols

    def test_returns_dataframe_with_predictions(self):
        df, model, scaler, feature_cols = self._trained()
        fc = df.tail(7).copy()
        result = predict_from_forecast(fc, model, scaler, feature_cols)
        assert "predicted_epex_p_kwh" in result.columns
        assert len(result) == 7

    def test_predictions_are_finite(self):
        df, model, scaler, feature_cols = self._trained()
        fc = df.tail(7).copy()
        result = predict_from_forecast(fc, model, scaler, feature_cols)
        assert result["predicted_epex_p_kwh"].notna().all()

    def test_epex_predictions_positive(self):
        # Wholesale prices can technically go negative, but for a typical week should be positive
        df, model, scaler, feature_cols = self._trained()
        fc = df.tail(7).copy()
        result = predict_from_forecast(fc, model, scaler, feature_cols)
        assert result["predicted_epex_p_kwh"].mean() > 0

    def test_warns_on_missing_features(self):
        df, model, scaler, feature_cols = self._trained()
        fc = df.tail(7).copy()
        # Intentionally null out a feature
        fc["gas_ttf_roll7"] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predict_from_forecast(fc, model, scaler, feature_cols)
        assert any("NaN in features" in str(x.message) for x in w)


# ── run_backtest (daily) ───────────────────────────────────────────────────────

class TestRunBacktest:
    def test_returns_tuple_of_two(self):
        df = _make_daily_df(90)
        result = run_backtest(df, holdout_days=30)
        assert len(result) == 2

    def test_metrics_contain_expected_keys(self):
        df = _make_daily_df(90)
        _, metrics = run_backtest(df, holdout_days=30)
        for key in ("mae", "rmse", "mape", "r2"):
            assert key in metrics

    def test_mae_is_positive(self):
        df = _make_daily_df(90)
        _, metrics = run_backtest(df, holdout_days=30)
        assert metrics["mae"] >= 0

    def test_too_little_data_returns_empty(self):
        df = _make_daily_df(20)
        bt_df, metrics = run_backtest(df, holdout_days=30)
        assert bt_df.empty


# ── run_halfhourly_backtest ────────────────────────────────────────────────────

class TestRunHHBacktest:
    def test_returns_tuple_of_two(self):
        df = _make_hh_df(60)
        bt_df, metrics = run_halfhourly_backtest(df, holdout_days=30)
        assert isinstance(bt_df, pd.DataFrame)
        assert isinstance(metrics, dict)

    def test_metrics_contain_expected_keys(self):
        df = _make_hh_df(60)
        _, metrics = run_halfhourly_backtest(df, holdout_days=30)
        for key in ("mae", "peak_mae", "offpeak_mae", "r2"):
            assert key in metrics

    def test_mae_is_positive(self):
        df = _make_hh_df(60)
        _, metrics = run_halfhourly_backtest(df, holdout_days=30)
        assert metrics["mae"] >= 0

    def test_empty_df_returns_gracefully(self):
        bt_df, metrics = run_halfhourly_backtest(pd.DataFrame(), holdout_days=30)
        assert bt_df.empty
        assert metrics == {}


# ── price_lag1_slot alignment ──────────────────────────────────────────────────

class TestPriceLag1Slot:
    """Verify the 24-hour per-slot lag is computed correctly in the HH dataframe."""

    @pytest.mark.integration
    def test_price_lag1_slot_is_24h_offset(self):
        """Each slot's price_lag1_slot should equal the price from exactly 24h earlier."""
        from app.analysis import build_halfhourly_df
        df = build_halfhourly_df(date(2026, 1, 1), date(2026, 2, 28))
        if df.empty:
            pytest.skip("No HH data available in DB")

        # Build a price lookup by local datetime
        price_by_dt = df.set_index("datetime_local")["epex_price_p_kwh"].to_dict()

        mismatches = 0
        checked = 0
        for _, row in df.iterrows():
            lag_dt = row["datetime_local"] - timedelta(hours=24)
            expected = price_by_dt.get(lag_dt)
            if expected is None or np.isnan(row["price_lag1_slot"]):
                continue
            if not math.isclose(row["price_lag1_slot"], expected, rel_tol=1e-6):
                mismatches += 1
            checked += 1

        assert checked > 0, "No lag values were verifiable"
        assert mismatches == 0, f"{mismatches}/{checked} slots had wrong price_lag1_slot"


# ── estimate_solar_from_radiation ──────────────────────────────────────────────

class TestEstimateSolar:
    def test_returns_non_negative(self):
        rng = np.random.default_rng(0)
        hist = pd.DataFrame({
            "shortwave_radiation": rng.uniform(0, 400, 60),
            "solar_gw":            rng.uniform(0, 6, 60),
        })
        fc_rad = pd.Series(rng.uniform(0, 400, 10))
        result = estimate_solar_from_radiation(hist, fc_rad)
        assert (result >= 0).all()

    def test_returns_zeros_when_insufficient_history(self):
        hist = pd.DataFrame({
            "shortwave_radiation": [100.0] * 5,
            "solar_gw":            [1.0] * 5,
        })
        fc_rad = pd.Series([150.0, 200.0])
        result = estimate_solar_from_radiation(hist, fc_rad)
        assert (result == 0.0).all()

    def test_length_matches_input(self):
        rng = np.random.default_rng(0)
        hist = pd.DataFrame({
            "shortwave_radiation": rng.uniform(0, 400, 60),
            "solar_gw":            rng.uniform(0, 6, 60),
        })
        fc_rad = pd.Series(rng.uniform(0, 400, 14))
        result = estimate_solar_from_radiation(hist, fc_rad)
        assert len(result) == 14


# ── estimate_wind_gen_from_speed ───────────────────────────────────────────────

class TestEstimateWindGen:
    def _hist(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "wind_dogger_bank": rng.uniform(5, 80, n),
            "wind_hornsea":     rng.uniform(5, 80, n),
            "wind_gen_mw":      rng.uniform(2000, 14000, n),
        })

    def test_returns_non_negative(self):
        hist = self._hist()
        result = estimate_wind_gen_from_speed(hist, pd.Series([20.0, 40.0, 60.0]))
        assert (result >= 0).all()

    def test_returns_nan_when_no_wind_site_columns(self):
        # Only demand_mw present — no wind_* site columns, no wind_gen_mw
        hist = pd.DataFrame({"demand_mw": [30000.0] * 60, "wind_gen_mw": [5000.0] * 60})
        # Strip wind_gen_mw so there are no wind_* columns at all
        hist = hist[["demand_mw"]]
        result = estimate_wind_gen_from_speed(hist, pd.Series([30.0, 50.0]))
        assert result.isna().all()

    def test_length_matches_input(self):
        hist = self._hist()
        result = estimate_wind_gen_from_speed(hist, pd.Series([10.0] * 7))
        assert len(result) == 7

    def test_polynomial_captures_nonlinearity(self):
        """Higher wind speed should give higher (or at least different) generation than linear."""
        hist = self._hist(n=200)
        low  = estimate_wind_gen_from_speed(hist, pd.Series([10.0]))
        high = estimate_wind_gen_from_speed(hist, pd.Series([60.0]))
        # High wind should produce more than low wind
        assert high.iloc[0] > low.iloc[0]


# ── build_demand_profile ───────────────────────────────────────────────────────

class TestBuildDemandProfile:
    def test_returns_dict_keyed_by_dow_hour(self):
        datetimes = [datetime(2025, 1, 1, tzinfo=LOCAL_TZ) + timedelta(minutes=30 * i)
                     for i in range(48 * 14)]
        df = pd.DataFrame({
            "datetime_local": datetimes,
            "demand_mw":      np.random.default_rng(0).uniform(25000, 40000, len(datetimes)),
        })
        profile = build_demand_profile(df)
        assert isinstance(profile, dict)
        assert len(profile) > 0
        # Keys should be (dow, hour) tuples
        for key in list(profile.keys())[:3]:
            assert len(key) == 2

    def test_returns_empty_when_no_demand_column(self):
        df = pd.DataFrame({"datetime_local": [datetime(2025, 1, 1, tzinfo=LOCAL_TZ)]})
        assert build_demand_profile(df) == {}

    def test_values_are_positive(self):
        datetimes = [datetime(2025, 1, 1, tzinfo=LOCAL_TZ) + timedelta(hours=i)
                     for i in range(168)]
        df = pd.DataFrame({
            "datetime_local": datetimes,
            "demand_mw":      np.full(168, 30000.0),
        })
        profile = build_demand_profile(df)
        assert all(v > 0 for v in profile.values())
