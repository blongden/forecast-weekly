# Energy Price Model — How It Works

This document explains the data sources, model architecture, feature engineering, and prediction
pipeline for the UK EPEX wholesale electricity price forecasting system.

---

## 1. What We Are Predicting

**EPEX SPOT GB day-ahead wholesale prices** in p/kWh. The day-ahead auction clears daily
(~12:00 for next-day delivery), producing 48 half-hourly settlement prices per day. Prices
range from near-zero (or negative) in windy off-peak periods to 15p+/kWh during winter
evening peaks.

We model wholesale prices directly — no retail markup, VAT, or network charges are included
in the training target.

---

## 2. Data Sources

| Source | What we collect | Granularity | Notes |
|---|---|---|---|
| **Elexon BMRS APXMIDP** | Half-hourly EPEX GB day-ahead prices (£/MWh) | 30-min slots | Converted to p/kWh (x0.1) |
| **Open-Meteo** | Temperature, solar radiation, precipitation at 6 UK sites | Hourly | UK average across Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff |
| **Open-Meteo** | Wind speed at 6 specific wind farm sites (100m hub height) | Hourly | Dogger Bank, Hornsea, Walney, Whitelee, Clyde Wind, Pen y Cymoedd |
| **Sheffield Solar PV_Live** | GB actual solar photovoltaic generation | 30-min slots | Period-end timestamps; shifted +30 min to align with EPEX period-start |
| **Elexon BMRS INDO** | GB electricity demand (Initial National Demand Outturn) | 30-min slots | Stored for correlation analysis; not used as model feature |
| **Elexon BMRS FUELHH** | GB generation by fuel type: wind, gas, nuclear, pumped storage, hydro, net imports | 30-min slots | Wind generation used via speed->generation estimator; others stored for analysis |
| **Yahoo Finance** | TTF natural gas, Brent crude, GBP/USD, USD index | Daily | Smoothed with 7-day rolling average; forward-filled up to 5 days for weekends/holidays |
| **GIE AGSI+** | EU and GB gas storage fill levels (%) | Daily | Forward-filled up to 5 days |
| **EIA** | US crude oil inventory (million barrels) | Weekly | Forward-filled to daily; week-over-week delta computed |

All data is stored locally in an SQLite database and refreshed daily.

---

## 3. Model Architecture — Ridge + LightGBM Ensemble

Both the daily and half-hourly models use a **Ridge + LightGBM ensemble** with walk-forward
cross-validation to determine optimal blend weights.

### Why an ensemble?

- **Ridge regression** is linear, interpretable, and handles collinearity well (important with 28+ correlated features). It captures the main price drivers but misses nonlinear patterns.
- **LightGBM** (gradient-boosted decision trees) captures nonlinear relationships, feature interactions, and threshold effects automatically. It typically outperforms Ridge on MAE.
- **Blending** combines the stability of Ridge with the flexibility of LightGBM. Walk-forward CV finds the optimal weight.

### Blend weight determination

Walk-forward expanding-window cross-validation (4 folds, 120-day minimum training):
- For each fold: fit both models on training data, predict test set
- Test all blend weights from 0.0 (pure LightGBM) to 1.0 (pure Ridge) in 0.1 steps
- Select the weight with lowest mean MAE across folds

Typical results:
- Daily model: ~50% Ridge / 50% LightGBM (blend weight ~0.5)
- HH model: ~0% Ridge / 100% LightGBM (pure LightGBM — enough data for trees to dominate)

### Prediction intervals

LightGBM quantile regression models (q10 and q90) provide 80% prediction intervals:
- Fitted on the same features as the point-estimate LightGBM
- q10 = "prices unlikely to go below this" (10th percentile)
- q90 = "prices unlikely to go above this" (90th percentile)

### LightGBM hyperparameters

| Parameter | Daily | Half-hourly | Why |
|---|---|---|---|
| `n_estimators` | 300 | 400 | More data -> more capacity |
| `num_leaves` | 15 | 31 | Conservative for ~364 rows; more for ~17k rows |
| `learning_rate` | 0.05 | 0.05 | Standard |
| `min_child_samples` | 10 | 20 | Regularisation |
| `subsample` | 0.8 | 0.8 | Row subsampling |
| `colsample_bytree` | 0.8 | 0.8 | Feature subsampling |

---

## 4. The Daily Model

Predicts the **daily average EPEX wholesale price** (p/kWh).

### Features (28 total)

Only features **genuinely available at forecast time** are included. Non-forecastable
features (demand, gas generation, nuclear, pumped storage, hydro, imports) were removed to
eliminate training/serving mismatch.

| Feature | Source | Why it matters |
|---|---|---|
| `temperature_2m` | Open-Meteo hourly avg, UK mean | Cold weather -> higher heating demand -> higher price |
| `heating_dd` | Derived: max(0, 15.5 - temp) | Non-linear demand signal; flat above 15.5C, rising steeply below |
| `precipitation` | Open-Meteo hourly avg | Wet/windy weather proxy |
| `solar_gw` | Sheffield Solar PV_Live (estimated from radiation for forecast) | More solar -> less gas needed -> lower price |
| `wind_gen_mw` | Estimated from wind site speeds (polynomial model) | More wind -> displaces gas -> lower price |
| `is_bank_holiday` | Calendar (England + Scotland) | Demand profile resembles Sunday regardless of day |
| `is_weekend` | Calendar (Sat/Sun) | Systematically lower commercial/industrial demand on weekends |
| `epex_lag1_gbp_mwh` | Elexon BMRS MID, 1-day lag | Strong price autocorrelation |
| `gas_ttf_roll7` | Yahoo Finance, 7-day rolling avg | TTF is the European gas hub benchmark |
| `brent_roll7` | Yahoo Finance, 7-day rolling avg | Oil price; correlated with gas |
| `gbpusd_roll7` | Yahoo Finance, 7-day rolling avg | Currency effect on imported energy costs |
| `dxy_roll7` | Yahoo Finance, 7-day rolling avg | Broader dollar strength signal |
| `eu_gas_storage_pct` | GIE AGSI+ | Low storage = winter anxiety -> higher prices |
| `gb_gas_storage_pct` | GIE AGSI+ | GB-specific storage signal |
| `us_crude_stocks_delta` | EIA weekly | Surprise builds/draws move global oil prices |
| `epex_lag7_gbp_mwh` | Derived from EPEX price | Same day last week |
| `epex_roll7_std` | Derived | Recent price volatility |
| `epex_roll7_min`, `epex_roll7_max` | Derived | Recent price range |
| `epex_momentum_7` | Derived: lag1 - lag7 | Rising or falling price trend |
| `wind_dogger_bank` ... `wind_pen_y_cymoedd` | Open-Meteo 100m | Site-specific wind speed at 6 major wind farms |
| `temp_x_wind` | Interaction: temperature x avg wind speed | Cold + calm = double squeeze |
| `wind_x_solar` | Interaction: wind x solar | Both renewables suppressed = gas-only market |

### Wind generation estimation

Wind generation is estimated from average wind speed using a **polynomial Ridge model**
(wind_speed, wind_speed^2, wind_speed^3) to capture the nonlinear turbine power curve:
cubic relationship below rated speed, plateau at rated, and cut-out at high speeds.

---

## 5. The Half-Hourly Model

Predicts the **EPEX wholesale price for each individual 30-minute slot** (p/kWh).

### Why a separate model?

The daily model captures the average price level. The HH model additionally captures
**intraday shape** — the sharp peak between 16:00-19:00, the overnight trough, and how
wind and solar shift prices within a day.

### Additional features (beyond daily)

| Feature | Source | Why it matters |
|---|---|---|
| `net_residual_mw` | Derived: demand - wind - solar*1000 - nuclear - hydro | How much gas is needed on the margin — the key price driver |
| `price_lag1_slot` | EPEX BMRS MID, yesterday's same slot | Strongest feature: yesterday's 17:00 price anchors today's 17:00 |
| `is_peak` | Derived: 1 if 16:00-18:59 | Baseline peak uplift |
| `hour_sin`, `hour_cos` | Derived from local time | Continuous encoding of time-of-day |
| `doy_sin`, `doy_cos` | Derived from day-of-year | Continuous encoding of seasonality |

### Key design decisions

**Recursive forecasting for D+2+:** Day 1 uses actual yesterday's prices as `price_lag1_slot`.
Day 2 uses Day 1's *predicted* prices as the lag. Day 3 uses Day 2's predictions, etc.
This prevents stale lags degrading multi-day forecast accuracy.

**`net_residual_mw` (marginal demand signal):** Demand minus renewables and baseload
(wind, solar, nuclear, hydro) tells the model how much gas-fired generation is needed.

**Non-forecastable features excluded:** `demand_mw`, `gas_gen_mw`, `nuclear_mw`,
`pumped_storage_mw`, `hydro_mw`, and `imports_mw` are excluded from both models
to prevent training/serving mismatch.

---

## 6. Forecast Pipeline

### Daily 7-day forecast

1. Fetch 7-day hourly weather forecast from Open-Meteo
2. Fetch 7-day hourly wind speed forecasts for each wind farm site
3. Hold commodity/inventory features constant at most recent known values
4. Estimate wind generation (polynomial), solar (Ridge on radiation)
5. Apply Ridge + LightGBM ensemble -> 7 daily predictions with q10/q90 intervals

### Half-hourly 7-day forecast

1. Same weather and wind site forecasts
2. Per slot: estimate solar, wind, net residual demand
3. Day 1: use actual yesterday's EPEX price at each slot as `price_lag1_slot`
4. Day 2-7: use previous day's predicted prices (recursive)
5. Apply Ridge + LightGBM ensemble day-by-day -> 336 per-slot predictions

---

## 7. Accuracy Evaluation

### 30-day hold-out backtest

- Model re-trained on all data except most recent 30 days
- Tested using **actual weather** (not a forecast) — best-case accuracy ceiling
- Reports MAE, RMSE, MAPE, R^2

### Walk-forward cross-validation

- Expanding-window CV with 5 folds and 120-day minimum training
- Tests each model independently (Ridge, LightGBM) and the blended ensemble
- Determines optimal blend weight from fold-level results
- Typical results: Ridge MAE ~1.17p, LightGBM ~1.07p, Ensemble ~1.05p

### Archived lead-time backtest

- Uses stored weather forecasts to measure real-world accuracy degradation D+1 through D+7
- Each forecast is made with the weather forecast available at that lead time (not actuals)
- Shows true operational accuracy including weather forecast error

### Half-hourly backtest

- Same hold-out methodology but on per-slot data
- Reports overall MAE plus peak vs off-peak split
- Typical: overall MAE ~1.27p, peak ~1.42p, off-peak ~1.25p

---

## 8. Customer Simulation

Models four load-shifting scenarios against Octopus Agile for comparison:
1. **No shifting** — price-inelastic baseline
2. **Light shifting** — dishwasher/washing to off-peak
3. **Heavy shifting** — smart appliances
4. **EV household** — base shift + 4 kWh/day overnight EV charging

All-in bill includes: wholesale, network charges, policy levies (3.3p/kWh),
supplier opex (1.5p/kWh), standing charge (61p/day), and 5% VAT.

Uses Elexon PC1 seasonal load profiles (winter/summer, weekday/weekend) with
DUoS red-band restricted to Mon-Fri.

---

## 9. Known Limitations

| Limitation | Impact |
|---|---|
| **No gas price forecast** | TTF/Brent rolling averages held constant across horizon |
| **Geopolitical shocks** | Step-changes in price regime not anticipated |
| **Interconnector flows** | Not reliably forecastable without published schedules |
| **Must-run gas constraints** | Gas may remain online for grid stability |
| **Price caps / interventions** | Regulatory interventions not modelled |
| **Weather forecast degradation** | D+5 to D+7 weather significantly less accurate than D+1 |
| **Electricity only** | Customer simulation does not model gas bills |
