# UK Electricity Price Analysis — Technical Reference

Architecture, model internals, data sources, and schema details.

---

## Project structure

```text
energy_analysis/
├── main.py               # CLI entry point (update / analyse / status)
├── requirements.txt
├── Makefile
├── energy.db             # SQLite database (auto-created on first run)
├── dashboard.html        # Generated interactive dashboard
├── charts/               # Generated static PNG charts
└── app/
    ├── config.py         # Constants: region, sites, paths, tariff coefficients
    ├── db.py             # SQLite schema + all query functions
    ├── octopus.py        # Octopus Agile API client
    ├── weather.py        # Open-Meteo historical + forecast client
    ├── pvlive.py         # Sheffield Solar PV_Live API client
    ├── demand.py         # Elexon BMRS INDO demand client
    ├── gas.py            # Yahoo Finance commodity price client
    ├── analysis.py       # Correlations, regression models, backtest, prediction
    ├── charts.py         # Matplotlib static PNG generation
    └── dashboard.py      # Plotly HTML dashboard generation
```

---

## Data sources and APIs

### Octopus Energy API

- **Endpoint:** `https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-N/standard-unit-rates/`
- **Tariff:** AGILE-24-10-01, Region N (Southern Scotland)
- **Resolution:** Half-hourly (48 slots/day), published ~16:00 for the following day
- **Fields stored:** `price_inc_vat`, `price_ex_vat` (÷1.05), `wholesale_price`, `is_peak`

### Open-Meteo (weather)

- **Historical:** `https://archive-api.open-meteo.com/v1/archive` — ~2-day lag
- **Forecast:** `https://api.open-meteo.com/v1/forecast` — up to 16 days
- **Sites:** 6 UK cities averaged (Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff)
- **Variables:** `temperature_2m`, `shortwave_radiation`, `precipitation` per site
- **Wind:** Fetched at 100m hub height (`wind_speed_100m`) for each offshore/onshore wind farm site only

### Sheffield Solar PV_Live

- **Endpoint:** `https://api.pvlive.uk/pvlive/api/v4/gsp/0` (GB national, GSP 0)
- **Resolution:** Half-hourly (30-min slots), UTC period-**end** timestamps
- **Fields stored:** `generation_mw` — GB national solar generation estimate
- **Timestamp note:** PV_Live uses period-end convention ("00:30" = 00:00–00:30 slot). A +30 min shift is applied when joining to Octopus data which uses period-start.

### Elexon BMRS — GB Demand

- **Endpoint:** `https://data.elexon.co.uk/bmrs/api/v1/demand/outturn`
- **Dataset:** INDO (Initial National Demand Outturn) — includes embedded/distributed generation
- **Resolution:** Half-hourly, UTC period-start timestamps (aligns directly with Octopus)
- **Max range per request:** 28 days (chunk size set to 14 days)
- **Fields stored:** `demand_mw`

### Yahoo Finance (commodity prices)

- **Library:** `yfinance`
- **Symbols:** `BZ=F` (Brent crude, USD/bbl), `TTF=F` (TTF natural gas, EUR/MWh)
- **Resolution:** Daily closing prices; forward-filled over weekends/holidays
- **Smoothing:** 7-day rolling average applied at feature-build time

---

## Database schema

All data is stored in `energy.db` (SQLite, WAL mode).

| Table | Key | Description |
| --- | --- | --- |
| `prices` | `datetime` (UTC) | Half-hourly Agile prices |
| `weather_uk_sites` | `(datetime, site_id)` | Hourly weather per UK city |
| `weather_wind_sites` | `(datetime, site_id)` | Hourly 100m wind speed per wind farm site |
| `solar_generation` | `datetime_gmt` (UTC) | Half-hourly GB solar generation (MW) |
| `demand_halfhourly` | `datetime_utc` | Half-hourly GB demand (MW) |
| `commodity_prices` | `date` | Daily Brent crude + TTF gas |
| `daily_predictions` | `(predicted_on, date)` | Stored daily forecasts for verification |
| `fetch_log` | `id` | Audit log of all API fetches |

---

## Price model

Octopus Agile prices follow a published formula:

```text
price_inc_vat  = API value (p/kWh, includes 5% VAT)
price_ex_vat   = price_inc_vat ÷ 1.05
wholesale_est  = price_ex_vat ÷ D              (off-peak slots)
wholesale_est  = (price_ex_vat − P) ÷ D        (peak 16:00–19:00 slots)
```

Region N constants: **D = 2.1** (distribution multiplier), **P = 13 p/kWh** (peak adder).

The primary metric throughout is `price_ex_vat` — VAT removed but network charges (TNUoS/DUoS) retained, because network charges are mandatory and identical across suppliers.

---

## Regression models

Both models use `Ridge(alpha=1.0)` with `StandardScaler`. Ridge is chosen over plain OLS to handle collinearity between the wind farm site features.

### Daily model

Predicts daily average `price_ex_vat` from:

| Feature group | Features |
| --- | --- |
| Weather (UK avg) | `temperature_2m`, `precipitation` |
| Solar supply | `solar_gw` (GB actual, from PV_Live) |
| Demand | `demand_mw` (GB daily avg, from Elexon BMRS) |
| Wind interaction terms | `temp_x_wind` (Temp × UK avg wind), `wind_x_solar` (UK avg wind × solar_gw) |
| Commodity | `gas_ttf_roll7`, `brent_roll7` (7-day rolling averages) |
| Wind farm sites | One column per site (100m wind speed daily avg) |

`uk_avg_wind` (mean of all wind farm site columns) is used only in interaction terms, not as a standalone feature, to prevent multicollinearity.

### Half-hourly model

Adds time-of-day features to the daily feature set:

| Feature | Description |
| --- | --- |
| `is_peak` | Binary flag: 1 if 16:00–19:00 local time |
| `hour_sin`, `hour_cos` | Cyclic encoding of hour-of-day |
| `doy_sin`, `doy_cos` | Cyclic encoding of day-of-year (seasonality) |
| `demand_mw` | Per-slot demand (not daily avg) |
| `solar_gw` | Per-slot solar generation (not daily avg) |

---

## Forecast pipeline

1. Fetch 7-day hourly weather forecast from Open-Meteo for all 6 UK sites → average
2. Fetch 7-day hourly 100m wind forecast for each wind farm site
3. **Estimate `solar_gw`** for forecast days using a Ridge linear model fitted on historical (shortwave_radiation → solar_gw) pairs from the past 90+ days. PV_Live does not provide free forecasts.
4. **Estimate `demand_mw`** for forecast days using a historical profile: mean demand per (day-of-week, hour-of-day) from the past 12 months.
5. Commodity features: latest 7-day rolling averages held constant across forecast horizon.
6. Apply daily model → 7-day daily price predictions (stored in DB for later verification).
7. Apply half-hourly model → 7-day × 48 slot predictions (shown in dashboard forecast chart).

---

## Backtest methodology

The 30-day **hold-out backtest** works as follows:

- Model is trained on all data **except** the most recent 30 days
- Hold-out period is predicted using **actual historical weather** (not a forecast) — this removes forecast uncertainty and shows the best possible accuracy achievable with this model and these features
- Errors reported: MAE (mean absolute error, p/kWh), RMSE, MAPE

**MAE (Mean Absolute Error):** the average difference in pence between predicted and actual price, regardless of direction. MAE = 2.6p means predictions are off by ±2.6p on average.

The backtest gives a more honest measure of model quality than in-sample R², because it tests on data the model has never seen.

---

## Wind farm sites

Configured in `app/config.py` (`WIND_SITES`):

| Site | Location | Capacity |
| --- | --- | --- |
| `dogger_bank` | North Sea, ~50 km offshore | 3.6 GW (largest offshore wind farm) |
| `hornsea` | North Sea, ~120 km offshore | ~2.5 GW |
| `walney` | Irish Sea, off Cumbria | ~0.7 GW |
| `whitelee` | Eaglesham Moor, Scotland | 539 MW (largest onshore UK) |
| `clyde_wind` | South Lanarkshire | 522 MW |
| `pen_y_cymoedd` | Neath Port Talbot, Wales | 228 MW |

Wind speed at 100m hub height is fetched from Open-Meteo's historical archive and forecast.

---

## Known limitations

- **No gas price forecast.** Commodity rolling averages are held constant across the horizon. A large gas price move during the forecast window won't be captured.
- **Linear model.** Extreme price spikes from grid constraints, interconnector failures, or emergency events are unlikely to be well captured. A gradient boosting model would improve accuracy at the cost of interpretability.
- **Demand estimate for forecast.** Day-of-week + hour profile is a reasonable but simplified proxy — it won't capture demand driven by temperature (cold snap → higher demand than seasonal average). This is a natural improvement area.
- **Solar estimate for forecast.** PV_Live doesn't provide free forecasts. The radiation → solar_gw linear model performs well in normal conditions but may underperform during unusual cloud cover patterns.
- **Region N constants (D, P)** are hard-coded in `app/config.py`. Octopus periodically updates these; check the tariff page if results look wrong.
