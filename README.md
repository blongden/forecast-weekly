# UK Electricity Price Analysis

Fetches EPEX SPOT GB day-ahead wholesale prices, UK weather, GB generation mix, and commodity prices; fits Ridge + LightGBM ensemble models; and publishes an interactive 7-day forecast dashboard.

---

## Quick start

```bash
make install      # create venv and install dependencies
make all          # fetch data + run analysis + open dashboard
```

On first run this downloads ~12 months of historical data (takes 1-2 minutes). Subsequent runs only fetch the gap since the last update.

API keys required in `.env` (see `.env.example`):
- `GIE_API_KEY` — EU/GB gas storage levels (free: https://agsi.gie.eu/account)
- `EIA_API_KEY` — US crude oil inventories (free: https://www.eia.gov/opendata/register.php)

---

## What you get

Open `index.html` in any browser. It contains:

| Section | What it shows |
| --- | --- |
| **Stat cards** | 12-month mean price, today's off-peak forecast, cheapest/most expensive day, out-of-sample forecast accuracy (MAE) |
| **7-Day Forecast** | Predicted half-hourly EPEX wholesale prices with 80% prediction intervals (q10/q90), peak shading (16:00-19:00) |
| **Tariff Design** | 3-band and 4-band time-of-use retail tariff with wholesale + network charge breakdown |
| **Customer Simulation** | All-in annual bill estimates across 4 load-shifting scenarios, with Ofgem cap comparison |
| **Price Drivers** | Gas & oil prices, GB demand, solar/wind generation vs wholesale price |
| **12-Month History** | Daily EPEX wholesale price |
| **Model Accuracy** | 30-day hold-out backtest and walk-forward cross-validation results |
| **Model Detail** | Daily and HH model sections — LightGBM feature importance, Ridge coefficients, correlations, prediction intervals |

---

## How to run

```bash
make all        # fetch missing data + run analysis (default)
make update     # fetch missing data only
make analyse    # re-run analysis on existing data, regenerate dashboard
make status     # show date ranges stored in the database
```

Or directly with Python:

```bash
python3 main.py           # equivalent to make all
python3 main.py update
python3 main.py analyse
python3 main.py status
```

---

## Output files

| File | Description |
| --- | --- |
| `index.html` | Self-contained interactive HTML dashboard |
| `energy.db` | SQLite database (all raw and aggregated data) |
| `charts/` | Static PNG charts (time series, scatter, forecast) |

---

## Data sources

| Source | What it provides |
| --- | --- |
| Elexon BMRS APXMIDP | Half-hourly EPEX GB day-ahead wholesale prices (£/MWh) |
| Open-Meteo | Hourly weather (temp, solar, precip) — averaged across 6 UK cities |
| Open-Meteo | Hourly 100m wind speed at 6 strategic wind farm sites |
| Sheffield Solar PV_Live | Half-hourly GB national solar generation (MW) |
| Elexon BMRS INDO | Half-hourly GB national electricity demand (MW) |
| Elexon BMRS FUELHH | Half-hourly GB generation mix (wind, gas, nuclear, hydro, imports) |
| Yahoo Finance | Daily Brent crude, TTF gas, GBP/USD, USD index |
| GIE AGSI+ | Daily EU/GB gas storage fill levels (%) |
| EIA | Weekly US crude oil inventory |
| Octopus Energy API | Agile tariff prices for customer simulation comparison |

---

## Model architecture

**Ridge + LightGBM ensemble** with walk-forward CV-optimised blend weights.

- **Daily model**: predicts daily avg EPEX wholesale price (p/kWh) from 28 features
- **Half-hourly model**: predicts per-slot EPEX price from 35+ features with recursive day-by-day forecasting
- **Quantile models**: LightGBM q10/q90 for 80% prediction intervals
- Walk-forward CV: Ridge MAE ~1.17p, LightGBM ~1.07p, Ensemble ~1.05p

For full model details see [MODEL.md](MODEL.md). For technical reference see [TECHNICAL.md](TECHNICAL.md).

---

## Caveats

- **No gas price forecast.** The most recent 7-day rolling average of TTF/Brent is held constant across the forecast horizon.
- **Weather forecast error** adds uncertainty on top of the model's own limitations.
- **Price spikes** from grid constraints or emergency events are not predictable from this model.
- **Electricity only.** The customer simulation models electricity bills only, not gas.
