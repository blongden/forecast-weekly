# UK Electricity Price Analysis

Fetches Octopus Agile prices, UK weather, GB solar generation, GB demand, and gas/oil prices; fits predictive models; and publishes an interactive 7-day forecast dashboard.

---

## Quick start

```bash
pip install -r requirements.txt
make all          # fetch data + run analysis + open dashboard
```

On first run this downloads ~12 months of historical data (takes 1–2 minutes). Subsequent runs only fetch the gap since the last update.

---

## What you get

Open `dashboard.html` in any browser. It contains:

| Section | What it shows |
| --- | --- |
| **Stat cards** | 12-month mean price, today's off-peak forecast, cheapest/most expensive day, out-of-sample forecast accuracy (MAE) |
| **7-Day Forecast** | Predicted half-hourly Agile prices, slot by slot, with peak shading (16:00–19:00) |
| **Price Drivers** | Gas & oil prices vs Agile tariff; GB demand & solar generation vs price |
| **12-Month History** | Daily Agile price with estimated wholesale cost |
| **Model Accuracy** | 30-day hold-out test — how well the model predicts prices it was never trained on |
| **Model Detail** | Correlations, regression coefficients, scatter plots (for the technically curious) |

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
| `dashboard.html` | Self-contained interactive HTML dashboard |
| `energy.db` | SQLite database (all raw and aggregated data) |
| `charts/` | Static PNG charts (time series, scatter, forecast) |

---

## Data sources

| Source | What it provides |
| --- | --- |
| Octopus Energy API | Half-hourly Agile tariff prices (p/kWh) |
| Open-Meteo | Hourly weather — averaged across 6 UK cities |
| Sheffield Solar PV_Live | Half-hourly GB national solar generation (MW) |
| Elexon BMRS | Half-hourly GB national electricity demand (MW) |
| Yahoo Finance | Daily Brent crude (USD/bbl) and TTF gas (EUR/MWh) |

**Tariff:** AGILE-24-10-01, Region N (Southern Scotland)

---

## Forecast accuracy

The dashboard reports **MAE** (mean absolute error) — the average error in pence per kWh across a 30-day hold-out test. For example, MAE = 2.6p means predictions are off by about 2.6p on average.

> This uses actual historical weather as a proxy for a perfect forecast, so real-world accuracy (which must predict weather too) will be somewhat higher. See [TECHNICAL.md](TECHNICAL.md) for details.

---

## Caveats

- **No gas price forecast.** The most recent 7-day rolling average of TTF/Brent is held constant across the forecast horizon. A large gas price move during the forecast window won't be captured.
- **Weather forecast error** adds uncertainty on top of the model's own limitations.
- **Price spikes** from grid constraints or emergency events are not predictable from this model.

---

For model internals, data schemas, and API details see [TECHNICAL.md](TECHNICAL.md).
