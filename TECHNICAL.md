# UK Electricity Price Analysis — Technical Reference

Architecture, model internals, data sources, and schema details.

---

## Project structure

```text
forecast-weekly/
├── main.py               # CLI entry point (update / analyse / status)
├── requirements.txt
├── Makefile
├── Dockerfile
├── app/
│   ├── config.py         # Constants: sites, paths, network charges, tariff params
│   ├── db.py             # SQLite schema + all query functions
│   ├── analysis.py       # Re-export shim (features / models / backtest)
│   ├── features.py       # Feature dictionaries, data loading, feature engineering
│   ├── models.py         # Ridge + LightGBM fitting, prediction, ensemble blending
│   ├── backtest.py       # Hold-out tests, walk-forward CV, lead-time analysis
│   ├── midprice.py       # Elexon BMRS APXMIDP EPEX price client
│   ├── weather.py        # Open-Meteo historical + forecast client
│   ├── pvlive.py         # Sheffield Solar PV_Live API client
│   ├── demand.py         # Elexon BMRS INDO demand client
│   ├── supply.py         # Elexon BMRS FUELHH generation mix client
│   ├── gas.py            # Yahoo Finance commodity + GIE gas storage + EIA oil client
│   ├── entsoe.py         # ENTSO-E scheduled exchanges + generation unavailability
│   ├── sysprice.py       # Elexon BMRS system prices (balancing mechanism)
│   ├── storage.py        # GIE AGSI+ gas storage client
│   ├── eia.py            # EIA weekly crude oil inventory client
│   ├── octopus.py        # Octopus Agile tariff prices (comparison only)
│   ├── summary.py        # LLM week-ahead narrative (OpenAI)
│   ├── charts.py         # Matplotlib static PNG generation
│   └── dashboard.py      # Plotly HTML dashboard generation
└── infra/
    ├── app.py            # CDK app entry point
    ├── stack.py          # Full AWS stack definition
    └── setup-github-oidc.sh  # One-time OIDC + IAM role setup
```

---

## Deployed infrastructure

### Overview

The pipeline runs entirely serverlessly on AWS. There is no always-on server.

```
GitHub push → GitHub Actions → ECR (new image)
                                     │
EventBridge (13:00 UTC daily) ───→ ECS Fargate task
                                     │
                              EFS (energy.db persists)
                                     │
                              S3 (index.html + charts)
                                     │
                            CloudFront (CDN)
                                     │
                    https://d16khkgn2figlo.cloudfront.net
```

### Resources

| Resource | Name / ID | Purpose |
|---|---|---|
| AWS Account | `711695043600` (forecast-weekly) | Dedicated isolated account, org management: `627266360979` |
| ECS Cluster | `EnergyAnalysis-ClusterEB0386A7-zz6UZzcErUMW` | Runs the Fargate task |
| ECS Task Definition | `EnergyAnalysisTaskDefE9704C45` | 1 vCPU, 4 GB RAM |
| ECR Repository | `cdk-hnb659fds-container-assets-711695043600-eu-west-2` | Docker image storage |
| EFS Filesystem | `fs-0af287c838d2820ff` | Persistent SQLite DB at `/data/energy.db` |
| S3 Bucket | `energyanalysis-dashboard…` | Static dashboard HTML + chart PNGs |
| CloudFront | `d16khkgn2figlo.cloudfront.net` | HTTPS CDN in front of S3 |
| EventBridge Rule | `EnergyAnalysis-DailyRunDEF7747D-lt8qiUZido6W` | Triggers Fargate at 13:00 UTC daily |
| Secrets Manager | `ApiKeys3BB3983D-DBoTRntuXRPa` | API keys injected as env vars at runtime |
| IAM User | `github-actions-forecast-weekly` | GitHub Actions deploy credentials |

### How a daily run works

1. EventBridge fires at 13:00 UTC (after the EPEX D+1 auction clears ~12:00 CET)
2. Fargate pulls the latest Docker image from ECR and starts a task
3. The EFS volume is mounted at `/data` — `energy.db` is already there from the previous run
4. `main.py` runs: fetches only the delta since the last run (incremental), trains models, generates the dashboard
5. `index.html` and chart PNGs are uploaded to S3; CloudFront cache is invalidated
6. The container exits; EFS retains the updated `energy.db` for the next run

### Database persistence

`energy.db` lives on EFS and persists between runs. This means:
- Historical predictions accumulate and are used for the Predicted vs Actual chart and lead-time accuracy analysis
- The weather forecast archive grows over time, enabling lead-time interval scaling
- Only the delta since the last run is fetched from APIs (typically one day of data)

All data in the DB is re-fetchable from public APIs except for two tables which can only be captured in real time:
- `daily_predictions` / `halfhourly_predictions` — what the model predicted on each past run
- `weather_forecast_archive` / `wind_site_forecast_archive` — NWP forecasts as-of each run date

### API keys

Keys are stored in AWS Secrets Manager (`EnergyAnalysis/ApiKeys`) and injected as environment variables into the Fargate container at runtime. To update a key:

```bash
aws secretsmanager put-secret-value \
  --secret-id ApiKeys3BB3983D-DBoTRntuXRPa \
  --region eu-west-2 \
  --secret-string '{"GIE_API_KEY":"...","EIA_API_KEY":"...","ENTSOE_API_KEY":"...","OPENAI_API_KEY":"..."}'
```

For local development, keys go in `.env` (gitignored).

### Deploying changes

Push to `main`. The GitHub Actions workflow (`.github/workflows/deploy.yml`) will:
1. Build and push a new Docker image to ECR (tagged with the commit SHA)
2. Register a new ECS task definition revision pointing to the new image
3. Update the EventBridge target to use the new revision

The change takes effect on the next scheduled run at 13:00 UTC.

### To redeploy the CDK stack (infrastructure changes)

```bash
cd infra
pip install -r requirements.txt  # if not already installed
cdk deploy
```

CDK changes (new resources, schedule tweaks, etc.) must be deployed separately — the GitHub Actions workflow only updates the application container, not the infrastructure.

### To trigger a run manually

Ensure your CLI is targeting the `forecast-weekly` account (`711695043600`) before running. You can assume the org access role or use a named profile.

```bash
aws ecs run-task \
  --cluster EnergyAnalysis-ClusterEB0386A7-zz6UZzcErUMW \
  --task-definition EnergyAnalysisTaskDefE9704C45 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0c6c3a2293c8d8710],securityGroups=[sg-0b8fe0f030d1de0f7],assignPublicIp=ENABLED}" \
  --region eu-west-2
```

---

## Data sources and APIs

### Elexon BMRS — EPEX Day-Ahead Prices

- **Endpoint:** `https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/market-index-data-prices`
- **Dataset:** APXMIDP — Market Index Data (EPEX SPOT GB day-ahead clearing price)
- **Resolution:** Half-hourly (48 slots/day), £/MWh converted to p/kWh (×0.1)
- **Fields stored:** `datetime_utc`, `price_gbp_mwh`, `volume_mwh`

### Open-Meteo (weather)

- **Historical:** `https://archive-api.open-meteo.com/v1/archive` — ~2-day lag
- **Forecast:** `https://api.open-meteo.com/v1/forecast` — up to 16 days
- **UK weather sites:** 6 UK cities averaged (Edinburgh, Newcastle, Manchester, Birmingham, London, Cardiff)
- **Variables:** `temperature_2m`, `shortwave_radiation`, `precipitation` per site
- **Wind sites:** 100m hub height (`wind_speed_100m`) at 6 offshore/onshore wind farm locations

### Sheffield Solar PV_Live

- **Endpoint:** `https://api.pvlive.uk/pvlive/api/v4/gsp/0` (GB national, GSP 0)
- **Resolution:** Half-hourly, UTC period-**end** timestamps
- **Timestamp note:** PV_Live uses period-end convention ("00:30" = 00:00-00:30 slot). A +30 min shift is applied when joining to EPEX data which uses period-start.

### Elexon BMRS — GB Demand

- **Endpoint:** `https://data.elexon.co.uk/bmrs/api/v1/demand/outturn`
- **Dataset:** INDO (Initial National Demand Outturn)
- **Resolution:** Half-hourly, UTC period-start timestamps
- **Max range per request:** 28 days (chunk size = 14 days)

### Elexon BMRS — Generation Mix

- **Endpoint:** `https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary`
- **Dataset:** FUELHH — half-hourly generation by fuel type
- **Fields:** wind, gas, nuclear, pumped storage, hydro, net imports

### Yahoo Finance (commodity prices)

- **Library:** `yfinance`
- **Symbols:** `BZ=F` (Brent crude, USD/bbl), `TTF=F` (TTF gas, EUR/MWh), `GBPUSD=X`, `DX-Y.NYB` (USD index)
- **Resolution:** Daily closing prices; forward-filled over weekends/holidays (limit=5 days)
- **Smoothing:** 7-day rolling average applied at feature-build time

### GIE AGSI+ (gas storage)

- **Library:** `gie-py`
- **Data:** EU and GB gas storage fill levels (%), working gas volume (TWh)
- **API key required:** `GIE_API_KEY` in `.env`

### EIA (oil inventory)

- **Endpoint:** `https://api.eia.gov/v2/petroleum/stoc/wstk/data/`
- **Data:** US weekly crude oil inventory (million barrels)
- **API key required:** `EIA_API_KEY` in `.env`

### Octopus Energy API (comparison only)

- **Tariff:** AGILE-24-10-01, Region N (Southern Scotland)
- **Used for:** Customer simulation comparison charts — NOT used for model training

---

## Database schema

All data is stored in `energy.db` (SQLite, WAL mode).

| Table | Key | Description |
| --- | --- | --- |
| `market_index_halfhourly` | `datetime_utc` | Half-hourly EPEX GB day-ahead prices |
| `prices` | `datetime` (UTC) | Half-hourly Octopus Agile prices (comparison only) |
| `weather_uk_sites` | `(datetime, site_id)` | Hourly weather per UK city |
| `weather_wind_sites` | `(datetime, site_id)` | Hourly 100m wind speed per wind farm site |
| `solar_generation` | `datetime_gmt` | Half-hourly GB solar generation (MW) |
| `demand_halfhourly` | `datetime_utc` | Half-hourly GB demand (MW) |
| `generation_halfhourly` | `datetime_utc` | Half-hourly GB generation by fuel type |
| `commodity_prices` | `date` | Daily Brent, TTF gas, GBP/USD, USD index |
| `gas_storage` | `date` | Daily EU/GB gas storage levels |
| `oil_inventory` | `date` | Weekly US crude oil inventory |
| `daily_predictions` | `(predicted_on, date)` | Stored daily forecasts for verification |
| `weather_forecast_archive` | `(fetch_date, target_date)` | Archived weather forecasts for lead-time backtest |
| `wind_site_forecast_archive` | `(fetch_date, target_date, site_id)` | Archived wind forecasts |
| `fetch_log` | `id` | Audit log of all API fetches |

---

## Network charge constants

Configured in `app/config.py` for SPD (Central Scotland) DNO area:

| Charge | Rate | Source |
| --- | --- | --- |
| DUoS Red (16:00-19:00 Mon-Fri) | 13.091 p/kWh | SPD LC14 Statement 2026 |
| DUoS Amber (07:00-16:00, 19:00-23:00 Mon-Fri) | 1.423 p/kWh | SPD LC14 Statement 2026 |
| DUoS Green (23:00-07:00, all weekends) | 0.036 p/kWh | SPD LC14 Statement 2026 |
| TNUoS | 0.40 p/kWh | National Grid ESO annual statement |
| BSUoS buffer | 0.35 p/kWh | ~75th percentile of recent actuals |
| Policy levies (RO/CfD/CM/FIT/WHD/ECO) | 3.3 p/kWh | Ofgem price cap methodology |
| Supplier operating costs | 1.5 p/kWh | Ofgem price cap methodology |
| Standing charge | 61 p/day | Ofgem Q1 2026 price cap |

---

## Model architecture

See [MODEL.md](MODEL.md) for full model documentation including features, ensemble blending, and walk-forward cross-validation.

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

| Limitation | Impact |
|---|---|
| **No gas price forecast** | TTF/Brent rolling averages held constant across forecast horizon |
| **Geopolitical shocks** | Step-changes in price regime cannot be anticipated by weather-based models |
| **Interconnector flows** | Not reliably forecastable without published day-ahead schedules |
| **Must-run gas constraints** | Gas may remain online for grid stability even when economically unnecessary |
| **Price caps / interventions** | Regulatory interventions not modelled |
| **Weather forecast degradation** | D+5 to D+7 weather forecasts significantly less accurate than D+1 |
