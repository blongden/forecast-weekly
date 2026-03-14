# Energy Inventory & Reserve Data Sources — Research

This document catalogues available data sources for gas/oil inventory levels, strategic reserves,
and maritime cargo tracking, with an assessment of their suitability for our UK electricity price
forecasting model.

---

## 1. European Gas Storage (GIE AGSI+) — RECOMMENDED

**Provider:** Gas Infrastructure Europe (GIE)
**URL:** https://agsi.gie.eu/
**Cost:** Free (registration required for API key)
**Update frequency:** Daily (19:30 CET, second update 23:00 CET)
**Historical data:** Several years, per facility/operator/country

**Why it matters:** Gas storage fill level is a major seasonal price driver for European energy markets.
Low storage entering winter triggers price anxiety and forward-buying; high storage suppresses prices.
The 2021-22 energy crisis was directly linked to low EU gas storage. TTF gas is the primary fuel-cost
driver of UK electricity prices, so EU storage % is a strong leading indicator.

**Data available:**
- Fill level (%) and working gas volume (TWh)
- Injection/withdrawal rates
- Working capacity
- Per facility, per operator, per country, and EU aggregate
- **GB is included** as a country code — gives UK-specific storage data

**Python access:** `pip install gie-py`
```python
from gie import GiePandasClient
client = GiePandasClient(api_key="YOUR_KEY")
df = client.query_gas_country("GB", start="2025-01-01", end="2025-12-31")
# Returns: gasInStorage, full (%), consumption, injection, withdrawal, etc.
```

**Integration plan:** New `app/storage.py` module. Store daily EU + GB fill % in a new `gas_storage`
table. Add `eu_gas_storage_pct` and `gb_gas_storage_pct` as model features (7-day rolling average).

---

## 2. UK Gas Storage & LNG Flows (National Gas)

**Provider:** National Gas Transmission (formerly National Grid Gas)
**URL:** https://data.nationalgas.com/
**Cost:** Free (open data under GSO licence)
**Update frequency:** Near real-time / daily
**API key:** Not required for basic access

**Why it matters:** Gives UK-specific granular storage data (Rough, Aldbrough, Hornsea, Hatfield Moor,
Hill Top, Holford) plus LNG terminal flow data (South Hook, Dragon LNG, Isle of Grain). UK is a net
gas importer — LNG arrivals directly affect domestic supply and short-term pricing.

**Data available:**
- Storage facility injection/withdrawal rates
- Linepack (gas in the NTS pipeline itself — a very short-term supply indicator)
- LNG terminal send-out rates
- Supply/demand forecasts

**API:** REST API at `https://data.nationalgas.com/apis/rest-apis`

**Note:** UK storage also appears in AGSI+ (GB country code) with daily granularity. The National Gas
portal offers near-real-time and more detailed facility-level data, but AGSI+ is simpler to integrate
and sufficient for our daily model.

---

## 3. US Crude Oil Inventories (EIA) — RECOMMENDED

**Provider:** US Energy Information Administration
**URL:** https://www.eia.gov/opendata/
**Cost:** Free (API key required — free registration)
**Update frequency:** Weekly (Wednesday release, covering week ending prior Friday)
**Historical data:** Decades

**Why it matters:** EIA weekly petroleum inventory reports are the most-watched oil data release globally.
Surprise builds/draws move Brent crude prices, which flow through to UK electricity costs via
gas-fired generation economics. While US-focused, Brent crude is a global benchmark priced off
the same supply/demand dynamics.

**Data available:**
- US commercial crude oil stocks (weekly, million barrels)
- US strategic petroleum reserve (SPR) levels
- Product stocks (gasoline, distillate, jet fuel)
- OECD country-level stocks (monthly, with lag)

**API access:** REST API v2
```
GET https://api.eia.gov/v2/petroleum/stoc/wstk/data/
    ?api_key=YOUR_KEY
    &frequency=weekly
    &data[0]=value
    &facets[product][]=EPC0    # crude oil
    &facets[process][]=SAE     # ending stocks
```

**Python:** Use `requests` directly (the `eiapy` package exists but is thin).

**Integration plan:** Add EIA fetching to `app/gas.py` (or new `app/eia.py`). Store weekly US crude
stocks in a new `oil_inventory` table. Forward-fill daily, add `us_crude_stocks_roll7` as a feature.
Weekly data means limited variance, but week-over-week change (`us_crude_stocks_delta`) may capture
surprise draws/builds.

---

## 4. EU Commercial Oil Stocks (Eurostat)

**Provider:** European Commission / Eurostat
**Dataset:** `nrg_stk_oilm` (monthly oil stock levels)
**URL:** https://ec.europa.eu/eurostat/databrowser/view/NRG_STK_OILM/
**Cost:** Free, no API key required
**Update frequency:** Monthly (2-3 month reporting lag)

**Why it matters:** EU oil stock levels indicate regional supply security. However, the 2-3 month lag
makes this less useful for a 7-day forecast model.

**Python access:** `pip install eurostat`
```python
import eurostat
df = eurostat.get_data_df('nrg_stk_oilm')
```

**Assessment:** Low priority for our model due to reporting lag. Useful for long-term trend analysis
but not actionable for short-term price forecasting.

---

## 5. Global Oil Data (JODI)

**Provider:** Joint Organisations Data Initiative
**URL:** https://www.jodidata.org/oil/
**Cost:** Free
**Update frequency:** Monthly (~20th of each month)
**Historical data:** From January 2002, covering ~100 countries

**Data available:** Production, refinery throughput, imports, exports, closing stocks — by product type.

**Assessment:** Good for global context but monthly frequency and reporting lag make it unsuitable
for our short-term model. More useful for research and understanding structural supply trends.

---

## 6. Oil/Gas Tankers at Sea & Floating Storage

### Commercial Providers (Paid)

| Provider | What they offer | Estimated cost | API |
|----------|----------------|----------------|-----|
| **Kpler** (owns MarineTraffic) | Global cargo flows, floating storage, vessel tracking | ~$30,000+/yr | REST + Python SDK |
| **Vortexa** | Crude/products/LNG flows, tanker-by-tanker tracking from 2016 | ~$25,000+/yr | REST + Python SDK |
| **MarineTraffic** | AIS vessel positions, port calls, vessel events | Tiered pricing | REST API |
| **LSEG/Refinitiv** | Everything (pricing, storage, vessels, inventories) | ~£15,000+/yr | REST + Python SDK |
| **Bloomberg** | Everything | ~$24,000/yr | Terminal API |

### Free AIS Alternatives

| Provider | Cost | Limitation |
|----------|------|------------|
| **AISstream.io** | Free | WebSocket streaming of raw AIS positions. You must filter for tanker types, interpret draft readings, and geofence yourself. |
| **AISHub** | Free (reciprocal) | Requires you to contribute AIS data from your own receiver. |
| **Datalastic** | Freemium | Basic vessel positions and details. |

**Key limitation:** Free AIS sources give raw vessel positions. Deriving "floating storage volume"
or "laden crude tankers heading to UK" requires combining AIS data with vessel databases, cargo
manifests, draft readings, and port/terminal geofencing. This is the value-add that Kpler and Vortexa
sell. Building this from scratch is a substantial engineering project and is not recommended for our
use case.

---

## 7. LNG Cargo Tracking

### GIE ALSI+ (EU LNG terminal levels)

**Provider:** Gas Infrastructure Europe
**URL:** https://alsi.gie.eu/
**Cost:** Free (same API key as AGSI+)
**Update frequency:** Daily
**Coverage:** 10 EU member states (Belgium, Croatia, France, Greece, Italy, Lithuania, Netherlands,
Poland, Portugal, Spain). **UK is NOT included** (post-Brexit).

**Data available:** LNG inventory, send-out rates, storage capacity, regasification capacity per terminal.

**Python:** Same `gie-py` package as AGSI+.

### UK LNG Terminals

UK has 3 LNG import terminals:
- **South Hook** (Milford Haven) — 15.6 mtpa, largest in Europe
- **Dragon LNG** (Milford Haven) — 7.5 mtpa
- **Isle of Grain** (Kent) — 14.8 mtpa

Data available via the **National Gas portal** (https://data.nationalgas.com/) as part of NTS
supply data items. No dedicated UK ALSI equivalent.

### Commercial LNG Tracking

Kpler, Vortexa, and ICIS LNG Edge offer vessel-level LNG cargo tracking (origin, destination, ETA,
cargo size). All are enterprise-priced.

---

## Implementation Priority

| Priority | Source | Feature | Rationale |
|----------|--------|---------|-----------|
| **1 — NOW** | GIE AGSI+ | `eu_gas_storage_pct`, `gb_gas_storage_pct` | Daily, free, strong seasonal price driver, easy Python integration |
| **2 — NOW** | EIA API | `us_crude_stocks_mb`, `us_crude_stocks_delta` | Weekly, free, globally watched inventory data |
| **3 — LATER** | National Gas portal | UK storage detail + LNG terminal flows | Near-RT, free, but requires more API integration work |
| **4 — LATER** | GIE ALSI+ | EU LNG terminal send-out | Daily, free, same API — but UK not included |
| **5 — NOT PLANNED** | Kpler / Vortexa | Tanker tracking, floating storage | Enterprise pricing, not justified for this project |
| **6 — NOT PLANNED** | Eurostat / JODI | EU/global oil stocks | Monthly lag too long for 7-day forecast model |

---

*Last updated: 12 March 2026*
