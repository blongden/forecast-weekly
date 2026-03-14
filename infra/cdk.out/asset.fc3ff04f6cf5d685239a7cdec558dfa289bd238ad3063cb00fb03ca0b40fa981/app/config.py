"""
Central configuration for the energy analysis app.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Public mode: hides client-specific tariff design and model internals.
# Set PUBLIC_MODE=1 in .env to enable.
PUBLIC_MODE: bool = os.environ.get("PUBLIC_MODE", "").strip() in ("1", "true", "yes")

# ── Paths ──────────────────────────────────────────────────────────────────────
# Override with env vars so the container can point at a mounted volume (/data).
BASE_DIR   = Path(__file__).parent.parent
DB_PATH    = Path(os.environ.get("DB_PATH",    BASE_DIR / "energy.db"))
CHARTS_DIR = Path(os.environ.get("CHARTS_DIR", BASE_DIR / "charts"))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Location ──────────────────────────────────────────────────────────────────
TIMEZONE = "Europe/London"

# ── UK representative weather sites (temperature, solar, precipitation) ────────
# Six cities covering Scotland, NE/NW/Midlands/SE England, and Wales.
# Temperature, solar radiation, and precipitation are averaged across these sites
# to give a demand-weighted UK signal rather than a single-location proxy.
UK_WEATHER_SITES = {
    "edinburgh":  {"lat": 55.95, "lon": -3.19},
    "newcastle":  {"lat": 54.97, "lon": -1.61},
    "manchester": {"lat": 53.48, "lon": -2.24},
    "birmingham": {"lat": 52.48, "lon": -1.90},
    "london":     {"lat": 51.50, "lon": -0.12},
    "cardiff":    {"lat": 51.48, "lon": -3.18},
}

# ── Offshore wind sites ────────────────────────────────────────────────────────
# Wind speed at these locations proxies for actual UK offshore generation output.
# hub_height=100m matches typical offshore turbine hub heights.
# Temperature/precipitation use UK_WEATHER_SITES above; wind is wind-farm-only.
# Sites are a mix of large offshore and large onshore farms to proxy UK total wind output.
# ── Network charge rate tables ─────────────────────────────────────────────────
# Rates for Central Scotland — SP Distribution (SPD) DNO area, LV half-hourly
# metered customers. Must be updated annually from the official published statements:
#
#   DUoS:  SP Distribution LC14 Charging Statement (published ~March, effective April)
#          https://www.scottishpower.com/about_us/our_businesses/sp_energy_networks/
#                  regulatory_documents/connections_use_of_system_and_metering_services
#          Look for "SPD – Schedule of Charges and Other Tables.xlsx"
#   TNUoS: National Grid ESO Annual Charging Statement
#          https://www.nationalgrideso.com/industry-information/charging/transmission-network-use-system-charges
#   BSUoS: Elexon — volatile, settled post-delivery; use rolling average of recent actuals
#          https://www.elexon.co.uk/operations-settlement/bsc-central-services/bsuos/
#
# DUoS (p/kWh) — keyed by time band as used in Agile / HH metered tariffs
# Red   = 16:00–19:00 Mon–Fri (peak demand period)
# Amber = 07:00–16:00 and 19:00–23:00 Mon–Fri
# Green = 23:00–07:00 and all day weekends
#
# Source: SPD LC14 Statement 2026, "Domestic Aggregated or CT with Residual"
# (Profile Classes 0, 1, 2 — standard domestic HH metered customers)
DUOS_RATES: dict[str, float] = {
    "red":   13.091,  # p/kWh
    "amber":  1.423,  # p/kWh
    "green":  0.036,  # p/kWh
}

# TNUoS (p/kWh) — simplified demand residual rate for LV commercial customers
# Actual rate depends on customer size and triad exposure; this is a flat-rate approximation.
TNUOS_RATE: float = 0.40  # p/kWh — update from National Grid ESO annual statement

# BSUoS buffer (p/kWh) — stochastic balancing charge; sized at ~75th percentile of recent actuals
# Real BSUoS is settled post-delivery and varies £0–10/MWh (0–1p/kWh); 0.35p covers most outcomes.
BSUOS_BUFFER: float = 0.35  # p/kWh

# ── Policy / environmental levies (p/kWh) ─────────────────────────────────────
# Combined pass-through costs that all suppliers must collect:
#   - Renewables Obligation (RO): ~1.5p/kWh
#   - Contracts for Difference (CfD): ~0.8p/kWh
#   - Capacity Market (CM): ~0.3p/kWh
#   - Feed-in Tariff (FIT): ~0.3p/kWh
#   - Warm Home Discount / ECO: ~0.4p/kWh
# Source: Ofgem typical domestic supply cost breakdown (updated each cap quarter)
POLICY_LEVY_P_KWH: float = 3.3  # p/kWh ex-VAT — combined environmental/social levies

# ── Supplier operating costs (p/kWh) ──────────────────────────────────────────
# Metering, billing, customer service, IT, bad debt — Ofgem allowance in price cap.
# Source: Ofgem price cap methodology — "operating costs" component
SUPPLIER_OPEX_P_KWH: float = 1.5  # p/kWh ex-VAT

# Supplier margin — two modes:
#   "flat"       : fixed p/kWh adder on every band (price = EPEX + network + margin)
#   "multiplier" : wholesale is scaled up, like Octopus Agile (price = EPEX × mult + network)
#                  The margin is implicit: margin_p_kwh = EPEX × (mult − 1)
SUPPLIER_MARGIN: float = 12.5        # p/kWh — used when SUPPLIER_MARGIN_MODE = "flat"
SUPPLIER_MULTIPLIER: float = 2.0     # wholesale multiplier — used when mode = "multiplier"
SUPPLIER_MARGIN_MODE: str = "multiplier"   # "flat" or "multiplier"

# ── Octopus Agile tariff — comparison data only ───────────────────────────────
# Actual Agile prices are fetched and stored in the prices table for use in the
# customer simulation comparison. NOT used for model training (EPEX data only).
# Region N = Southern Scotland (Scotland Central Belt)
OCTOPUS_BASE:    str   = "https://api.octopus.energy/v1"
OCTOPUS_PRODUCT: str   = "AGILE-24-10-01"
OCTOPUS_TARIFF:  str   = "E-1R-AGILE-24-10-01-N"
AGILE_D:         float = 2.1    # Wholesale multiplier in Agile pricing formula
AGILE_P:         float = 13.0   # Peak adder p/kWh ex-VAT (Mon–Fri 16:00–19:00)
AGILE_VAT:       float = 1.05   # Standard VAT rate

# ── Ofgem price cap reference ─────────────────────────────────────────────────
# Standard variable tariff cap rates — update each quarter from:
# https://www.ofgem.gov.uk/check-if-energy-price-cap-affects-you
# These are ALL-IN unit rates (inc wholesale, network, levies, opex, 5% VAT).
# Q1 2026 (Jan–Mar 2026)
OFGEM_UNIT_RATE_P_KWH: float = 24.50   # p/kWh inc VAT
OFGEM_STANDING_CHARGE_P_DAY: float = 61.0  # p/day inc VAT
OFGEM_CAP_QUARTER: str = "Q1 2026"

WIND_SITES = {
    # Offshore — North Sea
    "dogger_bank": {"lat": 54.5,  "lon":  2.0,  "label": "Dogger Bank (North Sea, 3.6 GW)"},
    "hornsea":     {"lat": 53.9,  "lon":  1.1,  "label": "Hornsea (North Sea, ~2.5 GW)"},
    # Offshore — Irish Sea / west coast
    "walney":      {"lat": 54.1,  "lon": -3.6,  "label": "Walney (Irish Sea, ~0.7 GW)"},
    # Large onshore — Scotland
    "whitelee":    {"lat": 55.7,  "lon": -4.2,  "label": "Whitelee (Eaglesham Moor, 539 MW)"},
    "clyde_wind":  {"lat": 55.45, "lon": -3.65, "label": "Clyde Wind Farm (522 MW)"},
    # Large onshore — England/Wales
    "pen_y_cymoedd": {"lat": 51.75, "lon": -3.6, "label": "Pen y Cymoedd (228 MW, Wales)"},
}
