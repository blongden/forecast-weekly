"""
Central configuration for the energy analysis app.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DB_PATH    = BASE_DIR / "energy.db"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Octopus Agile ─────────────────────────────────────────────────────────────
# Region N = Southern Scotland (Scotland Central Belt)
# https://www.guylipman.com/octopus/formulas.html
OCTOPUS_PRODUCT = "AGILE-24-10-01"
OCTOPUS_REGION  = "N"
OCTOPUS_TARIFF  = f"E-1R-{OCTOPUS_PRODUCT}-{OCTOPUS_REGION}"
OCTOPUS_BASE    = "https://api.octopus.energy/v1"

# Agile formula: price_ex_vat = D * wholesale + P (peak) / D * wholesale (off-peak)
# Reverse:       wholesale = (price_ex_vat - P) / D  [peak]
#                wholesale =  price_ex_vat / D        [off-peak]
AGILE_D   = 2.1   # distribution multiplier for Southern Scotland
AGILE_P   = 13.0  # peak adder p/kWh, applied 16:00–19:00 local time
AGILE_VAT = 1.05  # 5% VAT on electricity

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
