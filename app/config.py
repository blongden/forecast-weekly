"""
Central configuration for the energy analysis app.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

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
