"""
Database layer — SQLite via Python's built-in sqlite3.

Schema
------
prices          : half-hourly Agile prices (raw + wholesale)
weather_hourly  : hourly Open-Meteo historical data
daily_summary   : pre-aggregated daily view (rebuilt on demand)
"""
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Generator

from app.config import DB_PATH


# ── Connection helper ──────────────────────────────────────────────────────────
@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ──────────────────────────────────────────────────────
def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prices (
                datetime        TEXT PRIMARY KEY,  -- ISO-8601 UTC, e.g. 2025-03-09T00:00:00Z
                price_inc_vat   REAL NOT NULL,     -- p/kWh as returned by Octopus API
                price_ex_vat    REAL NOT NULL,     -- price_inc_vat / 1.05
                wholesale_price REAL NOT NULL,     -- reverse-engineered wholesale cost p/kWh
                is_peak         INTEGER NOT NULL   -- 1 = 16:00-19:00 Europe/London, else 0
            );

            CREATE TABLE IF NOT EXISTS weather_hourly (
                datetime            TEXT PRIMARY KEY,  -- ISO-8601 local time
                temperature_2m      REAL,
                wind_speed_10m      REAL,
                shortwave_radiation REAL,
                precipitation       REAL
            );

            CREATE TABLE IF NOT EXISTS commodity_prices (
                date             TEXT PRIMARY KEY,  -- YYYY-MM-DD
                brent_crude_usd  REAL,              -- USD per barrel (Brent)
                gas_ttf_eur      REAL,              -- EUR per MWh (TTF European gas)
                gbpusd           REAL,              -- GBP/USD exchange rate
                usd_index        REAL,              -- US Dollar Index (DXY)
                carbon_ets_gbp   REAL               -- EU ETS carbon price (GBP/tonne)
            );

            CREATE TABLE IF NOT EXISTS fetch_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,   -- 'octopus' | 'weather_historical' | 'weather_forecast'
                fetched_at  TEXT NOT NULL,
                date_from   TEXT NOT NULL,
                date_to     TEXT NOT NULL,
                records     INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS solar_generation (
                datetime_gmt  TEXT PRIMARY KEY,  -- UTC ISO-8601 (30-min intervals)
                generation_mw REAL               -- GB national solar generation (MW)
            );

            CREATE TABLE IF NOT EXISTS weather_uk_sites (
                datetime            TEXT NOT NULL,  -- ISO-8601 local time
                site_id             TEXT NOT NULL,  -- key from config.UK_WEATHER_SITES
                temperature_2m      REAL,
                shortwave_radiation REAL,
                precipitation       REAL,
                PRIMARY KEY (datetime, site_id)
            );

            CREATE TABLE IF NOT EXISTS weather_wind_sites (
                datetime   TEXT NOT NULL,  -- ISO-8601 local time
                site_id    TEXT NOT NULL,  -- key from config.WIND_SITES
                wind_speed REAL,           -- wind speed at 100m hub height (km/h)
                PRIMARY KEY (datetime, site_id)
            );

            CREATE TABLE IF NOT EXISTS daily_predictions (
                predicted_on         TEXT NOT NULL,  -- YYYY-MM-DD date prediction was made
                date                 TEXT NOT NULL,  -- YYYY-MM-DD date being predicted
                predicted_epex_p_kwh REAL,           -- predicted EPEX wholesale price (p/kWh)
                PRIMARY KEY (predicted_on, date)
            );

            CREATE TABLE IF NOT EXISTS halfhourly_predictions (
                predicted_on         TEXT NOT NULL,  -- YYYY-MM-DD date prediction was made
                datetime_utc         TEXT NOT NULL,  -- ISO-8601 UTC slot start
                predicted_epex_p_kwh REAL,           -- blended ensemble price (p/kWh)
                pred_q10             REAL,           -- 10th percentile (p/kWh)
                pred_q90             REAL,           -- 90th percentile (p/kWh)
                PRIMARY KEY (predicted_on, datetime_utc)
            );

            CREATE TABLE IF NOT EXISTS forecast_summary (
                predicted_on   TEXT NOT NULL,
                week_summary   TEXT,
                days_json      TEXT,
                created_at     TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (predicted_on)
            );

            CREATE TABLE IF NOT EXISTS demand_halfhourly (
                datetime_utc  TEXT PRIMARY KEY,  -- ISO-8601 UTC, period-start (aligns with prices)
                demand_mw     REAL               -- GB Initial National Demand Outturn (MW)
            );

            CREATE TABLE IF NOT EXISTS market_index_halfhourly (
                datetime_utc    TEXT PRIMARY KEY,  -- ISO-8601 UTC, period-start
                price_gbp_mwh   REAL,              -- EPEX SPOT GB day-ahead price (£/MWh)
                volume_mwh      REAL               -- traded volume (MWh)
            );

            CREATE TABLE IF NOT EXISTS generation_halfhourly (
                datetime_utc        TEXT PRIMARY KEY,  -- ISO-8601 UTC, period-start
                wind_mw             REAL,              -- GB wind generation (onshore + offshore)
                gas_mw              REAL,              -- CCGT + OCGT
                nuclear_mw          REAL,              -- nuclear
                pumped_storage_mw   REAL,              -- pumped hydro storage (PS) — price-reactive
                hydro_mw            REAL,              -- run-of-river hydro (NPSHYD)
                imports_mw          REAL               -- net interconnector flows (positive = importing)
            );

            CREATE TABLE IF NOT EXISTS gas_storage (
                date        TEXT PRIMARY KEY,  -- YYYY-MM-DD
                eu_gas_pct  REAL,              -- EU aggregate fill level (%)
                eu_gas_twh  REAL,              -- EU working gas in storage (TWh)
                gb_gas_pct  REAL,              -- GB fill level (%)
                gb_gas_twh  REAL               -- GB working gas in storage (TWh)
            );

            CREATE TABLE IF NOT EXISTS oil_inventory (
                date               TEXT PRIMARY KEY,  -- YYYY-MM-DD (weekly, EIA release date)
                us_crude_stocks_mb REAL               -- US commercial crude stocks (million barrels)
            );

            CREATE TABLE IF NOT EXISTS weather_forecast_archive (
                fetch_date          TEXT NOT NULL,   -- YYYY-MM-DD: date the forecast was made
                target_date         TEXT NOT NULL,   -- YYYY-MM-DD: date being forecast
                lead_days           INTEGER NOT NULL, -- target_date - fetch_date in days
                temperature_2m      REAL,            -- UK avg forecast temperature (°C)
                shortwave_radiation REAL,            -- UK avg forecast solar radiation (W/m²)
                precipitation       REAL,            -- UK avg forecast precipitation (mm)
                PRIMARY KEY (fetch_date, target_date)
            );

            CREATE TABLE IF NOT EXISTS wind_site_forecast_archive (
                fetch_date  TEXT NOT NULL,   -- YYYY-MM-DD: date the forecast was made
                target_date TEXT NOT NULL,   -- YYYY-MM-DD: date being forecast
                site_id     TEXT NOT NULL,   -- key from config.WIND_SITES
                wind_speed  REAL,            -- daily avg 100m wind speed forecast (km/h)
                PRIMARY KEY (fetch_date, target_date, site_id)
            );

            CREATE TABLE IF NOT EXISTS system_prices (
                datetime_utc       TEXT PRIMARY KEY,
                system_buy_price   REAL,
                system_sell_price  REAL,
                net_imbalance_mw   REAL
            );

            CREATE TABLE IF NOT EXISTS entsoe_scheduled_exchanges (
                datetime_utc   TEXT NOT NULL,   -- ISO-8601 UTC, hourly resolution
                country_from   TEXT NOT NULL,   -- e.g. 'FR', 'BE', 'NL', 'NO_2', 'IE_SEM', 'DK_1'
                country_to     TEXT NOT NULL,   -- e.g. 'GB' for imports, reverse for exports
                scheduled_mw   REAL,            -- MW scheduled in this direction
                PRIMARY KEY (datetime_utc, country_from, country_to)
            );

            CREATE TABLE IF NOT EXISTS entsoe_unavailability (
                date           TEXT NOT NULL,   -- YYYY-MM-DD
                fuel_type      TEXT NOT NULL,   -- 'nuclear', 'gas', 'coal', 'wind', 'other'
                unavailable_mw REAL,            -- sum of unavailable nominal power (MW)
                PRIMARY KEY (date, fuel_type)
            );
        """)
        # Migrate existing DBs: add columns introduced after initial schema
        _migrate_generation_schema(conn)
        _migrate_predictions_schema(conn)
        _migrate_commodity_schema(conn)
        _migrate_is_actual_column(conn)


def _migrate_predictions_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate daily_predictions if it has the old Agile-schema columns."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(daily_predictions)").fetchall()}
    if "predicted_ex_vat_p_kwh" in cols:
        # Old Agile predictions are invalid after EPEX model switch — drop and recreate
        conn.execute("DROP TABLE daily_predictions")
        conn.execute("""
            CREATE TABLE daily_predictions (
                predicted_on         TEXT NOT NULL,
                date                 TEXT NOT NULL,
                predicted_epex_p_kwh REAL,
                PRIMARY KEY (predicted_on, date)
            )
        """)


def _migrate_generation_schema(conn: sqlite3.Connection) -> None:
    """Add columns to generation_halfhourly that were introduced after the initial schema."""
    new_cols = [
        ("pumped_storage_mw", "REAL"),
        ("hydro_mw",          "REAL"),
    ]
    existing = {row[1] for row in conn.execute("PRAGMA table_info(generation_halfhourly)").fetchall()}
    for col, typ in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE generation_halfhourly ADD COLUMN {col} {typ}")


def _migrate_commodity_schema(conn: sqlite3.Connection) -> None:
    """Add gbpusd, usd_index, and carbon_ets_gbp columns to commodity_prices if missing."""
    new_cols = [
        ("gbpusd",          "REAL"),
        ("usd_index",       "REAL"),
        ("carbon_ets_gbp",  "REAL"),
    ]
    existing = {row[1] for row in conn.execute("PRAGMA table_info(commodity_prices)").fetchall()}
    for col, typ in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE commodity_prices ADD COLUMN {col} {typ}")


def _migrate_is_actual_column(conn: sqlite3.Connection) -> None:
    """Add is_actual column to prediction tables if missing."""
    for table in ("daily_predictions", "halfhourly_predictions"):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "is_actual" not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN is_actual INTEGER DEFAULT 0")


def commodity_needs_currency_data() -> bool:
    """Return True if gbpusd column exists but has no data (schema migrated, not yet re-fetched)."""
    with get_conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(commodity_prices)").fetchall()}
        if "gbpusd" not in cols:
            return False
        row = conn.execute(
            "SELECT COUNT(*) FROM commodity_prices WHERE gbpusd IS NOT NULL"
        ).fetchone()
    return row[0] == 0


def generation_needs_migration() -> bool:
    """Return True if pumped_storage_mw exists but has no data (schema migrated, not yet re-fetched)."""
    with get_conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(generation_halfhourly)").fetchall()}
        if "pumped_storage_mw" not in cols:
            return False
        row = conn.execute(
            "SELECT COUNT(*) FROM generation_halfhourly WHERE pumped_storage_mw IS NOT NULL"
        ).fetchone()
    return row[0] == 0


# ── Prices ─────────────────────────────────────────────────────────────────────
def upsert_prices(rows: list[dict]) -> int:
    """Insert or replace price rows. Returns number of rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO prices
               (datetime, price_inc_vat, price_ex_vat, wholesale_price, is_peak)
               VALUES (:datetime, :price_inc_vat, :price_ex_vat, :wholesale_price, :is_peak)""",
            rows,
        )
    return len(rows)


def get_price_date_range() -> tuple[str | None, str | None]:
    """Return (min_datetime, max_datetime) of stored prices, or (None, None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime), MAX(datetime) FROM prices"
        ).fetchone()
    return row[0], row[1]


def get_daily_prices(date_from: date, date_to: date):
    """Return daily average prices as list of sqlite3.Row."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                DATE(datetime) AS date,
                AVG(price_inc_vat)   AS avg_price_inc_vat,
                AVG(price_ex_vat)    AS avg_price_ex_vat,
                AVG(wholesale_price) AS avg_wholesale_price
            FROM prices
            WHERE DATE(datetime) BETWEEN ? AND ?
            GROUP BY DATE(datetime)
            ORDER BY date
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Weather ────────────────────────────────────────────────────────────────────
def upsert_weather(rows: list[dict]) -> int:
    """Insert or replace weather rows. Returns number of rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO weather_hourly
               (datetime, temperature_2m, wind_speed_10m, shortwave_radiation, precipitation)
               VALUES (:datetime, :temperature_2m, :wind_speed_10m,
                       :shortwave_radiation, :precipitation)""",
            rows,
        )
    return len(rows)


def get_weather_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime), MAX(datetime) FROM weather_hourly"
        ).fetchone()
    return row[0], row[1]


def get_daily_weather(date_from: date, date_to: date):
    """Return daily aggregated weather as list of sqlite3.Row."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                DATE(datetime) AS date,
                AVG(temperature_2m)      AS temperature_2m,
                AVG(wind_speed_10m)      AS wind_speed_10m,
                AVG(shortwave_radiation) AS shortwave_radiation,
                SUM(precipitation)       AS precipitation
            FROM weather_hourly
            WHERE DATE(datetime) BETWEEN ? AND ?
            GROUP BY DATE(datetime)
            ORDER BY date
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Solar generation (Sheffield Solar PV_Live) ────────────────────────────────
def upsert_solar(rows: list[dict]) -> int:
    """Insert or replace 30-min solar generation rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO solar_generation (datetime_gmt, generation_mw)
               VALUES (:datetime_gmt, :generation_mw)""",
            rows,
        )
    return len(rows)


def get_solar_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime_gmt), MAX(datetime_gmt) FROM solar_generation"
        ).fetchone()
    return row[0], row[1]


def get_daily_solar(date_from: date, date_to: date) -> list:
    """Return daily average GB solar generation (GW) and total GWh."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DATE(datetime_gmt) AS date,
                   AVG(generation_mw) / 1000.0 AS solar_gw,
                   SUM(generation_mw) * 0.5 / 1000.0 AS solar_gwh
            FROM solar_generation
            WHERE DATE(datetime_gmt) BETWEEN ? AND ?
            GROUP BY DATE(datetime_gmt)
            ORDER BY date
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_halfhourly_solar(date_from: date, date_to: date) -> list:
    """Return 30-min solar generation rows as (datetime_gmt, generation_mw)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT datetime_gmt, generation_mw
            FROM solar_generation
            WHERE DATE(datetime_gmt) BETWEEN ? AND ?
            ORDER BY datetime_gmt
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── UK weather sites (temperature, solar, precipitation) ─────────────────────
def upsert_uk_sites(rows: list[dict]) -> int:
    """Insert or replace UK weather site rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO weather_uk_sites
               (datetime, site_id, temperature_2m, shortwave_radiation, precipitation)
               VALUES (:datetime, :site_id, :temperature_2m,
                       :shortwave_radiation, :precipitation)""",
            rows,
        )
    return len(rows)


def get_uk_site_date_range(site_id: str) -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime), MAX(datetime) FROM weather_uk_sites WHERE site_id = ?",
            (site_id,),
        ).fetchone()
    return row[0], row[1]


def get_daily_uk_avg(date_from: date, date_to: date) -> list:
    """Return daily UK-average temperature, solar radiation, and precipitation."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            WITH daily_site AS (
                SELECT DATE(datetime) AS date, site_id,
                       AVG(temperature_2m)      AS temperature_2m,
                       AVG(shortwave_radiation) AS shortwave_radiation,
                       SUM(precipitation)       AS precipitation
                FROM weather_uk_sites
                WHERE DATE(datetime) BETWEEN ? AND ?
                GROUP BY DATE(datetime), site_id
            )
            SELECT date,
                   AVG(temperature_2m)      AS temperature_2m,
                   AVG(shortwave_radiation) AS shortwave_radiation,
                   AVG(precipitation)       AS precipitation
            FROM daily_site
            GROUP BY date
            ORDER BY date
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_hourly_uk_avg(date_from: date, date_to: date) -> list:
    """Return hourly UK-average temperature, solar radiation, and precipitation."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT datetime,
                   AVG(temperature_2m)      AS temperature_2m,
                   AVG(shortwave_radiation) AS shortwave_radiation,
                   AVG(precipitation)       AS precipitation
            FROM weather_uk_sites
            WHERE DATE(datetime) BETWEEN ? AND ?
            GROUP BY datetime
            ORDER BY datetime
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Offshore wind site weather ────────────────────────────────────────────────
def upsert_wind_sites(rows: list[dict]) -> int:
    """Insert or replace wind speed rows for a specific site. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO weather_wind_sites (datetime, site_id, wind_speed)
               VALUES (:datetime, :site_id, :wind_speed)""",
            rows,
        )
    return len(rows)


def get_wind_site_date_range(site_id: str) -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime), MAX(datetime) FROM weather_wind_sites WHERE site_id = ?",
            (site_id,),
        ).fetchone()
    return row[0], row[1]


def get_daily_wind_sites(date_from: date, date_to: date) -> list:
    """Return daily avg wind speed per site as list of (date, site_id, wind_speed)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DATE(datetime) AS date, site_id, AVG(wind_speed) AS wind_speed
            FROM weather_wind_sites
            WHERE DATE(datetime) BETWEEN ? AND ?
            GROUP BY DATE(datetime), site_id
            ORDER BY date, site_id
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_hourly_wind_sites(date_from: date, date_to: date) -> list:
    """Return hourly wind speed per site."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT datetime, site_id, wind_speed
            FROM weather_wind_sites
            WHERE DATE(datetime) BETWEEN ? AND ?
            ORDER BY datetime, site_id
            """,
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Commodity prices ──────────────────────────────────────────────────────────
def upsert_commodity(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO commodity_prices
               (date, brent_crude_usd, gas_ttf_eur, gbpusd, usd_index, carbon_ets_gbp)
               VALUES (:date, :brent_crude_usd, :gas_ttf_eur, :gbpusd, :usd_index, :carbon_ets_gbp)""",
            rows,
        )
    return len(rows)


def get_commodity_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM commodity_prices"
        ).fetchone()
    return row[0], row[1]


def get_commodity_prices(date_from: date, date_to: date):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date, brent_crude_usd, gas_ttf_eur, gbpusd, usd_index, carbon_ets_gbp
               FROM commodity_prices
               WHERE date BETWEEN ? AND ?
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Raw half-hourly / hourly fetches (for half-hourly model) ──────────────────
def get_halfhourly_prices(date_from: date, date_to: date):
    """Return actual Octopus Agile half-hourly price rows from the prices table."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime, price_ex_vat, price_inc_vat, wholesale_price, is_peak
               FROM prices
               WHERE DATE(datetime) BETWEEN ? AND ?
               ORDER BY datetime""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_hourly_weather(date_from: date, date_to: date):
    """Return raw hourly weather rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime, temperature_2m, wind_speed_10m,
                      shortwave_radiation, precipitation
               FROM weather_hourly
               WHERE DATE(datetime) BETWEEN ? AND ?
               ORDER BY datetime""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Stored predictions ────────────────────────────────────────────────────────
def upsert_daily_predictions(predicted_on: date, rows: list[dict]) -> int:
    """Store daily forecast rows keyed by (predicted_on, date)."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO daily_predictions
               (predicted_on, date, predicted_epex_p_kwh, is_actual)
               VALUES (:predicted_on, :date, :predicted_epex_p_kwh, :is_actual)""",
            [{"predicted_on": str(predicted_on), "is_actual": 0, **r} for r in rows],
        )
    return len(rows)


def upsert_halfhourly_predictions(predicted_on: date, rows: list[dict]) -> int:
    """Store half-hourly forecast rows keyed by (predicted_on, datetime_utc)."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO halfhourly_predictions
               (predicted_on, datetime_utc, predicted_epex_p_kwh, pred_q10, pred_q90, is_actual)
               VALUES (:predicted_on, :datetime_utc, :predicted_epex_p_kwh, :pred_q10, :pred_q90, :is_actual)""",
            [{"predicted_on": str(predicted_on), "is_actual": 0, **r} for r in rows],
        )
    return len(rows)


def get_verifiable_predictions(as_of: date) -> list:
    """
    Return stored predictions where actual price data now exists —
    i.e. the predicted date is in the past and we have real prices for it.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.predicted_on, p.date,
                   p.predicted_epex_p_kwh,
                   AVG(m.price_gbp_mwh) * 0.1 AS actual_epex_p_kwh
            FROM daily_predictions p
            JOIN market_index_halfhourly m ON DATE(m.datetime_utc) = p.date
            WHERE p.date <= ?
              AND (p.is_actual IS NULL OR p.is_actual = 0)
            GROUP BY p.predicted_on, p.date
            ORDER BY p.date, p.predicted_on
            """,
            (str(as_of),),
        ).fetchall()
    return rows


# ── Demand (Elexon BMRS INDO) ─────────────────────────────────────────────────
def upsert_demand(rows: list[dict]) -> int:
    """Insert or replace half-hourly GB demand rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO demand_halfhourly (datetime_utc, demand_mw)
               VALUES (:datetime_utc, :demand_mw)""",
            rows,
        )
    return len(rows)


def get_demand_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime_utc), MAX(datetime_utc) FROM demand_halfhourly"
        ).fetchone()
    return row[0], row[1]


def get_halfhourly_demand(date_from: date, date_to: date) -> list:
    """Return half-hourly demand rows as (datetime_utc, demand_mw)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime_utc, demand_mw
               FROM demand_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               ORDER BY datetime_utc""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_daily_demand(date_from: date, date_to: date) -> list:
    """Return daily average demand (MW) as (date, demand_mw)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DATE(datetime_utc) AS date, AVG(demand_mw) AS demand_mw
               FROM demand_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY DATE(datetime_utc)
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Generation mix (Elexon BMRS FUELHH) ──────────────────────────────────────
def upsert_generation(rows: list[dict]) -> int:
    """Insert or replace half-hourly generation mix rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO generation_halfhourly
               (datetime_utc, wind_mw, gas_mw, nuclear_mw,
                pumped_storage_mw, hydro_mw, imports_mw)
               VALUES (:datetime_utc, :wind_mw, :gas_mw, :nuclear_mw,
                       :pumped_storage_mw, :hydro_mw, :imports_mw)""",
            rows,
        )
    return len(rows)


def get_generation_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime_utc), MAX(datetime_utc) FROM generation_halfhourly"
        ).fetchone()
    return row[0], row[1]


def get_halfhourly_generation(date_from: date, date_to: date) -> list:
    """Return half-hourly generation rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime_utc, wind_mw, gas_mw, nuclear_mw,
                      pumped_storage_mw, hydro_mw, imports_mw
               FROM generation_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               ORDER BY datetime_utc""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_daily_generation(date_from: date, date_to: date) -> list:
    """Return daily average generation mix (MW)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DATE(datetime_utc) AS date,
                      AVG(wind_mw)           AS wind_mw,
                      AVG(gas_mw)            AS gas_mw,
                      AVG(nuclear_mw)        AS nuclear_mw,
                      AVG(pumped_storage_mw) AS pumped_storage_mw,
                      AVG(hydro_mw)          AS hydro_mw,
                      AVG(imports_mw)        AS imports_mw
               FROM generation_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY DATE(datetime_utc)
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── EPEX day-ahead market index prices ────────────────────────────────────────
def upsert_midprice(rows: list[dict]) -> int:
    """Insert or replace EPEX day-ahead price rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO market_index_halfhourly
               (datetime_utc, price_gbp_mwh, volume_mwh)
               VALUES (:datetime_utc, :price_gbp_mwh, :volume_mwh)""",
            rows,
        )
    return len(rows)


def get_midprice_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime_utc), MAX(datetime_utc) FROM market_index_halfhourly"
        ).fetchone()
    return row[0], row[1]


def upsert_forecast_summary(predicted_on, week_summary: str, days: list) -> None:
    import json
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO forecast_summary (predicted_on, week_summary, days_json)
               VALUES (?, ?, ?)
               ON CONFLICT(predicted_on) DO UPDATE SET
                   week_summary = excluded.week_summary,
                   days_json    = excluded.days_json,
                   created_at   = datetime('now')""",
            (str(predicted_on), week_summary, json.dumps(days)),
        )


def get_forecast_summary(predicted_on) -> dict | None:
    import json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT week_summary, days_json FROM forecast_summary WHERE predicted_on = ?",
            (str(predicted_on),),
        ).fetchone()
    if not row:
        return None
    return {"week_summary": row[0], "days": json.loads(row[1])}


def get_last_complete_midprice_date(min_slots: int = 46):
    """Return the latest date that has at least `min_slots` HH entries, or None."""
    from datetime import date as _date
    with get_conn() as conn:
        row = conn.execute(
            """SELECT date(datetime_utc) AS d
               FROM market_index_halfhourly
               GROUP BY d
               HAVING COUNT(*) >= ?
               ORDER BY d DESC
               LIMIT 1""",
            (min_slots,),
        ).fetchone()
    return _date.fromisoformat(row[0]) if row else None


def has_complete_midprice(target_date: date, min_slots: int = 46) -> bool:
    """Return True if `target_date` has at least `min_slots` HH entries."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM market_index_halfhourly
               WHERE DATE(datetime_utc) = ?""",
            (str(target_date),),
        ).fetchone()
    return row[0] >= min_slots


def get_halfhourly_midprice(date_from: date, date_to: date) -> list:
    """Return half-hourly EPEX day-ahead price rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime_utc, price_gbp_mwh
               FROM market_index_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               ORDER BY datetime_utc""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_daily_midprice(date_from: date, date_to: date) -> list:
    """Return daily average EPEX day-ahead price (£/MWh)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DATE(datetime_utc) AS date,
                      AVG(price_gbp_mwh) AS price_gbp_mwh
               FROM market_index_halfhourly
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY DATE(datetime_utc)
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Forecast archive ───────────────────────────────────────────────────────────
def upsert_weather_forecast_archive(rows: list[dict]) -> int:
    """Store daily UK-avg weather forecasts keyed by (fetch_date, target_date)."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO weather_forecast_archive
               (fetch_date, target_date, lead_days,
                temperature_2m, shortwave_radiation, precipitation)
               VALUES (:fetch_date, :target_date, :lead_days,
                       :temperature_2m, :shortwave_radiation, :precipitation)""",
            rows,
        )
    return len(rows)


def upsert_wind_site_forecast_archive(rows: list[dict]) -> int:
    """Store daily per-site wind speed forecasts keyed by (fetch_date, target_date, site_id)."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO wind_site_forecast_archive
               (fetch_date, target_date, site_id, wind_speed)
               VALUES (:fetch_date, :target_date, :site_id, :wind_speed)""",
            rows,
        )
    return len(rows)


def get_weather_forecast_archive(target_date_from: date, target_date_to: date) -> list:
    """Return all archived weather forecasts for a target date range."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT fetch_date, target_date, lead_days,
                      temperature_2m, shortwave_radiation, precipitation
               FROM weather_forecast_archive
               WHERE target_date BETWEEN ? AND ?
               ORDER BY target_date, fetch_date""",
            (str(target_date_from), str(target_date_to)),
        ).fetchall()
    return rows


def get_wind_site_forecast_archive(target_date_from: date, target_date_to: date) -> list:
    """Return all archived wind site forecasts for a target date range."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT fetch_date, target_date, site_id, wind_speed
               FROM wind_site_forecast_archive
               WHERE target_date BETWEEN ? AND ?
               ORDER BY target_date, fetch_date, site_id""",
            (str(target_date_from), str(target_date_to)),
        ).fetchall()
    return rows


def get_forecast_archive_fetch_dates() -> list[str]:
    """Return distinct fetch_dates in the weather_forecast_archive."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT fetch_date FROM weather_forecast_archive ORDER BY fetch_date"
        ).fetchall()
    return [r[0] for r in rows]


# ── Gas storage (GIE AGSI+) ──────────────────────────────────────────────────
def upsert_gas_storage(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO gas_storage
               (date, eu_gas_pct, eu_gas_twh, gb_gas_pct, gb_gas_twh)
               VALUES (:date, :eu_gas_pct, :eu_gas_twh, :gb_gas_pct, :gb_gas_twh)""",
            rows,
        )
    return len(rows)


def get_gas_storage_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM gas_storage"
        ).fetchone()
    return row[0], row[1]


def get_gas_storage(date_from: date, date_to: date):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date, eu_gas_pct, eu_gas_twh, gb_gas_pct, gb_gas_twh
               FROM gas_storage
               WHERE date BETWEEN ? AND ?
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Oil inventory (EIA) ─────────────────────────────────────────────────────
def upsert_oil_inventory(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO oil_inventory
               (date, us_crude_stocks_mb)
               VALUES (:date, :us_crude_stocks_mb)""",
            rows,
        )
    return len(rows)


def get_oil_inventory_date_range() -> tuple[str | None, str | None]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM oil_inventory"
        ).fetchone()
    return row[0], row[1]


def get_oil_inventory(date_from: date, date_to: date):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date, us_crude_stocks_mb
               FROM oil_inventory
               WHERE date BETWEEN ? AND ?
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── ENTSO-E scheduled exchanges & unavailability ──────────────────────────────

def upsert_entsoe_exchanges(rows: list[dict]) -> int:
    """Store hourly scheduled exchange rows: {datetime_utc, country_from, country_to, scheduled_mw}."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO entsoe_scheduled_exchanges
               (datetime_utc, country_from, country_to, scheduled_mw)
               VALUES (:datetime_utc, :country_from, :country_to, :scheduled_mw)""",
            rows,
        )
    return len(rows)


def upsert_entsoe_unavailability(rows: list[dict]) -> int:
    """Store daily unavailability rows: {date, fuel_type, unavailable_mw}."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO entsoe_unavailability
               (date, fuel_type, unavailable_mw)
               VALUES (:date, :fuel_type, :unavailable_mw)""",
            rows,
        )
    return len(rows)


def get_entsoe_exchanges_date_range():
    """Return (min_date, max_date) of scheduled exchange data, or (None, None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(DATE(datetime_utc)), MAX(DATE(datetime_utc)) FROM entsoe_scheduled_exchanges"
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    return row[0], row[1]


def get_entsoe_unavailability_date_range():
    """Return (min_date, max_date) of unavailability data, or (None, None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM entsoe_unavailability"
        ).fetchone()
    if row is None or row[0] is None:
        return None, None
    return row[0], row[1]


def get_daily_scheduled_imports(date_from: date, date_to: date) -> list:
    """Return daily net scheduled imports (MW) into GB, aggregated across all interconnectors."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DATE(datetime_utc) AS date,
                      (SUM(CASE WHEN country_to = 'GB' THEN scheduled_mw ELSE 0 END) -
                       SUM(CASE WHEN country_from = 'GB' THEN scheduled_mw ELSE 0 END))
                       / COUNT(DISTINCT datetime_utc) AS net_scheduled_imports_mw
               FROM entsoe_scheduled_exchanges
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY DATE(datetime_utc)
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_hourly_scheduled_imports(date_from: date, date_to: date) -> list:
    """Return hourly net scheduled imports (MW) into GB for HH model."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT datetime_utc,
                      SUM(CASE WHEN country_to = 'GB' THEN scheduled_mw ELSE 0 END) -
                      SUM(CASE WHEN country_from = 'GB' THEN scheduled_mw ELSE 0 END) AS net_mw
               FROM entsoe_scheduled_exchanges
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY datetime_utc
               ORDER BY datetime_utc""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


def get_daily_unavailability(date_from: date, date_to: date) -> list:
    """Return daily unavailability by fuel type: (date, fuel_type, unavailable_mw)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date, fuel_type, unavailable_mw
               FROM entsoe_unavailability
               WHERE date BETWEEN ? AND ?
               ORDER BY date, fuel_type""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── System prices (Elexon BMRS cash-out / imbalance) ────────────────────────

def upsert_system_prices(rows: list[dict]) -> int:
    """Insert or replace half-hourly system price rows. Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO system_prices (datetime_utc, system_buy_price, system_sell_price, net_imbalance_mw)
               VALUES (:datetime_utc, :system_buy_price, :system_sell_price, :net_imbalance_mw)
               ON CONFLICT(datetime_utc) DO UPDATE SET
                   system_buy_price  = excluded.system_buy_price,
                   system_sell_price = excluded.system_sell_price,
                   net_imbalance_mw  = excluded.net_imbalance_mw""",
            rows,
        )
    return len(rows)


def get_sysprice_date_range() -> tuple[str | None, str | None]:
    """Return (min_datetime, max_datetime) of stored system prices, or (None, None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(datetime_utc), MAX(datetime_utc) FROM system_prices"
        ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def get_daily_system_prices(date_from: date, date_to: date) -> list:
    """Return daily avg system buy price and mean absolute imbalance volume."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT DATE(datetime_utc) as date,
                      AVG(system_buy_price) as avg_system_price,
                      AVG(ABS(net_imbalance_mw)) as avg_abs_imbalance_mw
               FROM system_prices
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               GROUP BY DATE(datetime_utc)
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()


def get_halfhourly_system_prices(date_from: date, date_to: date) -> list:
    """Return half-hourly system prices as list of dicts."""
    with get_conn() as conn:
        conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
        return conn.execute(
            """SELECT datetime_utc, system_buy_price, system_sell_price, net_imbalance_mw
               FROM system_prices
               WHERE DATE(datetime_utc) BETWEEN ? AND ?
               ORDER BY datetime_utc""",
            (str(date_from), str(date_to)),
        ).fetchall()


# ── Fetch log ──────────────────────────────────────────────────────────────────
def log_fetch(source: str, date_from: date, date_to: date, records: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO fetch_log (source, fetched_at, date_from, date_to, records)
               VALUES (?, ?, ?, ?, ?)""",
            (source, datetime.utcnow().isoformat(), str(date_from), str(date_to), records),
        )
