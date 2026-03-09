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
                gas_ttf_eur      REAL               -- EUR per MWh (TTF European gas)
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
                predicted_on            TEXT NOT NULL,  -- YYYY-MM-DD date prediction was made
                date                    TEXT NOT NULL,  -- YYYY-MM-DD date being predicted
                predicted_ex_vat_p_kwh  REAL,
                predicted_inc_vat_p_kwh REAL,
                PRIMARY KEY (predicted_on, date)
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
        """)
        # Migrate existing DBs: add columns introduced after initial schema
        _migrate_generation_schema(conn)


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
            """INSERT OR REPLACE INTO commodity_prices (date, brent_crude_usd, gas_ttf_eur)
               VALUES (:date, :brent_crude_usd, :gas_ttf_eur)""",
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
            """SELECT date, brent_crude_usd, gas_ttf_eur
               FROM commodity_prices
               WHERE date BETWEEN ? AND ?
               ORDER BY date""",
            (str(date_from), str(date_to)),
        ).fetchall()
    return rows


# ── Raw half-hourly / hourly fetches (for half-hourly model) ──────────────────
def get_halfhourly_prices(date_from: date, date_to: date):
    """Return raw half-hourly price rows."""
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
               (predicted_on, date, predicted_ex_vat_p_kwh, predicted_inc_vat_p_kwh)
               VALUES (:predicted_on, :date, :predicted_ex_vat_p_kwh, :predicted_inc_vat_p_kwh)""",
            [{"predicted_on": str(predicted_on), **r} for r in rows],
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
                   p.predicted_ex_vat_p_kwh, p.predicted_inc_vat_p_kwh,
                   AVG(pr.price_ex_vat)  AS actual_ex_vat,
                   AVG(pr.price_inc_vat) AS actual_inc_vat
            FROM daily_predictions p
            JOIN prices pr ON DATE(pr.datetime) = p.date
            WHERE p.date <= ?
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


# ── Fetch log ──────────────────────────────────────────────────────────────────
def log_fetch(source: str, date_from: date, date_to: date, records: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO fetch_log (source, fetched_at, date_from, date_to, records)
               VALUES (?, ?, ?, ?, ?)""",
            (source, datetime.utcnow().isoformat(), str(date_from), str(date_to), records),
        )
