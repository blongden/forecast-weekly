"""
Customer behaviour simulation under our 3-band tariff vs Octopus Agile.

Models four load-shifting scenarios to estimate:
  - Customer annual bill (our tariff vs Agile)
  - Supplier annual profit per customer (our tariff vs Agile reseller)

Uses 60-day historical half-hourly data and annualises by (365 / n_days).
Supplier cost basis = actual EPEX wholesale price per slot (trading bought ahead).

Load profile: Elexon Profile Class 1 (domestic unrestricted), seasonal + day-type split.
  - Red band applies Mon–Fri only; weekends are entirely green rate (SPD LC14 definition).
  - Winter = Oct–Mar, Summer = Apr–Sep.
  - Fractions derived from published Elexon PC1 half-hourly coefficient tables.
"""
import pandas as pd
from app.config import (DUOS_RATES, TNUOS_RATE, BSUOS_BUFFER, SUPPLIER_MULTIPLIER,
                         POLICY_LEVY_P_KWH, SUPPLIER_OPEX_P_KWH,
                         OFGEM_STANDING_CHARGE_P_DAY, OFGEM_UNIT_RATE_P_KWH)

VAT = 1.05

# Behaviour scenarios: (id, shift_frac, label, ev_kwh_per_day)
#   shift_frac    — fraction of red-band base load shifted to green
#   ev_kwh_per_day — additional EV consumption added entirely to green band
#                    (models overnight charging, separate from shifted household load)
SCENARIOS = [
    ("no_shift",     0.00, "No shifting\n(price inelastic)",       0.0),
    ("light_shift",  0.25, "Light shifting\n(dishwasher/washing)",  0.0),
    ("heavy_shift",  0.50, "Heavy shifting\n(smart appliances)",     0.0),
    ("ev_household", 0.50, "EV household\n(base shift + EV overnight)", 4.0),
]

# UK average household: 8 kWh/day base (EV adds on top per scenario)
DAILY_KWH = 8.0

# ── Elexon PC1-derived band fractions (red, amber, green) ─────────────────────
# Source: Elexon Profile Class 1 half-hourly coefficient tables (domestic unrestricted).
# Red band (16:00–18:59) applies Mon–Fri only — weekends are entirely green rate.
# Fractions are of total daily_kwh and sum to 1.0 per profile.
#
# Winter (Oct–Mar): heavy evening peak, more heating load
# Summer (Apr–Sep): shallower evening peak, more daylight
#
#                              red    amber   green
SEASONAL_PROFILES = {
    ("winter", "weekday"): (0.22,  0.53,   0.25),
    ("winter", "weekend"): (0.00,  0.50,   0.50),
    ("summer", "weekday"): (0.14,  0.58,   0.28),
    ("summer", "weekend"): (0.00,  0.47,   0.53),
}


def _get_season(month: int) -> str:
    return "winter" if month in (10, 11, 12, 1, 2, 3) else "summer"


def _get_day_type(dow: int) -> str:
    """0=Mon … 6=Sun"""
    return "weekend" if dow >= 5 else "weekday"


def _hour_to_duos_band(h: int) -> str:
    """Map local hour to DUoS band (Mon–Fri only; weekends overridden to green after)."""
    if 16 <= h <= 18:
        return "red"
    if (7 <= h <= 15) or (19 <= h <= 22):
        return "amber"
    return "green"


def run_simulation(agile_hist: pd.DataFrame, daily_kwh: float = DAILY_KWH,
                   multiplier: float = SUPPLIER_MULTIPLIER) -> pd.DataFrame:
    """
    Simulate customer bills and supplier economics across four load-shifting scenarios.

    Parameters
    ----------
    agile_hist : DataFrame with columns datetime, price_ex_vat, price_inc_vat,
                 wholesale_price, is_peak — typically 60 days of actual half-hourly data.
    daily_kwh  : UK average household daily consumption (default 8.0 kWh/day).
    multiplier : Wholesale multiplier for our tariff pricing (default = SUPPLIER_MULTIPLIER).
                 Pass a different value to test alternative pricing strategies.

    Returns
    -------
    pd.DataFrame — one row per scenario with customer and supplier monetary figures
                   annualised to a full year (£/customer/year).
                   Includes a 'multiplier' column for when results from multiple runs
                   are concatenated for comparison.
    """
    if agile_hist is None or agile_hist.empty:
        return pd.DataFrame()

    # ── Prepare slot DataFrame ────────────────────────────────────────────────
    ah = agile_hist.copy()
    ah["dt_local"] = (
        pd.to_datetime(ah["datetime"], utc=True)
        .dt.tz_convert("Europe/London")
        .dt.tz_localize(None)
    )
    ah["date"]       = ah["dt_local"].dt.date
    ah["hour"]       = ah["dt_local"].dt.hour
    ah["month"]      = ah["dt_local"].dt.month
    ah["dow"]        = ah["dt_local"].dt.dayofweek   # 0=Mon
    ah["is_weekday"] = ah["dow"] < 5
    ah["season"]     = ah["month"].map(_get_season)
    ah["day_type"]   = ah["dow"].map(_get_day_type)

    # Assign DUoS band — red/amber by hour on weekdays, green all day on weekends
    ah["duos_band"] = ah["hour"].map(_hour_to_duos_band)
    ah.loc[~ah["is_weekday"], "duos_band"] = "green"   # weekends: always green rate

    # ── Our tariff: daily band-mean wholesale → multiplier + network charges ─
    band_means = (
        ah.groupby(["date", "duos_band"])["wholesale_price"]
        .mean()
        .reset_index()
        .rename(columns={"wholesale_price": "epex_band_mean"})
    )
    band_means["duos_charge"] = band_means["duos_band"].map(DUOS_RATES)
    band_means["our_ex_vat"]  = (
        band_means["epex_band_mean"] * multiplier
        + band_means["duos_charge"] + TNUOS_RATE + BSUOS_BUFFER
    )
    band_means["our_inc_vat"] = band_means["our_ex_vat"] * VAT

    ah = ah.merge(
        band_means[["date", "duos_band", "our_ex_vat", "our_inc_vat", "duos_charge"]],
        on=["date", "duos_band"], how="left",
    )

    # ── Slot counts per (date, band) for uniform kWh allocation ──────────────
    slot_counts = (
        ah.groupby(["date", "duos_band"]).size()
        .rename("n_slots").reset_index()
    )
    ah = ah.merge(slot_counts, on=["date", "duos_band"], how="left")

    # ── Seasonal per-day band kWh lookup ─────────────────────────────────────
    # Build a per-date table of (red_kwh, amber_kwh, green_kwh) before shifting,
    # using the appropriate Elexon PC1 seasonal profile.
    date_profiles = (
        ah[["date", "season", "day_type"]]
        .drop_duplicates("date")
        .copy()
    )
    date_profiles["red_base"]   = date_profiles.apply(
        lambda r: SEASONAL_PROFILES[(r["season"], r["day_type"])][0] * daily_kwh, axis=1)
    date_profiles["amber_base"] = date_profiles.apply(
        lambda r: SEASONAL_PROFILES[(r["season"], r["day_type"])][1] * daily_kwh, axis=1)
    date_profiles["green_base"] = date_profiles.apply(
        lambda r: SEASONAL_PROFILES[(r["season"], r["day_type"])][2] * daily_kwh, axis=1)

    ah = ah.merge(date_profiles[["date", "red_base", "amber_base", "green_base"]],
                  on="date", how="left")

    # Agile DUoS charge for reseller calculation (same band map)
    ah["agile_duos_charge"] = ah["duos_band"].map(DUOS_RATES)

    n_days = ah["date"].nunique()
    scale  = 365.0 / n_days   # annualisation factor

    rows = []
    for scenario_id, shift_frac, label, ev_kwh in SCENARIOS:
        # ── Per-slot kWh for our tariff (uniform within band, post-shifting) ─
        # Shift a fraction of each day's red_base consumption to green.
        # EV load is additional consumption placed entirely in the green band
        # (overnight charging — independent of the base household profile).
        ah["red_kwh"]   = ah["red_base"]   * (1 - shift_frac)
        ah["amber_kwh"] = ah["amber_base"]
        ah["green_kwh"] = ah["green_base"] + ah["red_base"] * shift_frac + ev_kwh

        ah["kwh_ours"] = ah.apply(
            lambda r: r[f"{r['duos_band']}_kwh"] / r["n_slots"], axis=1
        )

        # ── Per-slot kWh for Agile (green = inverse-price weighted) ──────────
        ah["kwh_agile"] = ah["kwh_ours"].copy()   # red + amber: same as ours

        green_mask = ah["duos_band"] == "green"
        for date_val, day_green in ah[green_mask].groupby("date"):
            green_kwh_today = day_green["green_kwh"].iloc[0]
            prices  = day_green["price_inc_vat"].clip(lower=0.001)
            weights = (1.0 / prices) / (1.0 / prices).sum()
            ah.loc[day_green.index, "kwh_agile"] = weights * green_kwh_today

        # ── Customer bills (inc VAT) — wholesale + network only ─────────────
        cust_ours_p  = (ah["kwh_ours"]  * ah["our_inc_vat"]).sum()
        cust_agile_p = (ah["kwh_agile"] * ah["price_inc_vat"]).sum()

        cust_ours_annual  = cust_ours_p  * scale / 100   # pence → £
        cust_agile_annual = cust_agile_p * scale / 100

        # ── All-in customer bill (realistic) ──────────────────────────────────
        # Adds: policy levies, supplier opex, standing charge — all inc 5% VAT
        annual_kwh_scenario = (daily_kwh + ev_kwh) * 365
        standing_annual   = OFGEM_STANDING_CHARGE_P_DAY * 365 / 100  # already inc VAT
        levy_annual       = POLICY_LEVY_P_KWH * annual_kwh_scenario * VAT / 100
        opex_annual       = SUPPLIER_OPEX_P_KWH * annual_kwh_scenario * VAT / 100
        ofgem_cap_annual  = OFGEM_UNIT_RATE_P_KWH * annual_kwh_scenario / 100 + standing_annual

        cust_allin_ours_annual  = cust_ours_annual + standing_annual + levy_annual + opex_annual
        cust_allin_agile_annual = cust_agile_annual + standing_annual + levy_annual + opex_annual

        # ── Supplier economics — our tariff (ex-VAT) ─────────────────────────
        sup_rev_p       = (ah["kwh_ours"] * ah["our_ex_vat"]).sum()
        sup_wholesale_p = (ah["kwh_ours"] * ah["wholesale_price"]).sum()
        sup_network_p   = (ah["kwh_ours"] * (ah["duos_charge"] + TNUOS_RATE + BSUOS_BUFFER)).sum()

        sup_rev_annual       = sup_rev_p       * scale / 100
        sup_wholesale_annual = sup_wholesale_p * scale / 100
        sup_network_annual   = sup_network_p   * scale / 100
        sup_profit_annual    = sup_rev_annual - sup_wholesale_annual - sup_network_annual

        # ── Agile reseller equivalent ─────────────────────────────────────────
        agile_rev_p       = (ah["kwh_agile"] * ah["price_ex_vat"]).sum()
        agile_wholesale_p = (ah["kwh_agile"] * ah["wholesale_price"]).sum()
        agile_network_p   = (ah["kwh_agile"] * (ah["agile_duos_charge"] + TNUOS_RATE + BSUOS_BUFFER)).sum()

        agile_rev_annual       = agile_rev_p       * scale / 100
        agile_wholesale_annual = agile_wholesale_p * scale / 100
        agile_network_annual   = agile_network_p   * scale / 100
        agile_profit_annual    = agile_rev_annual - agile_wholesale_annual - agile_network_annual

        rows.append({
            "scenario":                        scenario_id,
            "label":                           label,
            "shift_frac":                      shift_frac,
            "ev_kwh_per_day":                  ev_kwh,
            "multiplier":                      multiplier,
            "annual_kwh":                      annual_kwh_scenario,
            # Customer — wholesale + network only (energy component)
            "cust_bill_ours_annual_gbp":        round(cust_ours_annual,  2),
            "cust_bill_agile_annual_gbp":       round(cust_agile_annual, 2),
            "cust_saving_vs_agile_gbp":         round(cust_agile_annual - cust_ours_annual, 2),
            # Customer — all-in (inc standing charge, levies, opex)
            "cust_allin_ours_annual_gbp":       round(cust_allin_ours_annual, 2),
            "cust_allin_agile_annual_gbp":      round(cust_allin_agile_annual, 2),
            "standing_charge_annual_gbp":       round(standing_annual, 2),
            "levy_annual_gbp":                  round(levy_annual, 2),
            "opex_annual_gbp":                  round(opex_annual, 2),
            "ofgem_cap_annual_gbp":             round(ofgem_cap_annual, 2),
            # Effective unit rates (p/kWh = annual bill × 100 / annual kWh)
            "effective_p_kwh_ours":             round(cust_ours_annual  * 100 / annual_kwh_scenario, 2),
            "effective_p_kwh_agile":            round(cust_agile_annual * 100 / annual_kwh_scenario, 2),
            "effective_allin_p_kwh_ours":       round(cust_allin_ours_annual * 100 / annual_kwh_scenario, 2),
            "effective_allin_p_kwh_agile":      round(cust_allin_agile_annual * 100 / annual_kwh_scenario, 2),
            # Supplier — our tariff
            "sup_revenue_annual_gbp":           round(sup_rev_annual, 2),
            "sup_wholesale_cost_annual_gbp":    round(sup_wholesale_annual, 2),
            "sup_network_cost_annual_gbp":      round(sup_network_annual, 2),
            "sup_profit_annual_gbp":            round(sup_profit_annual, 2),
            # Agile reseller
            "agile_revenue_annual_gbp":         round(agile_rev_annual, 2),
            "agile_wholesale_cost_annual_gbp":  round(agile_wholesale_annual, 2),
            "agile_network_cost_annual_gbp":    round(agile_network_annual, 2),
            "agile_profit_annual_gbp":          round(agile_profit_annual, 2),
        })

    return pd.DataFrame(rows)
