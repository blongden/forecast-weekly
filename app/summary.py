"""
Generate a natural-language week-ahead price summary using an LLM.

Produces a one-liner per day plus an overall week summary, suitable for
display on the dashboard or consumption by other applications.
"""
import json
import os
from datetime import date

import pandas as pd

_MODEL = "gpt-4.1-mini"


def generate_week_summary(
    predictions: pd.DataFrame,
    hh_pred: pd.DataFrame,
    hist_mean: float,
) -> dict | None:
    """
    Call OpenAI to produce a week-ahead summary.

    Returns dict with keys:
        week_summary: str   — 1-2 sentence overview of the week
        days: list[dict]    — [{date, summary}, ...] one-liner per day

    Returns None if OPENAI_API_KEY is not set or the call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return None

    # Build context for the LLM
    daily_context = []
    for _, row in predictions.iterrows():
        d = row["date"]
        d_str = d.strftime("%A %d %b") if hasattr(d, "strftime") else str(d)
        entry = {"date": d_str, "epex_p_kwh": round(row["predicted_epex_p_kwh"], 2)}
        if "pred_q10" in row and pd.notna(row.get("pred_q10")):
            entry["q10"] = round(row["pred_q10"], 2)
            entry["q90"] = round(row["pred_q90"], 2)
        daily_context.append(entry)

    # HH peak/off-peak breakdown per day
    if not hh_pred.empty and "datetime_local" in hh_pred.columns:
        hh = hh_pred.copy()
        hh["_date"] = hh["datetime_local"].dt.date
        hh["_hour"] = hh["datetime_local"].dt.hour
        for entry in daily_context:
            d_str = entry["date"]
            for d_val in hh["_date"].unique():
                if pd.Timestamp(d_val).strftime("%A %d %b") == d_str:
                    day_slots = hh[hh["_date"] == d_val]
                    peak = day_slots[day_slots["_hour"].between(16, 18)]
                    offpeak = day_slots[~day_slots["_hour"].between(16, 18)]
                    if not peak.empty:
                        entry["peak_p_kwh"] = round(peak["predicted_epex_p_kwh"].mean(), 2)
                    if not offpeak.empty:
                        entry["offpeak_p_kwh"] = round(offpeak["predicted_epex_p_kwh"].mean(), 2)
                    break

    prompt = f"""You are an energy market analyst writing a concise week-ahead electricity price summary for UK consumers and businesses.

Context:
- These are EPEX SPOT GB day-ahead wholesale electricity prices in p/kWh (pence per kilowatt-hour)
- The 12-month historical average is {hist_mean:.1f}p/kWh
- Peak hours are 16:00-19:00, off-peak is the rest of the day
- Prices are driven by wind/solar generation, gas prices, demand, and interconnector flows

Forecast data:
{json.dumps(daily_context, indent=2)}

Write a JSON response with:
1. "week_summary": A 1-2 sentence overview of the week ahead (trends, notable days, comparison to historical average). Be specific with numbers.
2. "days": An array of objects, one per forecast day, each with "date" (matching the input) and "summary" (a single punchy sentence about that day's price — mention if it's cheap/expensive, peak vs off-peak spread, good for shifting load, etc.)

Keep language accessible — avoid jargon. Be direct and useful. Output valid JSON only."""

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[Summary] LLM call failed: {e}")
        return None
