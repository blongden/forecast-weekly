"""
Export half-hourly price forecasts to DynamoDB PriceForecast table.

This bridges the energy_analysis (Python/Fargate) and ev-smart (Go/Lambda)
systems via a shared DynamoDB table, enabling the charging optimizer to
schedule EV charging during the cheapest half-hour slots.
"""
import time
from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo

import boto3
import pandas as pd

TABLE_NAME = "PriceForecast"
TTL_DAYS = 30
LOCAL_TZ = ZoneInfo("Europe/London")


def export_halfhourly_forecast(hh_pred: pd.DataFrame, forecast_date: date) -> int:
    """
    Write half-hourly price predictions to the PriceForecast DynamoDB table.

    Parameters
    ----------
    hh_pred : DataFrame
        Output of predict_halfhourly_ensemble(). Must contain columns:
        datetime_local, predicted_epex_p_kwh, pred_q10, pred_q90.
    forecast_date : date
        The date the forecast was produced (partition key).

    Returns
    -------
    int
        Number of items written.
    """
    if hh_pred.empty:
        return 0

    dynamodb = boto3.resource("dynamodb", region_name="eu-west-2")
    table = dynamodb.Table(TABLE_NAME)

    forecast_date_str = str(forecast_date)
    ttl_epoch = int(time.time()) + TTL_DAYS * 86400
    count = 0

    with table.batch_writer() as batch:
        for _, row in hh_pred.iterrows():
            dt = pd.Timestamp(row["datetime_local"])
            # Localize if naive (Open-Meteo returns local time without tz info)
            if dt.tzinfo is None:
                dt = dt.tz_localize(LOCAL_TZ)
            # Convert local time to UTC ISO-8601 for SlotTime (matches Go consumer)
            slot_time = dt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

            item = {
                "ForecastDate": forecast_date_str,
                "SlotTime": slot_time,
                "PriceGBP": Decimal(str(round(row["predicted_epex_p_kwh"], 4))),
                "TTL": ttl_epoch,
            }

            if "pred_q10" in row and pd.notna(row["pred_q10"]):
                item["PriceQ10"] = Decimal(str(round(row["pred_q10"], 4)))
            if "pred_q90" in row and pd.notna(row["pred_q90"]):
                item["PriceQ90"] = Decimal(str(round(row["pred_q90"], 4)))

            batch.put_item(Item=item)
            count += 1

    return count
