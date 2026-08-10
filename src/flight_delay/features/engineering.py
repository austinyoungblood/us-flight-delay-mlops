"""Pre-departure schedule feature engineering."""

from math import cos, pi, sin

import pandas as pd

from flight_delay.data.preprocessing import DataQualityError, normalize_bts_columns, parse_crs_time

_REQUIRED_COLUMNS = frozenset({"Origin", "Dest", "DayOfWeek", "CRSDepTime", "CRSArrTime"})


def _time_parts(value: object, bucket_minutes: int) -> tuple[int, int, float, float]:
    parsed = parse_crs_time(value)
    minutes_since_midnight = parsed.hour * 60 + parsed.minute
    angle = 2 * pi * minutes_since_midnight / (24 * 60)
    return (
        parsed.hour,
        (parsed.minute // bucket_minutes) * bucket_minutes,
        sin(angle),
        cos(angle),
    )


def derive_schedule_features(frame: pd.DataFrame, *, bucket_minutes: int = 15) -> pd.DataFrame:
    """Derive route, scheduled time buckets, weekend, and cyclical encodings."""

    if bucket_minutes <= 0 or 60 % bucket_minutes:
        raise DataQualityError("bucket_minutes must be a positive divisor of 60")
    result = normalize_bts_columns(frame, required_columns=_REQUIRED_COLUMNS)

    day_of_week = pd.to_numeric(result["DayOfWeek"], errors="coerce")
    invalid_days = result.index[day_of_week.isna() | ~day_of_week.between(1, 7)]
    if len(invalid_days):
        raise DataQualityError(f"DayOfWeek must be in 1..7 at rows {list(invalid_days)[:10]}")

    result["route"] = (
        result["Origin"].astype(str).str.upper() + "-" + result["Dest"].astype(str).str.upper()
    )
    departure_parts = result["CRSDepTime"].map(lambda value: _time_parts(value, bucket_minutes))
    arrival_parts = result["CRSArrTime"].map(lambda value: _time_parts(value, bucket_minutes))

    result["scheduled_departure_hour"] = departure_parts.map(lambda parts: parts[0])
    result["scheduled_departure_minute_bucket"] = departure_parts.map(lambda parts: parts[1])
    result["scheduled_departure_sin"] = departure_parts.map(lambda parts: parts[2])
    result["scheduled_departure_cos"] = departure_parts.map(lambda parts: parts[3])
    result["scheduled_arrival_hour"] = arrival_parts.map(lambda parts: parts[0])
    result["scheduled_arrival_minute_bucket"] = arrival_parts.map(lambda parts: parts[1])
    result["scheduled_arrival_sin"] = arrival_parts.map(lambda parts: parts[2])
    result["scheduled_arrival_cos"] = arrival_parts.map(lambda parts: parts[3])
    result["is_weekend"] = day_of_week.isin({6, 7}).astype("int8")
    return result
