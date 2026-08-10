"""Explicit chronological train, validation, and test partitioning."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from flight_delay.data.preprocessing import DataQualityError


@dataclass(frozen=True)
class TemporalSplit:
    """Non-overlapping chronological partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _as_timestamp(value: str | date | datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise DataQualityError(f"invalid {name}: {value!r}") from error
    if timestamp.tz is not None:
        raise DataQualityError(f"{name} must be timezone-naive")
    return timestamp.normalize()


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_start: str | date | datetime | pd.Timestamp,
    validation_start: str | date | datetime | pd.Timestamp,
    test_start: str | date | datetime | pd.Timestamp,
    test_end: str | date | datetime | pd.Timestamp,
    date_column: str = "FlightDate",
) -> TemporalSplit:
    """Split rows into explicit half-open intervals without future-to-past leakage."""

    if date_column not in frame:
        raise DataQualityError(f"missing date column: {date_column}")
    boundaries = (
        _as_timestamp(train_start, "train_start"),
        _as_timestamp(validation_start, "validation_start"),
        _as_timestamp(test_start, "test_start"),
        _as_timestamp(test_end, "test_end"),
    )
    if not all(left < right for left, right in zip(boundaries, boundaries[1:])):
        raise DataQualityError("split boundaries must be strictly increasing")

    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        bad_rows = list(frame.index[dates.isna()])[:10]
        raise DataQualityError(f"invalid {date_column} values at rows {bad_rows}")

    train_start_ts, validation_start_ts, test_start_ts, test_end_ts = boundaries
    train = frame.loc[dates.ge(train_start_ts) & dates.lt(validation_start_ts)].copy()
    validation = frame.loc[dates.ge(validation_start_ts) & dates.lt(test_start_ts)].copy()
    test = frame.loc[dates.ge(test_start_ts) & dates.lt(test_end_ts)].copy()

    if not train.empty and not validation.empty:
        if dates.loc[train.index].max() >= dates.loc[validation.index].min():
            raise DataQualityError("train and validation chronology overlaps")
    if not validation.empty and not test.empty:
        if dates.loc[validation.index].max() >= dates.loc[test.index].min():
            raise DataQualityError("validation and test chronology overlaps")
    return TemporalSplit(train=train, validation=validation, test=test)
