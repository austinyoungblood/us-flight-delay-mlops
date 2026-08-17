"""Shared synthetic-only fixtures for governed v2 tests."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.modeling.v2.protocol import load_and_validate_v2_protocol


def make_v2_frame(months: range | tuple[int, ...] = range(1, 13)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in months:
        for index, day in enumerate((2, 9, 18, 25)):
            timestamp = pd.Timestamp(year=2025, month=month, day=day)
            departure_hour = 7 + index * 3
            arrival_hour = departure_hour + 2
            origin = "DEN" if index % 2 == 0 else "SFO"
            destination = "ORD" if index % 2 == 0 else "DEN"
            departure_minutes = departure_hour * 60
            arrival_minutes = arrival_hour * 60
            rows.append(
                {
                    "flight_date": timestamp,
                    "Month": month,
                    "DayofMonth": day,
                    "DayOfWeek": timestamp.dayofweek + 1,
                    "Reporting_Airline": "UA" if index % 2 == 0 else "AA",
                    "Origin": origin,
                    "Dest": destination,
                    "CRSDepTime": departure_hour * 100,
                    "CRSArrTime": arrival_hour * 100,
                    "CRSElapsedTime": 120,
                    "Distance": 850 + index * 50,
                    "route": f"{origin}-{destination}",
                    "scheduled_departure_hour": departure_hour,
                    "scheduled_arrival_hour": arrival_hour,
                    "scheduled_departure_minute_bucket": 0,
                    "scheduled_arrival_minute_bucket": 0,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "scheduled_departure_sin": math.sin(2 * math.pi * departure_minutes / 1440),
                    "scheduled_departure_cos": math.cos(2 * math.pi * departure_minutes / 1440),
                    "scheduled_arrival_sin": math.sin(2 * math.pi * arrival_minutes / 1440),
                    "scheduled_arrival_cos": math.cos(2 * math.pi * arrival_minutes / 1440),
                    "target": int(index % 2 == 1),
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def make_v3_frame(
    *,
    start: str = "2024-01-01",
    end: str = "2025-11-30",
    per_day: int = 4,
) -> pd.DataFrame:
    """Build a synthetic v3 frame spanning both calendar years so seasonal state has prior years."""

    carriers = ("UA", "AA", "DL", "WN")
    airports = ("DEN", "SFO", "ORD", "JFK")
    rows: list[dict[str, object]] = []
    for offset, timestamp in enumerate(pd.date_range(start, end, freq="D")):
        for index in range(per_day):
            departure_hour = (7 + index * 4) % 24
            arrival_hour = (departure_hour + 2) % 24
            origin = airports[(offset + index) % len(airports)]
            destination = airports[(offset + index + 1) % len(airports)]
            departure_minutes = departure_hour * 60
            arrival_minutes = arrival_hour * 60
            rows.append(
                {
                    "flight_date": timestamp,
                    "Month": timestamp.month,
                    "DayofMonth": timestamp.day,
                    "DayOfWeek": timestamp.dayofweek + 1,
                    "Reporting_Airline": carriers[(offset + index) % len(carriers)],
                    "Origin": origin,
                    "Dest": destination,
                    "CRSDepTime": departure_hour * 100,
                    "CRSArrTime": arrival_hour * 100,
                    "CRSElapsedTime": 120,
                    "Distance": 850 + index * 50,
                    "route": f"{origin}-{destination}",
                    "scheduled_departure_hour": departure_hour,
                    "scheduled_arrival_hour": arrival_hour,
                    "scheduled_departure_minute_bucket": 0,
                    "scheduled_arrival_minute_bucket": 0,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "scheduled_departure_sin": math.sin(2 * math.pi * departure_minutes / 1440),
                    "scheduled_departure_cos": math.cos(2 * math.pi * departure_minutes / 1440),
                    "scheduled_arrival_sin": math.sin(2 * math.pi * arrival_minutes / 1440),
                    "scheduled_arrival_cos": math.cos(2 * math.pi * arrival_minutes / 1440),
                    # A November-heavy target so seasonal tables carry real signal.
                    "target": int((index + offset + (3 if timestamp.month == 11 else 0)) % 3 == 0),
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


@pytest.fixture
def synthetic_v2_frame() -> pd.DataFrame:
    return make_v2_frame()


@pytest.fixture
def synthetic_v3_frame() -> pd.DataFrame:
    return make_v3_frame()


@pytest.fixture
def v3_protocol() -> dict[str, object]:
    from flight_delay.modeling.v3.protocol import load_and_validate_v3_protocol

    root = Path(__file__).resolve().parents[1]
    protocol, _lock, _sha = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
        repository_root=root,
    )
    return protocol


@pytest.fixture
def v2_protocol() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    protocol, _lock, _sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    return protocol
