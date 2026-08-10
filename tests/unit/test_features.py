import math

import pandas as pd
import pytest

from flight_delay.features.engineering import derive_schedule_features
from flight_delay.features.leakage import (
    ALLOWED_MODEL_FEATURES,
    FeatureLeakageError,
    validate_model_features,
)


def test_derived_schedule_features() -> None:
    frame = pd.DataFrame(
        {
            "Origin": ["den", "LAX"],
            "Dest": ["JFK", "SEA"],
            "DayOfWeek": [6, 3],
            "CRSDepTime": [0, 1347],
            "CRSArrTime": [600, 1559],
        }
    )
    result = derive_schedule_features(frame)
    assert result["route"].tolist() == ["DEN-JFK", "LAX-SEA"]
    assert result["scheduled_departure_hour"].tolist() == [0, 13]
    assert result["scheduled_departure_minute_bucket"].tolist() == [0, 45]
    assert result["scheduled_arrival_minute_bucket"].tolist() == [0, 45]
    assert result["is_weekend"].tolist() == [1, 0]
    assert result.loc[0, "scheduled_departure_sin"] == pytest.approx(0.0)
    assert result.loc[0, "scheduled_departure_cos"] == pytest.approx(1.0)
    assert math.isfinite(result.loc[1, "scheduled_arrival_sin"])


def test_forbidden_feature_rejection_is_alias_safe() -> None:
    with pytest.raises(FeatureLeakageError) as error:
        validate_model_features(["Origin", "dep_delay"])
    assert error.value.forbidden == frozenset({"DepDelay"})


def test_unapproved_feature_rejection() -> None:
    with pytest.raises(FeatureLeakageError) as error:
        validate_model_features(["Origin", "LiveWeather"])
    assert error.value.unapproved == frozenset({"LiveWeather"})


def test_allowlisted_schema_is_returned_unchanged() -> None:
    schema = {"Origin", "Dest", "route", "scheduled_departure_sin"}
    assert validate_model_features(schema) == frozenset(schema)
    assert "ArrDelay" not in ALLOWED_MODEL_FEATURES
