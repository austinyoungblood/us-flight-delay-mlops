from __future__ import annotations

import json

import pandas as pd
import pytest

from flight_delay.modeling.v2.features import (
    PRIOR_STRENGTH,
    HistoricalState,
    V2FeatureError,
    build_historical_state,
    transform_one,
    transform_training_rows,
    transform_with_state,
)
from flight_delay.modeling.v2.protocol import HISTORICAL_FEATURES, V2_FEATURES


def _month(frame: pd.DataFrame, number: int) -> pd.DataFrame:
    return frame.loc[pd.to_datetime(frame["flight_date"]).dt.month.eq(number)].copy()


def test_state_is_deterministic_hashable_and_round_trips(synthetic_v2_frame: pd.DataFrame) -> None:
    history = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(5)
    ]
    state = build_historical_state(history, as_of="2025-05-31")
    repeated = build_historical_state(history.sample(frac=1.0, random_state=7), as_of="2025-05-31")

    assert state.to_bytes() == repeated.to_bytes()
    assert state.sha256 == repeated.sha256
    assert state.global_counts.count == 20
    assert state.recent_global_counts.count == 12
    restored = HistoricalState.from_bytes(state.to_bytes())
    assert restored == state
    assert restored.sha256 == state.sha256


def test_empirical_bayes_smoothing_support_and_unseen_fallback(
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    january = _month(synthetic_v2_frame, 1)
    january.loc[january["Reporting_Airline"].eq("UA"), "target"] = 1
    january.loc[january["Reporting_Airline"].eq("AA"), "target"] = 0
    state = build_historical_state(january, as_of="2025-01-31")
    known = transform_one(_month(synthetic_v2_frame, 2).iloc[0], state)
    expected = (2 + PRIOR_STRENGTH * 0.5) / (2 + PRIOR_STRENGTH)
    assert known["prior_carrier_delay_rate"] == pytest.approx(expected)
    assert known["log_route_support"] == pytest.approx(__import__("math").log1p(2))

    unseen = _month(synthetic_v2_frame, 2).iloc[0].copy()
    unseen["Reporting_Airline"] = "ZZ"
    unseen["Origin"] = "SEA"
    unseen["Dest"] = "BOS"
    unseen["route"] = "SEA-BOS"
    transformed = transform_one(unseen, state)
    for name in (
        "prior_carrier_delay_rate",
        "prior_origin_delay_rate",
        "prior_destination_delay_rate",
        "prior_route_delay_rate",
        "recent_route_delay_rate_3m",
    ):
        assert transformed[name] == pytest.approx(state.global_rate)
    assert transformed["log_route_support"] == 0.0
    assert transformed["log_carrier_route_support"] == 0.0


def test_training_uses_strict_prior_month_and_january_burn_in(
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    through_april = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(4)
    ]
    model_rows = through_april.loc[pd.to_datetime(through_april["flight_date"]).dt.month.ge(2)]
    transformed = transform_training_rows(through_april, model_rows)

    assert tuple(transformed.features.columns) == V2_FEATURES
    assert set(transformed.monthly_state_sha256) == {"2025-02", "2025-03", "2025-04"}
    february_state = build_historical_state(_month(through_april, 1), as_of="2025-01-31")
    expected = transform_one(_month(model_rows, 2).iloc[0], february_state)
    actual = transformed.features.loc[_month(model_rows, 2).index[0], list(HISTORICAL_FEATURES)]
    assert actual.to_dict() == pytest.approx(expected)

    with pytest.raises(V2FeatureError, match="burn-in"):
        transform_training_rows(through_april, _month(through_april, 1))


def test_same_month_and_future_label_changes_cannot_affect_features(
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    history = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(4)
    ].copy()
    february = _month(history, 2)
    baseline = transform_training_rows(history, february).features
    mutated = history.copy()
    mutated.loc[pd.to_datetime(mutated["flight_date"]).dt.month.ge(2), "target"] = (
        1 - mutated["target"]
    )
    observed = transform_training_rows(mutated, february).features
    pd.testing.assert_frame_equal(baseline, observed)


def test_november_and_december_reuse_the_october_state(
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    through_october = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(10)
    ]
    state = build_historical_state(through_october, as_of="2025-10-31")
    november = _month(synthetic_v2_frame, 11).iloc[[0]].copy()
    december = _month(synthetic_v2_frame, 12).iloc[[0]].copy()
    december.index = november.index
    november_features = transform_with_state(november, state)
    december_features = transform_with_state(december, state)
    pd.testing.assert_frame_equal(
        november_features.loc[:, HISTORICAL_FEATURES],
        december_features.loc[:, HISTORICAL_FEATURES],
    )


def test_state_rejects_same_or_future_labels_and_invalid_artifacts(
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    january_february = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(2)
    ]
    with pytest.raises(V2FeatureError, match="after the state cutoff"):
        build_historical_state(january_february, as_of="2025-01-31")

    state = build_historical_state(_month(synthetic_v2_frame, 1), as_of="2025-01-31")
    with pytest.raises(V2FeatureError, match="end before"):
        transform_with_state(_month(synthetic_v2_frame, 1), state)
    with pytest.raises(V2FeatureError, match="valid JSON"):
        HistoricalState.from_bytes(b"not-json")
    payload = state.as_dict()
    payload["schema"] = "wrong"
    with pytest.raises(V2FeatureError, match="schema mismatch"):
        HistoricalState.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Reporting_Airline", "", "non-empty"),
        ("scheduled_departure_hour", 25, "hour"),
        ("target", 2, "binary"),
    ],
)
def test_state_input_validation(
    synthetic_v2_frame: pd.DataFrame, column: str, value: object, message: str
) -> None:
    january = _month(synthetic_v2_frame, 1)
    january.loc[january.index[0], column] = value
    with pytest.raises(V2FeatureError, match=message):
        build_historical_state(january, as_of="2025-01-31")
