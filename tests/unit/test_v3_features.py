"""Seasonal historical state: prior-year semantics, parity, and leakage refusal."""

from __future__ import annotations

import pandas as pd
import pytest

from flight_delay.modeling.v3.features import (
    SEASONAL_TABLE_COLUMNS,
    V3FeatureError,
    V3HistoricalState,
    build_v3_historical_state,
    transform_one_v3,
    transform_v3_training_rows,
    transform_with_v3_state,
)
from flight_delay.modeling.v3.protocol import (
    SEASONAL_HISTORICAL_FEATURES,
    V3_FEATURES,
    V3_HISTORICAL_FEATURES,
)
from tests.conftest import make_v3_frame


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return make_v3_frame(start="2024-01-01", end="2025-10-31")


@pytest.fixture(scope="module")
def november() -> pd.DataFrame:
    return make_v3_frame(start="2025-11-01", end="2025-11-30")


@pytest.fixture(scope="module")
def november_state(history: pd.DataFrame) -> V3HistoricalState:
    return build_v3_historical_state(history, as_of="2025-10-31")


def test_state_exposes_every_seasonal_table(november_state: V3HistoricalState) -> None:
    assert set(november_state.seasonal_tables) == set(SEASONAL_TABLE_COLUMNS)
    assert november_state.as_of.isoformat() == "2025-10-31"
    assert november_state.prior_strength == 50


def test_november_seasonal_state_comes_only_from_november_2024(
    november_state: V3HistoricalState,
) -> None:
    """The locked requirement: November 2024 may contribute, November 2025 may not."""

    assert november_state.same_calendar_month_max_year[11] == 2024
    assert november_state.same_calendar_month_max_year[12] == 2024
    # Months already observed in 2025 legitimately carry a 2025 max.
    assert november_state.same_calendar_month_max_year[10] == 2025


def test_transform_produces_the_full_forty_eight_feature_schema(
    november: pd.DataFrame, november_state: V3HistoricalState
) -> None:
    transformed = transform_with_v3_state(november, november_state)
    assert tuple(transformed.columns) == V3_FEATURES
    assert len(transformed) == len(november)
    assert transformed[list(V3_HISTORICAL_FEATURES)].notna().all().all()


def test_batch_transform_matches_single_row_serving_exactly(
    november: pd.DataFrame, november_state: V3HistoricalState
) -> None:
    batch = transform_with_v3_state(november, november_state)
    sample = november.sample(n=40, random_state=11)
    for index, row in sample.iterrows():
        single = transform_one_v3(row, november_state)
        for name in V3_FEATURES[20:]:
            assert single[name] == pytest.approx(batch.loc[index, name], abs=1e-12), name


def test_seasonal_rates_are_smoothed_probabilities(
    november: pd.DataFrame, november_state: V3HistoricalState
) -> None:
    transformed = transform_with_v3_state(november, november_state)
    seasonal = transformed[list(SEASONAL_HISTORICAL_FEATURES)]
    assert ((seasonal > 0.0) & (seasonal < 1.0)).all().all()


def test_unseen_categories_fall_back_to_the_global_rate(
    november: pd.DataFrame, november_state: V3HistoricalState
) -> None:
    unseen = november.head(1).copy()
    unseen.loc[:, "Reporting_Airline"] = "ZZ"
    unseen.loc[:, "Origin"] = "ZZZ"
    unseen.loc[:, "Dest"] = "YYY"
    unseen.loc[:, "route"] = "ZZZ-YYY"
    transformed = transform_with_v3_state(unseen, november_state)
    for name in SEASONAL_HISTORICAL_FEATURES:
        if name != "prior_same_calendar_month_global_delay_rate":
            assert transformed.iloc[0][name] == pytest.approx(november_state.global_rate)


def test_state_refuses_labels_after_its_cutoff(history: pd.DataFrame) -> None:
    with pytest.raises(V3FeatureError):
        build_v3_historical_state(history, as_of="2024-06-30")


def test_transform_refuses_a_state_that_reaches_into_the_model_month(
    november: pd.DataFrame, history: pd.DataFrame
) -> None:
    contaminated = pd.concat([history, november], ignore_index=True)
    state = build_v3_historical_state(contaminated, as_of="2025-11-30")
    with pytest.raises(V3FeatureError, match="must end before every model-row month"):
        transform_with_v3_state(november, state)


def test_single_row_path_refuses_same_year_seasonal_contribution(
    november: pd.DataFrame, history: pd.DataFrame
) -> None:
    contaminated = pd.concat([history, november], ignore_index=True)
    state = build_v3_historical_state(contaminated, as_of="2025-11-30")
    assert state.same_calendar_month_max_year[11] == 2025
    with pytest.raises(V3FeatureError, match="own year"):
        transform_one_v3(november.iloc[0], state)


def test_state_serialization_round_trips_to_the_same_digest(
    november_state: V3HistoricalState,
) -> None:
    restored = V3HistoricalState.from_bytes(november_state.to_bytes())
    assert restored.sha256 == november_state.sha256
    assert restored.same_calendar_month_max_year == november_state.same_calendar_month_max_year
    assert restored.as_of == november_state.as_of


def test_state_digest_is_stable_across_rebuilds(history: pd.DataFrame) -> None:
    first = build_v3_historical_state(history, as_of="2025-10-31")
    second = build_v3_historical_state(history.sample(frac=1.0, random_state=5), as_of="2025-10-31")
    assert first.sha256 == second.sha256


def test_state_schema_digest_covers_the_v3_feature_list(
    november_state: V3HistoricalState,
) -> None:
    from flight_delay.modeling.v3.protocol import canonical_sha256

    assert november_state.schema_sha256 == canonical_sha256(list(V3_FEATURES))


@pytest.mark.parametrize(
    "corruption",
    [
        b"not json",
        b'{"schema": "wrong"}',
        b'{"schema": "flight-delay-historical-state-v3"}',
    ],
)
def test_corrupt_state_bytes_are_refused(corruption: bytes) -> None:
    with pytest.raises(V3FeatureError):
        V3HistoricalState.from_bytes(corruption)


def test_training_transform_walks_month_by_month(history: pd.DataFrame) -> None:
    model_rows = history.loc[pd.to_datetime(history["flight_date"]).ge("2025-08-01")].copy()
    transform = transform_v3_training_rows(history, model_rows)
    assert tuple(transform.features.columns) == V3_FEATURES
    assert len(transform.features) == len(model_rows)
    assert sorted(transform.monthly_state_sha256) == ["2025-08", "2025-09", "2025-10"]
    assert len(set(transform.monthly_state_sha256.values())) == 3


def test_training_transform_rejects_january_2024_burn_in_rows(history: pd.DataFrame) -> None:
    burn_in = history.loc[pd.to_datetime(history["flight_date"]).lt("2024-02-01")].copy()
    with pytest.raises(V3FeatureError, match="burn-in"):
        transform_v3_training_rows(history, burn_in)


def test_training_transform_stops_before_december(
    history: pd.DataFrame, november: pd.DataFrame
) -> None:
    december = make_v3_frame(start="2025-12-01", end="2025-12-05")
    rows = pd.concat([november, december], ignore_index=True)
    everything = pd.concat([history, november], ignore_index=True)
    with pytest.raises(V3FeatureError, match="before December"):
        transform_v3_training_rows(everything, rows)
