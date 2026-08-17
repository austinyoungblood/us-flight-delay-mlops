"""Deterministic seasonal features: holiday algebra, bounded distances, and batch parity."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v3.protocol import (
    CHRISTMAS_WINDOW_RANGE,
    DAY_DISTANCE_CLIP,
    DETERMINISTIC_SEASONAL_FEATURES,
    THANKSGIVING_WINDOW_RANGE,
)
from flight_delay.modeling.v3.seasonal import (
    V3SeasonalError,
    christmas,
    derive_seasonal_features,
    seasonal_features_for_date,
    thanksgiving,
)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2023, date(2023, 11, 23)),
        (2024, date(2024, 11, 28)),
        (2025, date(2025, 11, 27)),
        (2026, date(2026, 11, 26)),
        (2027, date(2027, 11, 25)),
    ],
)
def test_thanksgiving_is_the_fourth_thursday_of_november(year: int, expected: date) -> None:
    observed = thanksgiving(year)
    assert observed == expected
    assert observed.weekday() == 3
    assert observed.month == 11
    # Exactly three Thursdays precede it inside November.
    earlier = [day for day in range(1, observed.day) if date(year, 11, day).weekday() == 3]
    assert len(earlier) == 3


def test_christmas_is_december_25() -> None:
    assert christmas(2025) == date(2025, 12, 25)


def test_distances_are_zero_on_the_holiday_itself() -> None:
    features = seasonal_features_for_date(date(2025, 11, 27))
    assert features["days_to_thanksgiving"] == 0
    assert features["is_thanksgiving_window"] == 1
    assert seasonal_features_for_date(date(2025, 12, 25))["days_to_christmas"] == 0


def test_sign_convention_is_positive_when_the_holiday_is_ahead() -> None:
    before = seasonal_features_for_date(date(2025, 11, 20))
    after = seasonal_features_for_date(date(2025, 12, 4))
    assert before["days_to_thanksgiving"] == 7
    assert after["days_to_thanksgiving"] == -7


def test_distances_are_clipped_to_the_declared_bound() -> None:
    lower, upper = DAY_DISTANCE_CLIP
    midsummer = seasonal_features_for_date(date(2025, 7, 4))
    assert midsummer["days_to_thanksgiving"] == upper
    assert midsummer["days_to_christmas"] == upper
    for offset in range(0, 400, 7):
        features = seasonal_features_for_date(date(2024, 1, 1) + timedelta(days=offset))
        assert lower <= features["days_to_thanksgiving"] <= upper
        assert lower <= features["days_to_christmas"] <= upper


def test_thanksgiving_window_spans_tuesday_before_through_monday_after() -> None:
    anchor = thanksgiving(2025)
    low, high = THANKSGIVING_WINDOW_RANGE
    inside = [anchor - timedelta(days=delta) for delta in range(low, high + 1)]
    for day in inside:
        assert seasonal_features_for_date(day)["is_thanksgiving_window"] == 1
    assert (
        seasonal_features_for_date(anchor - timedelta(days=high + 1))["is_thanksgiving_window"] == 0
    )
    assert (
        seasonal_features_for_date(anchor - timedelta(days=low - 1))["is_thanksgiving_window"] == 0
    )
    assert len(inside) == 7


def test_christmas_window_spans_december_21_through_january_4() -> None:
    low, high = CHRISTMAS_WINDOW_RANGE
    assert seasonal_features_for_date(date(2025, 12, 21))["is_christmas_window"] == 1
    assert seasonal_features_for_date(date(2026, 1, 4))["is_christmas_window"] == 1
    assert seasonal_features_for_date(date(2025, 12, 20))["is_christmas_window"] == 0
    assert seasonal_features_for_date(date(2026, 1, 5))["is_christmas_window"] == 0
    assert high - low + 1 == 15


def test_new_year_rollover_selects_the_prior_year_anchor() -> None:
    features = seasonal_features_for_date(date(2026, 1, 2))
    assert features["days_to_christmas"] == -8
    assert features["is_christmas_window"] == 1


def test_day_of_year_encoding_is_cyclical_and_leap_year_aware() -> None:
    first = seasonal_features_for_date(date(2025, 1, 1))
    assert first["day_of_year_sin"] == pytest.approx(0.0, abs=1e-12)
    assert first["day_of_year_cos"] == pytest.approx(1.0, abs=1e-12)
    leap = seasonal_features_for_date(date(2024, 12, 31))
    common = seasonal_features_for_date(date(2025, 12, 31))
    # Both land one step short of a full revolution, on 366- and 365-day circles respectively.
    assert leap["day_of_year_sin"] == pytest.approx(np.sin(2 * np.pi * 365 / 366), abs=1e-12)
    assert common["day_of_year_sin"] == pytest.approx(np.sin(2 * np.pi * 364 / 365), abs=1e-12)
    for features in (first, leap, common):
        assert features["day_of_year_sin"] ** 2 + features["day_of_year_cos"] ** 2 == pytest.approx(
            1.0
        )


def test_batch_and_single_row_paths_agree_exactly() -> None:
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(0, 1096)]
    batch = derive_seasonal_features(pd.Series(pd.to_datetime(days)))
    assert tuple(batch.columns) == DETERMINISTIC_SEASONAL_FEATURES
    for position, day in enumerate(days):
        single = seasonal_features_for_date(day)
        for name, value in single.items():
            assert batch.iloc[position][name] == pytest.approx(value, abs=1e-12), (day, name)


def test_features_depend_only_on_the_date() -> None:
    assert seasonal_features_for_date(date(2025, 11, 27)) == seasonal_features_for_date(
        date(2025, 11, 27)
    )


def test_batch_rejects_missing_or_empty_dates() -> None:
    with pytest.raises(V3SeasonalError):
        derive_seasonal_features(pd.Series([], dtype="datetime64[ns]"))
    with pytest.raises(V3SeasonalError):
        derive_seasonal_features(pd.Series([pd.NaT, pd.Timestamp("2025-01-01")]))


def test_scalar_path_rejects_a_non_date() -> None:
    with pytest.raises(V3SeasonalError):
        seasonal_features_for_date("2025-11-27")  # type: ignore[arg-type]
