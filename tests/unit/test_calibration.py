from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from flight_delay.modeling.calibration import (
    calibration_audit,
    calibration_table,
    fit_sigmoid_calibrator,
    mean_probability_gap,
    partition_development_data,
    reliability_table,
)


def _month(start: str, periods: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_date": pd.date_range(start, periods=periods),
            "x": range(periods),
            "target": [0, 1, 0],
        }
    )


def test_development_partitions_are_exact_and_disjoint() -> None:
    train = pd.concat(
        [_month("2025-01-01"), _month("2025-09-01"), _month("2025-10-01")],
        ignore_index=True,
    ).sample(frac=1, random_state=42)
    validation = _month("2025-11-01")

    partitions = partition_development_data(train, validation)

    assert len(partitions.base_fit) == 3
    assert len(partitions.tuning) == 3
    assert len(partitions.refit) == 6
    assert len(partitions.calibration) == 3
    assert len(partitions.validation) == 3
    assert partitions.base_fit["flight_date"].is_monotonic_increasing
    assert partitions.refit["flight_date"].is_monotonic_increasing
    assert set(partitions.refit.index).isdisjoint(partitions.calibration.index)


def test_sigmoid_calibration_and_reliability_evidence() -> None:
    features = pd.DataFrame({"x": list(range(-10, 0)) + list(range(1, 11))})
    target = pd.Series([0] * 10 + [1] * 10)
    base = LogisticRegression().fit(features, target)
    calibrated = fit_sigmoid_calibrator(base, features, target)
    probabilities = calibrated.predict_proba(features)[:, 1]

    table, ece = reliability_table(target, probabilities, bins=3)

    assert len(table) == 3
    assert sum(row["count"] for row in table) == 20
    assert 0 <= ece <= 1


def test_reliability_rejects_one_class() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        reliability_table([0, 0], [0.1, 0.2])


def test_perfect_and_deliberately_bad_calibration_fixtures() -> None:
    perfect = calibration_audit([0, 1], [0.5, 0.5])
    bad = calibration_audit([0, 1], [0.9, 0.9])

    assert perfect.equal_width_ece_10 == pytest.approx(0.0)
    assert perfect.equal_frequency_ece_15 == pytest.approx(0.0)
    assert bad.mean_probability_gap == pytest.approx(0.4)
    assert bad.equal_frequency_mce_15 == pytest.approx(0.4)


def test_ece_is_independent_of_global_mean_gap() -> None:
    target = [0, 1, 0, 1]
    probabilities = [0.2, 0.2, 0.8, 0.8]

    audit = calibration_audit(target, probabilities)

    assert mean_probability_gap(target, probabilities) == pytest.approx(0.0)
    assert audit.equal_frequency_ece_15 == pytest.approx(0.3)


def test_equal_width_table_preserves_empty_bins() -> None:
    table, ece, mce = calibration_table(
        [0, 1, 0, 1], [0.1, 0.1, 0.9, 0.9], bins=10, strategy="equal_width"
    )

    assert len(table) == 10
    assert sum(row["count"] == 0 for row in table) == 8
    assert ece == pytest.approx(0.4)
    assert mce == pytest.approx(0.4)


def test_equal_frequency_ties_are_not_split_and_boundaries_are_deterministic() -> None:
    target = [0, 1, 0, 1, 0, 1, 0, 1]
    probabilities = [0.1] * 4 + [0.9] * 4

    first, _, _ = calibration_table(target, probabilities, bins=15, strategy="equal_frequency")
    second, _, _ = calibration_table(target, probabilities, bins=15, strategy="equal_frequency")

    assert first == second
    assert [row["count"] for row in first] == [4, 4]
