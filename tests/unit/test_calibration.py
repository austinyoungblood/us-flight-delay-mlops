from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from flight_delay.modeling.calibration import (
    fit_sigmoid_calibrator,
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
    ).sort_values("flight_date", ignore_index=True)
    validation = _month("2025-11-01")

    partitions = partition_development_data(train, validation)

    assert len(partitions.base_fit) == 3
    assert len(partitions.tuning) == 3
    assert len(partitions.refit) == 6
    assert len(partitions.calibration) == 3
    assert len(partitions.validation) == 3
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
