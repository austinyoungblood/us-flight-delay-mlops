import pandas as pd
import pytest

from flight_delay.data.preprocessing import DataQualityError
from flight_delay.data.sampling import deterministic_monthly_sample
from flight_delay.data.splitting import chronological_split


def test_exact_temporal_boundaries_prevent_future_to_past_leakage() -> None:
    frame = pd.DataFrame(
        {
            "FlightDate": [
                "2024-12-31",
                "2025-01-01",
                "2025-10-31",
                "2025-11-01",
                "2025-12-31",
                "2026-01-01",
                "2026-05-31",
                "2026-06-01",
            ],
            "id": list(range(8)),
        }
    )
    split = chronological_split(
        frame,
        train_start="2025-01-01",
        validation_start="2025-11-01",
        test_start="2026-01-01",
        test_end="2026-06-01",
    )
    assert split.train["id"].tolist() == [1, 2]
    assert split.validation["id"].tolist() == [3, 4]
    assert split.test["id"].tolist() == [5, 6]
    assert (
        pd.to_datetime(split.train["FlightDate"]).max()
        < pd.to_datetime(split.validation["FlightDate"]).min()
    )
    assert (
        pd.to_datetime(split.validation["FlightDate"]).max()
        < pd.to_datetime(split.test["FlightDate"]).min()
    )


def test_temporal_boundaries_must_be_strictly_ordered() -> None:
    with pytest.raises(DataQualityError, match="strictly increasing"):
        chronological_split(
            pd.DataFrame({"FlightDate": ["2025-01-01"]}),
            train_start="2025-01-01",
            validation_start="2026-01-01",
            test_start="2025-11-01",
            test_end="2026-06-01",
        )


def test_monthly_sample_is_capped_stratified_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "FlightDate": ["2025-01-01"] * 10 + ["2025-02-01"] * 8,
            "target": [0] * 8 + [1] * 2 + [0] * 4 + [1] * 4,
            "id": range(18),
        }
    )
    first = deterministic_monthly_sample(frame, max_rows_per_month=4, seed=42)
    second = deterministic_monthly_sample(frame, max_rows_per_month=4, seed=42)
    pd.testing.assert_frame_equal(first, second)
    counts = pd.to_datetime(first["FlightDate"]).dt.to_period("M").value_counts()
    assert counts.to_dict() == {pd.Period("2025-01", "M"): 4, pd.Period("2025-02", "M"): 4}
    january = first[pd.to_datetime(first["FlightDate"]).dt.month.eq(1)]
    assert set(january["target"]) == {0, 1}
