from datetime import time

import pandas as pd
import pytest

from flight_delay.data.preprocessing import (
    DataQualityError,
    InvalidCRSTimeError,
    construct_binary_target,
    filter_eligible_flights,
    normalize_bts_columns,
    parse_crs_time,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, time(0, 0)),
        ("0000", time(0, 0)),
        (5, time(0, 5)),
        ("0930", time(9, 30)),
        (2400, time(0, 0)),
        (1535.0, time(15, 35)),
    ],
)
def test_parse_crs_time_edge_cases(raw: object, expected: time) -> None:
    assert parse_crs_time(raw) == expected


@pytest.mark.parametrize("raw", [None, pd.NA, "", "09:30", "abcd", -1, 2360, 2500, 930.5, True])
def test_parse_crs_time_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(InvalidCRSTimeError):
        parse_crs_time(raw)


def test_normalize_columns_and_verify_required_fields() -> None:
    raw = pd.DataFrame({"\ufeffreporting airline ": ["UA"], "crs_dep_time": [700]})
    normalized = normalize_bts_columns(
        raw, required_columns=frozenset({"Reporting_Airline", "CRSDepTime"})
    )
    assert list(normalized.columns) == ["Reporting_Airline", "CRSDepTime"]
    with pytest.raises(DataQualityError, match="missing required"):
        normalize_bts_columns(raw, required_columns=frozenset({"Dest"}))


def test_target_construction_is_binary_and_typed() -> None:
    result = construct_binary_target(pd.DataFrame({"ArrDel15": [0.0, 1.0]}))
    assert result["target"].tolist() == [0, 1]
    assert str(result["target"].dtype) == "int8"
    with pytest.raises(DataQualityError, match="must be binary"):
        construct_binary_target(pd.DataFrame({"ArrDel15": [2]}))


def test_eligibility_counts_are_sequential_and_reconcile() -> None:
    frame = pd.DataFrame(
        {
            "Cancelled": [1, 0, 0, 0, 1],
            "Diverted": [1, 1, 0, 0, 0],
            "ArrDel15": [None, 1, None, 0, 1],
            "row": ["both", "diverted", "missing", "keep", "cancelled"],
        }
    )
    result = filter_eligible_flights(frame)
    assert result.frame["row"].tolist() == ["keep"]
    assert result.counts.input_rows == 5
    assert result.counts.cancelled == 2
    assert result.counts.diverted == 1
    assert result.counts.missing_target == 1
    assert result.counts.eligible_rows == 1
    assert result.counts.excluded_rows == 4


@pytest.mark.parametrize("column", ["Cancelled", "Diverted"])
def test_eligibility_rejects_missing_or_invalid_outcome_flags(column: str) -> None:
    frame = pd.DataFrame(
        {
            "Cancelled": ["unknown" if column == "Cancelled" else 0],
            "Diverted": ["unknown" if column == "Diverted" else 0],
            "ArrDel15": [1],
        }
    )
    with pytest.raises(DataQualityError, match="missing/invalid"):
        filter_eligible_flights(frame)
