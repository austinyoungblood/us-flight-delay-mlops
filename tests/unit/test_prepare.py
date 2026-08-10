from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from flight_delay.data.download import YearMonth
from flight_delay.data.manifest import validate_manifest, with_manifest_digest
from flight_delay.data.prepare import (
    CANDIDATE_A_FEATURES,
    OUTPUT_COLUMNS,
    PROCESSED_FEATURES,
    process_month_archive,
)
from flight_delay.data.preprocessing import DataQualityError
from flight_delay.features.leakage import validate_model_features


def _source_rows(month: int, rows: int = 20, *, one_class: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": [month] * rows,
            "DayofMonth": [(index % 28) + 1 for index in range(rows)],
            "DayOfWeek": [(index % 7) + 1 for index in range(rows)],
            "FlightDate": [f"2025-{month:02d}-{(index % 28) + 1:02d}" for index in range(rows)],
            "Reporting_Airline": ["UA"] * rows,
            "Origin": ["DEN"] * rows,
            "Dest": ["LAX"] * rows,
            "CRSDepTime": ["0700"] * rows,
            "CRSArrTime": ["0830"] * rows,
            "CRSElapsedTime": [150.0] * rows,
            "Distance": [862.0] * rows,
            "Cancelled": [1.0 if index == 0 else 0.0 for index in range(rows)],
            "Diverted": [1.0 if index == 1 else 0.0 for index in range(rows)],
            "ArrDel15": [0 if one_class else index % 2 for index in range(rows)],
        }
    )


def _write_zip(path: Path, member: str, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())


def test_process_month_is_stratified_capped_and_leakage_safe(tmp_path: Path) -> None:
    archive = tmp_path / "month.zip"
    member = "data.csv"
    _write_zip(archive, member, _source_rows(1))
    first, stats = process_month_archive(
        archive, member, YearMonth(2025, 1), sample_cap=10, seed=42
    )
    second, _ = process_month_archive(archive, member, YearMonth(2025, 1), sample_cap=10, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == OUTPUT_COLUMNS
    assert len(first) == 10
    assert set(first["target"]) == {0, 1}
    assert stats.source_rows == 20
    assert stats.cancelled_rows == 1
    assert stats.diverted_rows == 1
    assert stats.eligible_rows == 18
    assert stats.invalid_schedule_rows == 0
    assert stats.model_eligible_rows == 18
    assert abs(stats.model_eligible_target_prevalence - stats.sampled_target_prevalence) <= 0.05
    assert "Cancelled" not in first
    assert "ArrDel15" not in first
    validate_model_features(PROCESSED_FEATURES)
    validate_model_features(CANDIDATE_A_FEATURES)


def test_process_month_counts_and_excludes_invalid_schedule_rows(tmp_path: Path) -> None:
    archive = tmp_path / "invalid-schedule.zip"
    rows = _source_rows(1)
    rows.loc[2, "CRSElapsedTime"] = None
    rows.loc[3, "CRSDepTime"] = "2360"
    _write_zip(archive, "data.csv", rows)

    prepared, stats = process_month_archive(
        archive, "data.csv", YearMonth(2025, 1), sample_cap=None, seed=42
    )

    assert stats.eligible_rows == 18
    assert stats.invalid_schedule_rows == 2
    assert stats.model_eligible_rows == 16
    assert len(prepared) == 16


def test_process_month_rejects_one_class(tmp_path: Path) -> None:
    archive = tmp_path / "one-class.zip"
    _write_zip(archive, "data.csv", _source_rows(1, one_class=True))
    with pytest.raises(DataQualityError, match="both target classes"):
        process_month_archive(archive, "data.csv", YearMonth(2025, 1), sample_cap=10, seed=42)


def test_processed_manifest_digest_is_stable() -> None:
    payload = {
        "schema_version": 1,
        "source_manifest_digest": "a" * 64,
        "split_counts": {
            "train": {"row_count": 10, "target_prevalence": 0.2},
            "validation": {"row_count": 4, "target_prevalence": 0.25},
        },
    }
    first = with_manifest_digest(payload)
    second = with_manifest_digest({key: payload[key] for key in reversed(payload)})
    assert first["manifest_digest"] == second["manifest_digest"]
    assert validate_manifest(first) == first["manifest_digest"]
