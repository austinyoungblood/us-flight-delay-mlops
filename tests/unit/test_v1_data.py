from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.modeling import v1_data
from flight_delay.modeling.v1_data import (
    V1_CATEGORICAL_FEATURES,
    V1_FEATURES,
    V1DataGuardError,
    adapt_v1_frame,
    development_period,
    load_december_data,
    load_development_data,
    require_allowed_v1_path,
)

ROOT = Path(__file__).resolve().parents[2]


def _frame(date: str = "2025-07-01", rows: int = 4) -> pd.DataFrame:
    values: dict[str, object] = {
        "flight_date": pd.date_range(date, periods=rows),
        "Month": [7] * rows,
        "DayofMonth": list(range(1, rows + 1)),
        "DayOfWeek": [1, 2, 3, 4][:rows],
        "Reporting_Airline": ["UA"] * rows,
        "Origin": ["DEN"] * rows,
        "Dest": ["SFO"] * rows,
        "CRSDepTime": [800] * rows,
        "CRSArrTime": [1000] * rows,
        "CRSElapsedTime": [120.0] * rows,
        "Distance": [950.0] * rows,
        "route": ["DEN-SFO"] * rows,
        "scheduled_departure_hour": [8] * rows,
        "scheduled_arrival_hour": [10] * rows,
        "scheduled_departure_minute_bucket": [0] * rows,
        "scheduled_arrival_minute_bucket": [0] * rows,
        "is_weekend": [0] * rows,
        "scheduled_departure_sin": [0.8] * rows,
        "scheduled_departure_cos": [-0.2] * rows,
        "scheduled_arrival_sin": [0.5] * rows,
        "scheduled_arrival_cos": [-0.5] * rows,
        "target": [0, 1, 0, 1][:rows],
    }
    return pd.DataFrame(values, columns=OUTPUT_COLUMNS)


def test_feature_adapter_is_exact_native_and_does_not_mutate_source() -> None:
    frame = _frame().sample(frac=1, random_state=7)
    before = frame.copy(deep=True)
    adapted = adapt_v1_frame(frame)
    assert tuple(adapted.features.columns) == V1_FEATURES
    assert tuple(name for name in adapted.features if adapted.features[name].dtype == object) == (
        V1_CATEGORICAL_FEATURES
    )
    assert adapted.flight_date.is_monotonic_increasing
    pd.testing.assert_frame_equal(frame, before)
    assert "flight_date" not in adapted.features and "target" not in adapted.features


@pytest.mark.parametrize("column", V1_CATEGORICAL_FEATURES)
def test_feature_adapter_rejects_missing_categories(column: str) -> None:
    frame = _frame()
    frame.loc[0, column] = None
    with pytest.raises(V1DataGuardError, match="categorical feature"):
        adapt_v1_frame(frame)


def test_feature_adapter_rejects_nonfinite_and_forbidden_inputs() -> None:
    nonfinite = _frame()
    nonfinite.loc[0, "Distance"] = np.inf
    with pytest.raises(V1DataGuardError, match="must be finite"):
        adapt_v1_frame(nonfinite)
    forbidden = _frame().rename(columns={"Distance": "ArrDelay"})
    with pytest.raises(V1DataGuardError, match="forbidden outcome"):
        adapt_v1_frame(forbidden)


def test_feature_adapter_rejects_schema_dates_empty_category_and_target() -> None:
    with pytest.raises(V1DataGuardError, match="exact canonical"):
        adapt_v1_frame(_frame().drop(columns="Distance"))
    invalid_date = _frame()
    invalid_date.loc[0, "flight_date"] = pd.NaT
    with pytest.raises(V1DataGuardError, match="flight_date"):
        adapt_v1_frame(invalid_date)
    empty_category = _frame()
    empty_category.loc[0, "Origin"] = "  "
    with pytest.raises(V1DataGuardError, match="empty values"):
        adapt_v1_frame(empty_category)
    invalid_target = _frame()
    invalid_target.loc[0, "target"] = 2
    with pytest.raises(V1DataGuardError, match="binary"):
        adapt_v1_frame(invalid_target)


def test_data_guard_refuses_test_and_arbitrary_paths() -> None:
    with pytest.raises(V1DataGuardError, match="test.parquet"):
        require_allowed_v1_path(ROOT, ROOT / "data/processed/test.parquet")
    with pytest.raises(V1DataGuardError, match="only canonical"):
        require_allowed_v1_path(ROOT, ROOT / "tmp/train.parquet")


def test_development_and_december_reads_use_exact_predicate_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, object]] = []
    monkeypatch.setattr("flight_delay.modeling.v1_data._verify_split_hash", lambda *_args: None)

    def reader(path: Path, *, filters: object) -> pd.DataFrame:
        requests.append((path.name, filters))
        if path.name == "train.parquet":
            return _frame("2025-07-01")
        start = filters[0][2]
        return _frame(start.strftime("%Y-%m-%d"))

    development = load_development_data(ROOT, reader=reader)
    december = load_december_data(ROOT, reader=reader)
    assert development.train["flight_date"].max() < pd.Timestamp("2025-11-01")
    assert development.november["flight_date"].min() == pd.Timestamp("2025-11-01")
    assert december["flight_date"].min() == pd.Timestamp("2025-12-01")
    assert [name for name, _filters in requests] == [
        "train.parquet",
        "validation.parquet",
        "validation.parquet",
    ]
    assert requests[1][1][0][2] == pd.Timestamp("2025-11-01")
    assert requests[2][1][0][2] == pd.Timestamp("2025-12-01")


def test_hash_failure_occurs_before_parquet_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flight_delay.modeling.v1_data._verify_split_hash",
        lambda *_args: (_ for _ in ()).throw(V1DataGuardError("hash failed")),
    )
    opened = False

    def reader(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        return _frame()

    with pytest.raises(V1DataGuardError, match="hash failed"):
        load_development_data(ROOT, reader=reader)
    assert opened is False


def test_split_hash_guard_requires_file_and_exact_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(V1DataGuardError, match="is missing"):
        v1_data._verify_split_hash(missing, "train", {"parquet_files": {"train": {}}})
    path = tmp_path / "train.parquet"
    path.write_bytes(b"toy")
    manifest = {"parquet_files": {"train": {"sha256": "expected"}}}
    monkeypatch.setattr(v1_data, "sha256_file", lambda _path: "wrong")
    with pytest.raises(V1DataGuardError, match="SHA256 mismatch"):
        v1_data._verify_split_hash(path, "train", manifest)
    monkeypatch.setattr(v1_data, "sha256_file", lambda _path: "expected")
    v1_data._verify_split_hash(path, "train", manifest)


def test_post_read_boundary_and_development_december_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flight_delay.modeling.v1_data._verify_split_hash", lambda *_args: None)

    def out_of_window(path: Path, **_kwargs: object) -> pd.DataFrame:
        return _frame("2025-12-01" if path.name == "validation.parquet" else "2025-07-01")

    with pytest.raises(V1DataGuardError, match="outside the requested period"):
        load_development_data(ROOT, reader=out_of_window)
    with pytest.raises(V1DataGuardError, match="cannot request December"):
        development_period(_frame(), "2025-12-01", "2026-01-01")
    with pytest.raises(V1DataGuardError, match="is empty"):
        development_period(_frame(), "2025-08-01", "2025-09-01")
