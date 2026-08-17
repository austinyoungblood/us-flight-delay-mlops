"""Tracking adapters and the remaining v3 split-read guards."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flight_delay.data.manifest import write_manifest
from flight_delay.data.prepare_v3 import V3_PROCESSED_DIRECTORY, V3_PROCESSED_MANIFEST
from flight_delay.modeling.v3.data import V3DataGuardError, prepare_development_data
from flight_delay.modeling.v3.tracking import NullTracker, WandbTracker
from tests.unit.test_v3_data import build_root


def test_null_tracker_records_runs_and_payloads() -> None:
    tracker = NullTracker()
    with tracker.start_run(name="v3-run", group="v3-group", metadata={"stage": "screening"}) as run:
        run.log({"FOLD_1/precision": 0.31})
        run.log({"FOLD_2/precision": 0.33})
    assert len(tracker.runs) == 1
    assert tracker.runs[0].metadata["group"] == "v3-group"
    assert tracker.runs[0].logged == [
        {"FOLD_1/precision": 0.31},
        {"FOLD_2/precision": 0.33},
    ]
    assert tracker.runs[0].id == "null"


def test_online_tracker_requires_entity_and_project() -> None:
    with pytest.raises(ValueError, match="entity and project"):
        WandbTracker(entity="", project="us-flight-delay-mlops")
    with pytest.raises(ValueError, match="entity and project"):
        WandbTracker(entity="team", project="")


def test_online_tracker_starts_a_governed_v3_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "run-handle"

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    tracker = WandbTracker(entity="team", project="us-flight-delay-mlops")
    assert (
        tracker.start_run(name="v3-CB04", group="v3-group", metadata={"stage": "x"}) == "run-handle"
    )
    assert captured["job_type"] == "governed-v3-experiment"
    assert captured["mode"] == "online"
    assert captured["group"] == "v3-group"
    assert captured["config"] == {"stage": "x"}


def _mutate_manifest(root: Path, **updates: Any) -> None:
    path = root / V3_PROCESSED_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manifest_digest", None)
    for key, value in updates.items():
        payload[key] = value
    write_manifest(path, payload)


def test_split_size_mismatch_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    manifest = json.loads((root / V3_PROCESSED_MANIFEST).read_text(encoding="utf-8"))
    files = manifest["parquet_files"]
    files["v3_history"]["byte_size"] += 1
    _mutate_manifest(root, parquet_files=files)
    with pytest.raises(V3DataGuardError, match="size mismatch"):
        prepare_development_data(root, verify_history_hash=False)


def test_split_hash_mismatch_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    with pytest.raises(V3DataGuardError, match="SHA256 mismatch"):
        prepare_development_data(root, verify_history_hash=True)


def test_a_reader_returning_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)

    def bad_reader(path: Path, **_: Any) -> pd.DataFrame:
        return pd.DataFrame({"unexpected": [1, 2, 3]})

    with pytest.raises(V3DataGuardError, match="schema differs"):
        prepare_development_data(root, reader=bad_reader, verify_history_hash=False)


def test_a_reader_leaking_rows_outside_the_period_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    real = pd.read_parquet

    def leaky_reader(path: Path, **kwargs: Any) -> pd.DataFrame:
        # Ignore the pushed-down filters so December-shaped rows slip through.
        if Path(path).name == "v3_november.parquet":
            frame = real(path)
            leaked = frame.head(5).copy()
            leaked["flight_date"] = pd.Timestamp("2025-12-05")
            return pd.concat([frame, leaked], ignore_index=True)
        return real(path, **kwargs)

    with pytest.raises(V3DataGuardError, match="outside the requested period"):
        prepare_development_data(root, reader=leaky_reader, verify_history_hash=False)


def test_an_empty_split_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    real = pd.read_parquet

    def empty_reader(path: Path, **kwargs: Any) -> pd.DataFrame:
        return real(path, **kwargs).head(0)

    with pytest.raises(V3DataGuardError, match="empty"):
        prepare_development_data(root, reader=empty_reader, verify_history_hash=False)


def test_a_missing_split_file_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    (root / V3_PROCESSED_DIRECTORY / "v3_history.parquet").unlink()
    with pytest.raises(V3DataGuardError, match="missing"):
        prepare_development_data(root, verify_history_hash=False)


def test_a_manifest_that_decoded_2026_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    _mutate_manifest(root, january_may_2026_decoded=True)
    with pytest.raises(V3DataGuardError, match="never be decoded"):
        prepare_development_data(root, verify_history_hash=False)


def test_a_manifest_describing_december_is_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    manifest = json.loads((root / V3_PROCESSED_MANIFEST).read_text(encoding="utf-8"))
    files = dict(manifest["parquet_files"])
    files["v3_december"] = {"filename": "v3_december.parquet", "byte_size": 1, "row_count": 1}
    _mutate_manifest(root, parquet_files=files)
    with pytest.raises(V3DataGuardError, match="must not describe December"):
        prepare_development_data(root, verify_history_hash=False)
