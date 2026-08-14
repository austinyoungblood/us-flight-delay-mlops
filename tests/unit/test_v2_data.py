from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flight_delay.modeling.v2.data import (
    V2DataGuardError,
    load_december_features,
    prepare_development_data,
    require_allowed_v2_path,
)
from flight_delay.modeling.v2.features import build_historical_state
from flight_delay.modeling.v2.protocol import HISTORICAL_FEATURES, V2_FEATURES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_repository_root(tmp_path: Path) -> Path:
    """Stage governed metadata plus synthetic source sentinels outside the checkout."""

    governed_files = (
        "configs/v2_experiment_protocol.yaml",
        "experiments/v2/protocol_lock.json",
        "configs/v1_experiment_protocol.yaml",
        "experiments/v1/protocol_lock.json",
        "experiments/v1/development_result.json",
        "docs/v1-model-experiment-result.md",
        "release/selection_lock.json",
        "release/release_decision.json",
        "deploy/deployment_manifest.json",
        "data/manifests/processed_manifest.json",
        "data/manifests/source_manifest.json",
    )
    for relative in governed_files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(ROOT / relative)

    manifest = json.loads((ROOT / "data/manifests/processed_manifest.json").read_text())
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    for split in ("train", "validation"):
        specification = manifest["parquet_files"][split]
        with (processed / specification["filename"]).open("wb") as stream:
            stream.truncate(specification["byte_size"])
    return tmp_path


def _reader(frame: pd.DataFrame, calls: list[tuple[str, str]]) -> Any:
    def read(path: Path, *, filters: list[tuple[str, str, datetime]]) -> pd.DataFrame:
        start = filters[0][2]
        end = filters[1][2]
        calls.append((start.date().isoformat(), end.date().isoformat()))
        dates = pd.to_datetime(frame["flight_date"])
        return frame.loc[dates.ge(start) & dates.lt(end)].copy()

    return read


def test_prepare_development_data_stops_before_december_and_uses_full_history(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_v2_frame: pd.DataFrame,
    synthetic_repository_root: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    hash_calls: list[str] = []
    train_sha = json.loads((ROOT / "data/manifests/processed_manifest.json").read_text())[
        "parquet_files"
    ]["train"]["sha256"]

    def fake_sha(path: Path) -> str:
        hash_calls.append(path.name)
        return train_sha

    monkeypatch.setattr("flight_delay.modeling.v2.data.sha256_file", fake_sha)
    prepared = prepare_development_data(
        synthetic_repository_root,
        reader=_reader(synthetic_v2_frame, calls),
    )
    assert calls == [("2025-01-01", "2025-11-01"), ("2025-11-01", "2025-12-01")]
    assert hash_calls == ["train.parquet"]
    assert tuple(prepared.search.features.columns) == V2_FEATURES
    assert len(prepared.search.features) == 36
    assert len(prepared.full_refit.features) == 36
    assert len(prepared.calibration_features) == 2
    assert len(prepared.selection_features) == 2
    assert prepared.november_state.as_of.isoformat() == "2025-10-31"
    assert prepared.lineage["eligible_row_counts"] == {
        "burn_in_january": 4,
        "february_october": 36,
    }
    assert prepared.lineage["november_state_sha256"] == prepared.november_state.sha256


def test_december_loader_is_separate_and_reuses_october_state(
    synthetic_v2_frame: pd.DataFrame,
    synthetic_repository_root: Path,
) -> None:
    history = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(10)
    ]
    state = build_historical_state(history, as_of="2025-10-31")
    calls: list[tuple[str, str]] = []
    features, target, dates = load_december_features(
        synthetic_repository_root,
        state=state,
        reader=_reader(synthetic_v2_frame, calls),
        verify_source_hash=False,
    )
    assert calls == [("2025-12-01", "2026-01-01")]
    assert len(features) == len(target) == len(dates) == 4
    assert tuple(features.columns) == V2_FEATURES
    assert features.loc[:, HISTORICAL_FEATURES].notna().all().all()

    wrong_state = build_historical_state(history.iloc[:-4], as_of="2025-09-30")
    with pytest.raises(V2DataGuardError, match="October-31"):
        load_december_features(
            synthetic_repository_root,
            state=wrong_state,
            reader=_reader(synthetic_v2_frame, []),
            verify_source_hash=False,
        )


def test_data_path_guard_refuses_test_and_arbitrary_paths() -> None:
    split, path = require_allowed_v2_path(ROOT, ROOT / "data/processed/train.parquet")
    assert split == "train"
    assert path == (ROOT / "data/processed/train.parquet").resolve()
    with pytest.raises(V2DataGuardError, match="test.parquet"):
        require_allowed_v2_path(ROOT, ROOT / "data/processed/test.parquet")
    with pytest.raises(V2DataGuardError, match="canonical"):
        require_allowed_v2_path(ROOT, ROOT / "data/processed/other.parquet")
