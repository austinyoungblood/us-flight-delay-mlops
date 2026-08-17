"""V3 data guards: canonical paths, sealed December, and the November split."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from flight_delay.data.manifest import write_manifest
from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.data.prepare_v3 import V3_PROCESSED_DIRECTORY, V3_PROCESSED_MANIFEST
from flight_delay.modeling.v3.data import (
    V3DataGuardError,
    prepare_development_data,
    require_allowed_v3_path,
)
from flight_delay.modeling.v3.protocol import V3_FEATURES
from tests.conftest import make_v3_frame

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V3_SOURCE_MANIFEST_DIGEST = "673cac214739e8c0d2991a1bdbd1591a90e8907d7cf5bdbc34caddd72015b6af"

LINEAGE_FILES = (
    "configs/v1_experiment_protocol.yaml",
    "configs/v2_experiment_protocol.yaml",
    "configs/v3_experiment_protocol.yaml",
    "experiments/v1/protocol_lock.json",
    "experiments/v1/development_result.json",
    "experiments/v2/protocol_lock.json",
    "experiments/v3/protocol_lock.json",
    "docs/v1-model-experiment-result.md",
    "release/selection_lock.json",
    "release/release_decision.json",
    "deploy/deployment_manifest.json",
    "data/manifests/source_manifest.json",
    "data/manifests/processed_manifest.json",
    "data/manifests/v3_source_manifest.json",
)


def build_root(tmp_path: Path, *, include_december: bool = False) -> Path:
    """Create a synthetic repository whose lineage bytes match the real frozen artifacts."""

    root = tmp_path / "repo"
    for relative in LINEAGE_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, destination)

    processed = root / V3_PROCESSED_DIRECTORY
    processed.mkdir(parents=True, exist_ok=True)
    splits = {
        "v3_history": make_v3_frame(start="2024-01-01", end="2025-10-31"),
        "v3_november": make_v3_frame(start="2025-11-01", end="2025-11-30"),
    }
    if include_december:
        splits["v3_december"] = make_v3_frame(start="2025-12-01", end="2025-12-31")

    parquet_files = {}
    split_counts = {}
    for name, frame in splits.items():
        path = processed / f"{name}.parquet"
        frame.loc[:, OUTPUT_COLUMNS].to_parquet(path, engine="pyarrow", index=False)
        parquet_files[name] = {
            "filename": path.name,
            "byte_size": path.stat().st_size,
            "row_count": len(frame),
            "sha256": "0" * 64,
        }
        split_counts[name] = {
            "row_count": len(frame),
            "target_prevalence": float(frame["target"].mean()),
        }
    write_manifest(
        root / V3_PROCESSED_MANIFEST,
        {
            "schema_version": 1,
            "source_manifest_digest": V3_SOURCE_MANIFEST_DIGEST,
            "december_2025_decoded": include_december,
            "january_may_2026_decoded": False,
            "v3_feature_schema": list(V3_FEATURES),
            "split_counts": split_counts,
            "parquet_files": parquet_files,
            "monthly_counts": [],
        },
    )
    return root


def test_only_the_two_development_splits_resolve(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    processed = root / V3_PROCESSED_DIRECTORY
    assert require_allowed_v3_path(root, processed / "v3_history.parquet")[0] == "v3_history"
    assert require_allowed_v3_path(root, processed / "v3_november.parquet")[0] == "v3_november"


def test_december_and_the_sealed_test_are_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    processed = root / V3_PROCESSED_DIRECTORY
    with pytest.raises(V3DataGuardError, match="prohibited during development"):
        require_allowed_v3_path(root, processed / "v3_december.parquet")
    with pytest.raises(V3DataGuardError, match="prohibited during development"):
        require_allowed_v3_path(root, root / "data/processed/test.parquet")


def test_arbitrary_paths_are_refused(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    with pytest.raises(V3DataGuardError, match="canonical"):
        require_allowed_v3_path(root, tmp_path / "elsewhere.parquet")


def test_development_preparation_builds_every_matrix(tmp_path: Path) -> None:
    prepared = prepare_development_data(build_root(tmp_path), verify_history_hash=False)
    assert tuple(prepared.search.features.columns) == V3_FEATURES
    assert tuple(prepared.full_refit.features.columns) == V3_FEATURES
    assert len(prepared.calibration_target) > 0
    assert len(prepared.selection_target) > 0
    assert prepared.november_state.as_of.isoformat() == "2025-10-31"


def test_fold_matrix_reaches_november_while_refit_stops_at_october(tmp_path: Path) -> None:
    prepared = prepare_development_data(build_root(tmp_path), verify_history_hash=False)
    search_dates = pd.to_datetime(prepared.search.flight_date)
    refit_dates = pd.to_datetime(prepared.full_refit.flight_date)
    assert search_dates.max() >= pd.Timestamp("2025-11-01")
    assert refit_dates.max() < pd.Timestamp("2025-11-01")
    assert refit_dates.min() == pd.Timestamp("2024-02-01")
    assert prepared.lineage["model_row_counts"]["search_november_evaluation_only"] > 0


def test_january_2024_is_burn_in_only(tmp_path: Path) -> None:
    prepared = prepare_development_data(build_root(tmp_path), verify_history_hash=False)
    assert prepared.lineage["eligible_row_counts"]["burn_in_january_2024"] > 0
    assert pd.to_datetime(prepared.full_refit.flight_date).min() == pd.Timestamp("2024-02-01")


def test_november_halves_split_on_the_sixteenth(tmp_path: Path) -> None:
    prepared = prepare_development_data(build_root(tmp_path), verify_history_hash=False)
    assert pd.to_datetime(prepared.calibration_date).max() < pd.Timestamp("2025-11-16")
    assert pd.to_datetime(prepared.selection_date).min() >= pd.Timestamp("2025-11-16")


def test_lineage_records_that_december_stayed_sealed(tmp_path: Path) -> None:
    prepared = prepare_development_data(build_root(tmp_path), verify_history_hash=False)
    assert prepared.lineage["december_decoded"] is False
    assert prepared.lineage["january_may_2026_accessed"] is False
    assert prepared.lineage["november_state_sha256"]


def test_development_refuses_a_manifest_that_decoded_december(tmp_path: Path) -> None:
    root = build_root(tmp_path, include_december=True)
    with pytest.raises(V3DataGuardError, match="must not be decoded"):
        prepare_development_data(root, verify_history_hash=False)


def test_development_refuses_a_missing_split(tmp_path: Path) -> None:
    root = build_root(tmp_path)
    (root / V3_PROCESSED_DIRECTORY / "v3_november.parquet").unlink()
    with pytest.raises(V3DataGuardError):
        prepare_development_data(root, verify_history_hash=False)
