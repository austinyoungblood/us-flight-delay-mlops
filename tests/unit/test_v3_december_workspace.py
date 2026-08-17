"""BLOCKER 2: December materializes only into the Git-ignored qualification workspace.

Qualifying must never require mutating a tracked file. A tracked mutation would dirty the worktree
that the clean-main guard depends on and invalidate the frozen winner's code lineage, so December
data lives entirely under ``artifacts/v3/qualification``.

No model is fit anywhere in this module.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from flight_delay.data.download import YearMonth
from flight_delay.data.prepare_v3 import (
    DECEMBER_AUTHORIZATION,
    QUALIFICATION_DECEMBER_MANIFEST,
    QUALIFICATION_DECEMBER_PARQUET,
    QUALIFICATION_WORKSPACE,
    V3_PROCESSED_MANIFEST,
    V3PreparationError,
    materialize_december_qualification_data,
    split_for_month,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "data/manifests/v3_source_manifest.json"


def _synthetic_december_archive(directory: Path) -> tuple[Path, str]:
    """Build a real BTS-shaped ZIP for December 2025 so no genuine archive is ever decoded.

    Proving the artifact-placement contract does not require the real December labels, and
    decoding them here would consume a set that exists precisely to stay untouched.
    """

    import csv
    import io
    import zipfile

    from flight_delay.data.download import sha256_file
    from flight_delay.data.prepare import REQUIRED_SOURCE_COLUMNS

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(REQUIRED_SOURCE_COLUMNS))
    writer.writeheader()
    for index, day in enumerate((3, 9, 14, 20, 27)):
        for arrival_delayed in (0, 1):
            writer.writerow(
                {
                    "Month": 12,
                    "DayofMonth": day,
                    "DayOfWeek": (index % 7) + 1,
                    "FlightDate": f"2025-12-{day:02d}",
                    "Reporting_Airline": "UA" if index % 2 else "AA",
                    "Origin": "DEN" if index % 2 else "SFO",
                    "Dest": "ORD" if index % 2 else "DEN",
                    "CRSDepTime": f"{7 + index:02d}30",
                    "CRSArrTime": f"{9 + index:02d}45",
                    "CRSElapsedTime": 120,
                    "Distance": 900 + index * 25,
                    "Cancelled": 0,
                    "Diverted": 0,
                    "ArrDel15": arrival_delayed,
                }
            )

    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "synthetic_2025_12.zip"
    member = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_12.csv"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())
    return archive_path, sha256_file(archive_path)


@pytest.fixture
def target_root(tmp_path: Path) -> Path:
    """A clean repository root for materialization, separate from the fixture's own inputs."""

    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def synthetic_source(tmp_path: Path) -> Path:
    """A v3-shaped source manifest whose only December record is the synthetic archive."""

    from flight_delay.data.manifest import write_manifest

    raw = tmp_path / "raw"
    archive_path, checksum = _synthetic_december_archive(raw)
    manifest_path = tmp_path / "synthetic_source_manifest.json"
    write_manifest(
        manifest_path,
        {
            "schema_version": 1,
            "dataset_name": "synthetic",
            "start_month": "2024-01",
            "end_month": "2025-12",
            "expected_archive_count": 1,
            "files": [
                {
                    "year": 2025,
                    "month": 12,
                    "url": "synthetic",
                    "archive_filename": archive_path.name,
                    "byte_size": archive_path.stat().st_size,
                    "sha256": checksum,
                    "selected_csv_member": (
                        "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_12.csv"
                    ),
                    "zip_members": [
                        "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_12.csv"
                    ],
                }
            ],
        },
    )
    return manifest_path


def test_qualification_paths_live_under_ignored_artifacts() -> None:
    assert str(QUALIFICATION_WORKSPACE) == "artifacts/v3/qualification"
    assert str(QUALIFICATION_DECEMBER_PARQUET).startswith("artifacts/v3/qualification/")
    assert str(QUALIFICATION_DECEMBER_MANIFEST).startswith("artifacts/v3/qualification/")
    assert not str(QUALIFICATION_DECEMBER_PARQUET).startswith("data/")
    assert not str(QUALIFICATION_DECEMBER_MANIFEST).startswith("data/")


def test_the_qualification_workspace_is_git_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(QUALIFICATION_DECEMBER_PARQUET)],
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, "the qualification workspace must be Git-ignored"


def test_development_cannot_materialize_december() -> None:
    """Development routing has no December branch, so no flag can turn it on."""

    assert split_for_month(YearMonth(2025, 12)) is None
    import inspect

    from flight_delay.data.prepare_v3 import prepare_v3_dataset

    assert "december" not in str(inspect.signature(prepare_v3_dataset)).casefold()


def test_a_wrong_authorization_is_refused(target_root: Path, synthetic_source: Path) -> None:
    for wrong in ("", "please", "december-2025", DECEMBER_AUTHORIZATION.upper()):
        with pytest.raises(V3PreparationError, match="exact qualification authorization"):
            materialize_december_qualification_data(
                target_root,
                december_authorization=wrong,
                source_manifest_path=synthetic_source,
                raw_directory=synthetic_source.parent / "raw",
            )
    assert not (target_root / QUALIFICATION_WORKSPACE).exists()


def test_a_wrong_authorization_writes_nothing(target_root: Path, synthetic_source: Path) -> None:
    with pytest.raises(V3PreparationError):
        materialize_december_qualification_data(
            target_root,
            december_authorization="nope",
            source_manifest_path=synthetic_source,
            raw_directory=synthetic_source.parent / "raw",
        )
    assert not (target_root / QUALIFICATION_DECEMBER_PARQUET).exists()
    assert not (target_root / QUALIFICATION_DECEMBER_MANIFEST).exists()


def test_materialization_refuses_a_manifest_referencing_2026(
    target_root: Path, tmp_path: Path, synthetic_source: Path
) -> None:
    from flight_delay.data.manifest import write_manifest

    source = json.loads(synthetic_source.read_text(encoding="utf-8"))
    source["files"].append({**source["files"][0], "year": 2026, "month": 1})
    tainted = tmp_path / "tainted.json"
    write_manifest(tainted, {k: v for k, v in source.items() if k != "manifest_digest"})
    with pytest.raises(V3PreparationError, match="2026"):
        materialize_december_qualification_data(
            target_root,
            december_authorization=DECEMBER_AUTHORIZATION,
            source_manifest_path=tainted,
            raw_directory=synthetic_source.parent / "raw",
        )


def test_materialization_refuses_a_checksum_mismatch(
    target_root: Path, tmp_path: Path, synthetic_source: Path
) -> None:
    from flight_delay.data.manifest import write_manifest

    source = json.loads(synthetic_source.read_text(encoding="utf-8"))
    for record in source["files"]:
        if record["year"] == 2025 and record["month"] == 12:
            record["sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    write_manifest(tampered, {k: v for k, v in source.items() if k != "manifest_digest"})
    with pytest.raises(V3PreparationError, match="checksum mismatch|archive"):
        materialize_december_qualification_data(
            target_root,
            december_authorization=DECEMBER_AUTHORIZATION,
            source_manifest_path=tampered,
            raw_directory=synthetic_source.parent / "raw",
        )


def test_authorized_materialization_writes_only_ignored_artifacts(
    target_root: Path, synthetic_source: Path
) -> None:
    result = materialize_december_qualification_data(
        target_root,
        december_authorization=DECEMBER_AUTHORIZATION,
        source_manifest_path=synthetic_source,
        raw_directory=synthetic_source.parent / "raw",
    )
    assert result.parquet_path == target_root / QUALIFICATION_DECEMBER_PARQUET
    assert result.manifest_path == target_root / QUALIFICATION_DECEMBER_MANIFEST
    assert result.parquet_path.is_file()
    assert result.manifest_path.is_file()
    assert result.tracked_development_manifest_mutated is False
    assert result.manifest["december_2025_materialized"] is True
    assert result.manifest["january_may_2026_referenced"] is False
    assert result.stats.month == "2025-12"

    # Nothing at all was written outside the qualification workspace.
    written = {path.relative_to(target_root) for path in target_root.rglob("*") if path.is_file()}
    assert all(str(path).startswith("artifacts/v3/qualification/") for path in written), written


def test_materialization_never_writes_the_tracked_development_manifest(
    target_root: Path, synthetic_source: Path
) -> None:
    tracked = ROOT / V3_PROCESSED_MANIFEST
    before = tracked.read_bytes() if tracked.is_file() else None
    materialize_december_qualification_data(
        target_root,
        december_authorization=DECEMBER_AUTHORIZATION,
        source_manifest_path=synthetic_source,
        raw_directory=synthetic_source.parent / "raw",
    )
    assert not (target_root / V3_PROCESSED_MANIFEST).exists()
    if before is not None:
        assert tracked.read_bytes() == before, "tracked development manifest must stay identical"


def test_materialized_december_contains_only_december_2025(
    target_root: Path, synthetic_source: Path
) -> None:
    result = materialize_december_qualification_data(
        target_root,
        december_authorization=DECEMBER_AUTHORIZATION,
        source_manifest_path=synthetic_source,
        raw_directory=synthetic_source.parent / "raw",
    )
    dates = pd.to_datetime(pd.read_parquet(result.parquet_path)["flight_date"])
    assert dates.min() >= pd.Timestamp("2025-12-01")
    assert dates.max() < pd.Timestamp("2026-01-01")


def test_materialization_refuses_to_run_twice(target_root: Path, synthetic_source: Path) -> None:
    materialize_december_qualification_data(
        target_root,
        december_authorization=DECEMBER_AUTHORIZATION,
        source_manifest_path=synthetic_source,
        raw_directory=synthetic_source.parent / "raw",
    )
    with pytest.raises(V3PreparationError, match="already been materialized"):
        materialize_december_qualification_data(
            target_root,
            december_authorization=DECEMBER_AUTHORIZATION,
            source_manifest_path=synthetic_source,
            raw_directory=synthetic_source.parent / "raw",
        )


def test_qualification_materializes_only_after_the_winner_is_validated() -> None:
    """Ordering matters: the handoff validation must precede any December decode."""

    import inspect

    from flight_delay.modeling.v3.execution import run_december_apply

    source = inspect.getsource(run_december_apply)
    assert source.index("validate_december_handoff") < source.index(
        "materialize_december_qualification_data"
    )
    assert source.index("require_merged_applied_state") < source.index(
        "materialize_december_qualification_data"
    )
    assert source.index("create_marker") < source.index("materialize_december_qualification_data")


def test_qualification_performs_no_refit_recalibration_or_state_update() -> None:
    import inspect

    from flight_delay.modeling.v3.execution import run_december_apply

    source = inspect.getsource(run_december_apply)
    for forbidden in ("run_refit_and_november", "build_calibration_variant", "fit_candidate"):
        assert forbidden not in source
    assert '"refit_performed": False' in source
    assert '"recalibration_performed": False' in source
    assert '"threshold_changed": False' in source
    assert '"historical_state_updated": False' in source


def test_january_may_2026_remains_inaccessible() -> None:
    """No 2026 archive is manifested, so December materialization cannot reach one."""

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert all(int(record["year"]) < 2026 for record in manifest["files"])
    with pytest.raises(V3PreparationError, match="prohibited"):
        split_for_month(YearMonth(2026, 1))


def _write_qualification(root: Path, frame: pd.DataFrame, **manifest_overrides: object) -> Path:
    """Write a December parquet plus qualification manifest, allowing targeted corruption."""

    from flight_delay.data.manifest import write_manifest
    from flight_delay.data.prepare import OUTPUT_COLUMNS

    path = root / QUALIFICATION_DECEMBER_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, OUTPUT_COLUMNS].to_parquet(path, engine="pyarrow", index=False)
    payload: dict[str, object] = {
        "schema_version": 1,
        "december_2025_materialized": True,
        "january_may_2026_referenced": False,
        "tracked_development_manifest_mutated": False,
        "parquet_files": {
            "v3_december": {
                "filename": path.name,
                "byte_size": path.stat().st_size,
                "row_count": len(frame),
                "sha256": "0" * 64,
            }
        },
    }
    payload.update(manifest_overrides)
    write_manifest(root / QUALIFICATION_DECEMBER_MANIFEST, payload)
    return path


def _december_frame() -> pd.DataFrame:
    from tests.conftest import make_v3_frame

    return make_v3_frame(start="2025-12-01", end="2025-12-31")


def _state():
    from flight_delay.modeling.v3.features import build_v3_historical_state
    from tests.conftest import make_v3_frame

    return build_v3_historical_state(
        make_v3_frame(start="2024-01-01", end="2025-10-31"), as_of="2025-10-31"
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"december_2025_materialized": False}, "does not declare December materialization"),
        ({"january_may_2026_referenced": True}, "never be referenced"),
        ({"tracked_development_manifest_mutated": True}, "must not mutate the tracked"),
    ],
)
def test_qualification_manifest_guards(target_root: Path, override: dict, message: str) -> None:
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    _write_qualification(target_root, _december_frame(), **override)
    with pytest.raises(V3DataGuardError, match=message):
        load_december_features(target_root, state=_state(), verify_source_hash=False)


def test_a_missing_materialized_parquet_is_refused(target_root: Path) -> None:
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    path = _write_qualification(target_root, _december_frame())
    path.unlink()
    with pytest.raises(V3DataGuardError, match="materialized December parquet is missing"):
        load_december_features(target_root, state=_state(), verify_source_hash=False)


def test_a_size_or_hash_mismatch_is_refused(target_root: Path) -> None:
    from flight_delay.data.manifest import write_manifest
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    _write_qualification(target_root, _december_frame())
    manifest_path = target_root / QUALIFICATION_DECEMBER_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("manifest_digest", None)
    payload["parquet_files"]["v3_december"]["byte_size"] += 1
    write_manifest(manifest_path, payload)
    with pytest.raises(V3DataGuardError, match="size mismatch"):
        load_december_features(target_root, state=_state(), verify_source_hash=False)

    payload["parquet_files"]["v3_december"]["byte_size"] -= 1
    payload.pop("manifest_digest", None)
    write_manifest(manifest_path, payload)
    with pytest.raises(V3DataGuardError, match="SHA256 mismatch"):
        load_december_features(target_root, state=_state(), verify_source_hash=True)


def test_a_reader_leaking_non_december_rows_is_refused(target_root: Path) -> None:
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    _write_qualification(target_root, _december_frame())
    real = pd.read_parquet

    def leaky(path, **kwargs):
        frame = real(path)
        leaked = frame.head(3).copy()
        leaked["flight_date"] = pd.Timestamp("2026-01-04")
        return pd.concat([frame, leaked], ignore_index=True)

    with pytest.raises(V3DataGuardError, match="outside December 2025"):
        load_december_features(target_root, state=_state(), reader=leaky, verify_source_hash=False)


def test_a_wrong_schema_or_empty_read_is_refused(target_root: Path) -> None:
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    _write_qualification(target_root, _december_frame())
    with pytest.raises(V3DataGuardError, match="schema differs"):
        load_december_features(
            target_root,
            state=_state(),
            reader=lambda path, **kwargs: pd.DataFrame({"nope": [1]}),
            verify_source_hash=False,
        )
    real = pd.read_parquet
    with pytest.raises(V3DataGuardError, match="invalid"):
        load_december_features(
            target_root,
            state=_state(),
            reader=lambda path, **kwargs: real(path).head(0),
            verify_source_hash=False,
        )


def test_a_tracked_manifest_claiming_december_still_blocks_qualification(
    target_root: Path,
) -> None:
    """Defence in depth: a tampered development manifest fails even at qualification time."""

    from flight_delay.data.manifest import write_manifest
    from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features

    _write_qualification(target_root, _december_frame())
    write_manifest(
        target_root / V3_PROCESSED_MANIFEST,
        {
            "schema_version": 1,
            "december_2025_decoded": True,
            "january_may_2026_decoded": False,
            "parquet_files": {},
        },
    )
    with pytest.raises(V3DataGuardError, match="must not be decoded"):
        load_december_features(target_root, state=_state(), verify_source_hash=False)
