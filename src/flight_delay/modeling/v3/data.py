"""Fail-closed v3 data access that stops before December 2025 and the sealed 2026 test."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.data.manifest import read_manifest
from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.data.prepare_v3 import (
    DECEMBER_SPLIT,
    HISTORY_SPLIT,
    NOVEMBER_SPLIT,
    QUALIFICATION_DECEMBER_MANIFEST,
    QUALIFICATION_DECEMBER_PARQUET,
    V3_PROCESSED_DIRECTORY,
    V3_PROCESSED_MANIFEST,
)
from flight_delay.data.sampling import deterministic_monthly_sample
from flight_delay.modeling.v1_data import ParquetReader
from flight_delay.modeling.v3.features import (
    V3HistoricalState,
    V3TrainingTransform,
    build_v3_historical_state,
    transform_v3_training_rows,
    transform_with_v3_state,
)
from flight_delay.modeling.v3.protocol import load_and_validate_v3_protocol, sha256_file

NOVEMBER_CALIBRATION_END_EXCLUSIVE = "2025-11-16"
FULL_REFIT_END_EXCLUSIVE = "2025-11-01"
MODEL_ROW_START = "2024-02-01"


class V3DataGuardError(RuntimeError):
    """Raised before a prohibited or temporally invalid v3 data operation."""


@dataclass(frozen=True)
class PreparedV3Data:
    search: V3TrainingTransform
    full_refit: V3TrainingTransform
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    calibration_date: pd.Series
    selection_features: pd.DataFrame
    selection_target: pd.Series
    selection_date: pd.Series
    november_state: V3HistoricalState
    raw_history: pd.DataFrame
    raw_november: pd.DataFrame
    lineage: dict[str, Any]


def require_allowed_v3_path(repository_root: Path, path: Path) -> tuple[str, Path]:
    """Resolve only the two v3 development splits, rejecting December and the sealed test."""

    resolved = path.resolve()
    if resolved.name in {"v3_december.parquet", "test.parquet"}:
        raise V3DataGuardError(
            "December 2025 and the sealed January-May 2026 test are prohibited during development"
        )
    processed = (repository_root / V3_PROCESSED_DIRECTORY).resolve()
    allowed = {
        HISTORY_SPLIT: processed / f"{HISTORY_SPLIT}.parquet",
        NOVEMBER_SPLIT: processed / f"{NOVEMBER_SPLIT}.parquet",
    }
    for split, canonical in allowed.items():
        if resolved == canonical:
            return split, canonical
    raise V3DataGuardError("only the canonical v3 history and November splits are allowed")


def _frame_sha256(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _filters(start: str, end_exclusive: str) -> list[tuple[str, str, datetime]]:
    return [
        ("flight_date", ">=", datetime.fromisoformat(start)),
        ("flight_date", "<", datetime.fromisoformat(end_exclusive)),
    ]


def _read_split(
    *,
    root: Path,
    split: str,
    manifest: dict[str, Any],
    start: str,
    end_exclusive: str,
    reader: ParquetReader,
    verify_content_hash: bool,
) -> pd.DataFrame:
    path = (root / V3_PROCESSED_DIRECTORY / f"{split}.parquet").resolve()
    observed_split, path = require_allowed_v3_path(root, path)
    if observed_split != split:
        raise V3DataGuardError("requested split does not match its canonical path")
    if not path.is_file():
        raise V3DataGuardError(f"canonical {split} parquet is missing")
    specification = manifest["parquet_files"][split]
    if path.stat().st_size != specification["byte_size"]:
        raise V3DataGuardError(f"canonical {split} parquet size mismatch")
    if verify_content_hash and sha256_file(path) != specification["sha256"]:
        raise V3DataGuardError(f"canonical {split} parquet SHA256 mismatch")
    frame = reader(path, filters=_filters(start, end_exclusive))
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise V3DataGuardError("v3 processed parquet schema differs from the canonical contract")
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    if frame.empty or dates.isna().any():
        raise V3DataGuardError("filtered v3 data is empty or contains invalid dates")
    if not (dates.ge(start).all() and dates.lt(end_exclusive).all()):
        raise V3DataGuardError("parquet reader returned rows outside the requested period")
    result = frame.copy()
    result["flight_date"] = dates
    return result.sort_values("flight_date", kind="stable").reset_index(drop=True)


def _require_no_december(manifest: dict[str, Any]) -> None:
    if manifest.get("december_2025_decoded") is not False:
        raise V3DataGuardError("December 2025 must not be decoded during v3 development")
    if manifest.get("january_may_2026_decoded") is not False:
        raise V3DataGuardError("January-May 2026 must never be decoded")
    if DECEMBER_SPLIT in (manifest.get("parquet_files") or {}):
        raise V3DataGuardError("the v3 development manifest must not describe December 2025")


def _split_november(
    november: pd.DataFrame, state: V3HistoricalState
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.to_datetime(november["flight_date"], errors="coerce").dt.normalize()
    if november.empty or dates.isna().any():
        raise V3DataGuardError("November rows must be non-empty with valid flight dates")
    if not (dates.ge("2025-11-01").all() and dates.lt("2025-12-01").all()):
        raise V3DataGuardError("development received rows outside November 2025")
    calibration = november.loc[dates.lt(NOVEMBER_CALIBRATION_END_EXCLUSIVE)].copy()
    selection = november.loc[dates.ge(NOVEMBER_CALIBRATION_END_EXCLUSIVE)].copy()
    if calibration.empty or selection.empty:
        raise V3DataGuardError("both frozen November halves must contain rows")
    return (
        transform_with_v3_state(calibration, state),
        calibration["target"].astype(int),
        dates.loc[calibration.index],
        transform_with_v3_state(selection, state),
        selection["target"].astype(int),
        dates.loc[selection.index],
    )


def prepare_development_data(
    repository_root: Path,
    *,
    reader: ParquetReader = pd.read_parquet,
    verify_history_hash: bool = True,
) -> PreparedV3Data:
    """Read 2024-01 through 2025-11 only and build search/refit matrices from full history."""

    root = repository_root.resolve()
    protocol, _lock, protocol_sha = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
        repository_root=root,
    )
    manifest = read_manifest(root / V3_PROCESSED_MANIFEST)
    _require_no_december(manifest)

    history = _read_split(
        root=root,
        split=HISTORY_SPLIT,
        manifest=manifest,
        start="2024-01-01",
        end_exclusive=FULL_REFIT_END_EXCLUSIVE,
        reader=reader,
        verify_content_hash=verify_history_hash,
    )
    november = _read_split(
        root=root,
        split=NOVEMBER_SPLIT,
        manifest=manifest,
        start="2025-11-01",
        end_exclusive="2025-12-01",
        reader=reader,
        verify_content_hash=False,
    )
    history_dates = pd.to_datetime(history["flight_date"], errors="coerce").dt.normalize()
    if history.empty or history_dates.isna().any():
        raise V3DataGuardError("January 2024 through October 2025 source rows are invalid")
    if history_dates.max() >= pd.Timestamp(FULL_REFIT_END_EXCLUSIVE):
        raise V3DataGuardError("v3 history must stop before November 2025")
    model_rows = history.loc[history_dates.ge(MODEL_ROW_START)].copy()
    if model_rows.empty:
        raise V3DataGuardError("v3 refit rows must begin on 2024-02-01")

    # FOLD_4 evaluates November 2025, so the rolling-fold matrix must carry November rows even
    # though no fold ever fits on them: every fit window ends at or before 2025-11-01. The
    # authoritative refit matrix stays strictly February 2024 - October 2025.
    search_source = pd.concat([model_rows, november], ignore_index=True).sort_values(
        "flight_date", kind="stable"
    )
    sampled = deterministic_monthly_sample(
        search_source,
        max_rows_per_month=int(protocol["sampling"]["search_rows_per_month_max"]),
        seed=int(protocol["sampling"]["sample_seed"]),
        date_column="flight_date",
        stratify_column="target",
    )
    search = transform_v3_training_rows(history, sampled)
    full_refit = transform_v3_training_rows(history, model_rows)
    state = build_v3_historical_state(history, as_of="2025-10-31")
    (
        calibration_features,
        calibration_target,
        calibration_date,
        selection_features,
        selection_target,
        selection_date,
    ) = _split_november(november, state)

    lineage = {
        "protocol_sha256": protocol_sha,
        "v3_dataset_manifest_digest": manifest["manifest_digest"],
        "v3_processed_manifest_digest": manifest["manifest_digest"],
        "v3_source_manifest_digest": manifest["source_manifest_digest"],
        "source_row_counts": {
            "history_2024_01_to_2025_10": len(history),
            "november_2025": len(november),
        },
        "eligible_row_counts": {
            "burn_in_january_2024": int(history_dates.lt(MODEL_ROW_START).sum()),
            "model_rows_2024_02_to_2025_10": len(model_rows),
        },
        "model_row_counts": {
            "search": len(sampled),
            "search_november_evaluation_only": int(
                pd.to_datetime(sampled["flight_date"]).ge("2025-11-01").sum()
            ),
            "full_refit": len(model_rows),
            "november_calibration": len(calibration_target),
            "november_selection": len(selection_target),
        },
        "frame_sha256": {
            "history": _frame_sha256(history),
            "search_rows": _frame_sha256(sampled),
            "full_refit_rows": _frame_sha256(model_rows),
            "november": _frame_sha256(november),
        },
        "monthly_search_state_sha256": search.monthly_state_sha256,
        "monthly_full_refit_state_sha256": full_refit.monthly_state_sha256,
        "november_state_sha256": state.sha256,
        "november_state_schema_sha256": state.schema_sha256,
        "december_decoded": False,
        "january_may_2026_accessed": False,
    }
    return PreparedV3Data(
        search=search,
        full_refit=full_refit,
        calibration_features=calibration_features,
        calibration_target=calibration_target,
        calibration_date=calibration_date,
        selection_features=selection_features,
        selection_target=selection_target,
        selection_date=selection_date,
        november_state=state,
        raw_history=history,
        raw_november=november,
        lineage=lineage,
    )


def load_december_features(
    repository_root: Path,
    *,
    state: V3HistoricalState,
    reader: ParquetReader = pd.read_parquet,
    verify_source_hash: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read December from the Git-ignored qualification workspace using the frozen state.

    The tracked development manifest is never consulted or modified here: December lives only in
    ``artifacts/v3/qualification``, so qualification leaves the worktree byte-identical.
    """

    if state.as_of.isoformat() != "2025-10-31":
        raise V3DataGuardError("December must reuse the frozen October-31 feature state")
    root = repository_root.resolve()

    development = root / V3_PROCESSED_MANIFEST
    if development.is_file():
        _require_no_december(read_manifest(development))

    manifest_path = root / QUALIFICATION_DECEMBER_MANIFEST
    if not manifest_path.is_file():
        raise V3DataGuardError(
            "December qualification requires materialized qualification-workspace data"
        )
    manifest = read_manifest(manifest_path)
    if manifest.get("december_2025_materialized") is not True:
        raise V3DataGuardError("qualification manifest does not declare December materialization")
    if manifest.get("january_may_2026_referenced") is not False:
        raise V3DataGuardError("January-May 2026 must never be referenced")
    if manifest.get("tracked_development_manifest_mutated") is not False:
        raise V3DataGuardError("qualification must not mutate the tracked development manifest")

    path = (root / QUALIFICATION_DECEMBER_PARQUET).resolve()
    if not path.is_file():
        raise V3DataGuardError("materialized December parquet is missing")
    specification = manifest["parquet_files"][DECEMBER_SPLIT]
    if path.stat().st_size != specification["byte_size"]:
        raise V3DataGuardError("materialized December parquet size mismatch")
    if verify_source_hash and sha256_file(path) != specification["sha256"]:
        raise V3DataGuardError("materialized December parquet SHA256 mismatch")

    december = reader(path, filters=_filters("2025-12-01", "2026-01-01"))
    if tuple(december.columns) != OUTPUT_COLUMNS:
        raise V3DataGuardError("December parquet schema differs from the canonical contract")
    dates = pd.to_datetime(december["flight_date"], errors="coerce").dt.normalize()
    if december.empty or dates.isna().any():
        raise V3DataGuardError("December qualification rows are invalid")
    if not (dates.ge("2025-12-01").all() and dates.lt("2026-01-01").all()):
        raise V3DataGuardError("December read returned rows outside December 2025")
    december = december.copy()
    december["flight_date"] = dates
    december = december.sort_values("flight_date", kind="stable").reset_index(drop=True)
    dates = december["flight_date"]
    return transform_with_v3_state(december, state), december["target"].astype(int), dates
