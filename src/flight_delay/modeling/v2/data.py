"""Fail-closed v2 data preparation that stops before December and the historical test."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.data.manifest import read_manifest
from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.data.sampling import deterministic_monthly_sample
from flight_delay.modeling.v1_data import (
    ParquetReader,
    V1DataGuardError,
    require_allowed_v1_path,
)
from flight_delay.modeling.v2.features import (
    HistoricalState,
    TrainingTransform,
    build_historical_state,
    transform_training_rows,
    transform_with_state,
)
from flight_delay.modeling.v2.protocol import load_and_validate_v2_protocol, sha256_file


class V2DataGuardError(RuntimeError):
    """Raised before a prohibited or temporally invalid v2 data operation."""


@dataclass(frozen=True)
class PreparedV2Data:
    search: TrainingTransform
    full_refit: TrainingTransform
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    calibration_date: pd.Series
    selection_features: pd.DataFrame
    selection_target: pd.Series
    selection_date: pd.Series
    november_state: HistoricalState
    raw_train: pd.DataFrame
    raw_november: pd.DataFrame
    lineage: dict[str, Any]


def require_allowed_v2_path(repository_root: Path, path: Path) -> tuple[str, Path]:
    try:
        return require_allowed_v1_path(repository_root, path)
    except V1DataGuardError as error:
        raise V2DataGuardError(str(error)) from error


def _frame_sha256(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _filters(start: str, end_exclusive: str) -> list[tuple[str, str, datetime]]:
    return [
        ("flight_date", ">=", datetime.fromisoformat(start)),
        ("flight_date", "<", datetime.fromisoformat(end_exclusive)),
    ]


def _read_period(
    *,
    root: Path,
    path: Path,
    split: str,
    manifest: dict[str, Any],
    start: str,
    end_exclusive: str,
    reader: ParquetReader,
    verify_content_hash: bool,
) -> pd.DataFrame:
    observed_split, canonical = require_allowed_v2_path(root, path)
    if observed_split != split:
        raise V2DataGuardError("requested split does not match its canonical path")
    if not canonical.is_file():
        raise V2DataGuardError(f"canonical {split} parquet is missing")
    specification = manifest["parquet_files"][split]
    if canonical.stat().st_size != specification["byte_size"]:
        raise V2DataGuardError(f"canonical {split} parquet size mismatch")
    if verify_content_hash and sha256_file(canonical) != specification["sha256"]:
        raise V2DataGuardError(f"canonical {split} parquet SHA256 mismatch")
    frame = reader(canonical, filters=_filters(start, end_exclusive))
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise V2DataGuardError("processed parquet schema differs from the canonical contract")
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    if frame.empty or dates.isna().any():
        raise V2DataGuardError("filtered v2 data is empty or contains invalid dates")
    if not (dates.ge(start).all() and dates.lt(end_exclusive).all()):
        raise V2DataGuardError("parquet reader returned rows outside the requested period")
    result = frame.copy()
    result["flight_date"] = dates
    return result.sort_values("flight_date", kind="stable").reset_index(drop=True)


def _split_november(
    november: pd.DataFrame, state: HistoricalState
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.to_datetime(november["flight_date"], errors="coerce").dt.normalize()
    if november.empty or dates.isna().any():
        raise V2DataGuardError("November rows must be non-empty with valid flight dates")
    if not (dates.ge("2025-11-01").all() and dates.lt("2025-12-01").all()):
        raise V2DataGuardError("development received rows outside November 2025")
    calibration = november.loc[dates.lt("2025-11-16")].copy()
    selection = november.loc[dates.ge("2025-11-16")].copy()
    if calibration.empty or selection.empty:
        raise V2DataGuardError("both frozen November halves must contain rows")
    return (
        transform_with_state(calibration, state),
        calibration["target"].astype(int),
        dates.loc[calibration.index],
        transform_with_state(selection, state),
        selection["target"].astype(int),
        dates.loc[selection.index],
    )


def prepare_development_data(
    repository_root: Path,
    *,
    reader: ParquetReader = pd.read_parquet,
    verify_train_hash: bool = True,
) -> PreparedV2Data:
    """Read January-November only and build sample/full matrices from full prior history."""

    root = repository_root.resolve()
    protocol, _lock, protocol_sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    manifest = read_manifest(root / "data/manifests/processed_manifest.json")
    processed = root / "data/processed"
    train = _read_period(
        root=root,
        path=processed / "train.parquet",
        split="train",
        manifest=manifest,
        start="2025-01-01",
        end_exclusive="2025-11-01",
        reader=reader,
        verify_content_hash=verify_train_hash,
    )
    november = _read_period(
        root=root,
        path=processed / "validation.parquet",
        split="validation",
        manifest=manifest,
        start="2025-11-01",
        end_exclusive="2025-12-01",
        reader=reader,
        verify_content_hash=False,
    )
    train_dates = pd.to_datetime(train["flight_date"], errors="coerce").dt.normalize()
    if train.empty or train_dates.isna().any():
        raise V2DataGuardError("January-October source rows are invalid")
    model_rows = train.loc[train_dates.ge("2025-02-01")].copy()
    if model_rows.empty or train_dates.max() >= pd.Timestamp("2025-11-01"):
        raise V2DataGuardError("v2 refit rows must be February-October 2025")
    sampled = deterministic_monthly_sample(
        model_rows,
        max_rows_per_month=int(protocol["sampling"]["search_rows_per_month_max"]),
        seed=int(protocol["sampling"]["sample_seed"]),
        date_column="flight_date",
        stratify_column="target",
    )
    search = transform_training_rows(train, sampled)
    full_refit = transform_training_rows(train, model_rows)
    state = build_historical_state(train, as_of="2025-10-31")
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
        "dataset_manifest_digest": manifest["manifest_digest"],
        "source_row_counts": {
            "january_october": len(train),
            "november": len(november),
        },
        "eligible_row_counts": {
            "burn_in_january": int(train_dates.lt("2025-02-01").sum()),
            "february_october": len(model_rows),
        },
        "model_row_counts": {
            "search": len(sampled),
            "full_refit": len(model_rows),
            "november_calibration": len(calibration_target),
            "november_selection": len(selection_target),
        },
        "frame_sha256": {
            "january_october": _frame_sha256(train),
            "search_rows": _frame_sha256(sampled),
            "full_refit_rows": _frame_sha256(model_rows),
            "november": _frame_sha256(november),
        },
        "monthly_search_state_sha256": search.monthly_state_sha256,
        "monthly_full_refit_state_sha256": full_refit.monthly_state_sha256,
        "november_state_sha256": state.sha256,
        "november_state_schema_sha256": state.schema_sha256,
    }
    return PreparedV2Data(
        search=search,
        full_refit=full_refit,
        calibration_features=calibration_features,
        calibration_target=calibration_target,
        calibration_date=calibration_date,
        selection_features=selection_features,
        selection_target=selection_target,
        selection_date=selection_date,
        november_state=state,
        raw_train=train,
        raw_november=november,
        lineage=lineage,
    )


def load_december_features(
    repository_root: Path,
    *,
    state: HistoricalState,
    reader: ParquetReader = pd.read_parquet,
    verify_source_hash: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read December only from the qualification path and reuse the October-31 state."""

    if state.as_of.isoformat() != "2025-10-31":
        raise V2DataGuardError("December must reuse the frozen October-31 feature state")
    root = repository_root.resolve()
    manifest = read_manifest(root / "data/manifests/processed_manifest.json")
    december = _read_period(
        root=root,
        path=root / "data/processed/validation.parquet",
        split="validation",
        manifest=manifest,
        start="2025-12-01",
        end_exclusive="2026-01-01",
        reader=reader,
        verify_content_hash=verify_source_hash,
    )
    dates = pd.to_datetime(december["flight_date"], errors="coerce").dt.normalize()
    if december.empty or dates.isna().any():
        raise V2DataGuardError("December qualification rows are invalid")
    return transform_with_state(december, state), december["target"].astype(int), dates
