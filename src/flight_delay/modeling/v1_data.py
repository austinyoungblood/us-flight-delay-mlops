"""Fail-closed canonical data access and feature adaptation for governed v1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flight_delay.data.manifest import read_manifest
from flight_delay.data.prepare import OUTPUT_COLUMNS
from flight_delay.features.leakage import FORBIDDEN_FEATURES, validate_model_features
from flight_delay.modeling.v1_protocol import load_and_validate_v1_protocol, sha256_file

V1_FEATURES: tuple[str, ...] = (
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "route",
    "scheduled_departure_hour",
    "scheduled_arrival_hour",
    "scheduled_departure_minute_bucket",
    "scheduled_arrival_minute_bucket",
    "is_weekend",
    "scheduled_departure_sin",
    "scheduled_departure_cos",
    "scheduled_arrival_sin",
    "scheduled_arrival_cos",
)
V1_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "route",
)


class V1DataGuardError(RuntimeError):
    """Raised before an ungoverned or temporally invalid data read can proceed."""


@dataclass(frozen=True)
class DevelopmentData:
    train: pd.DataFrame
    november: pd.DataFrame
    manifest: dict[str, Any]
    protocol: dict[str, Any]
    protocol_sha256: str


@dataclass(frozen=True)
class AdaptedV1Frame:
    features: pd.DataFrame
    target: pd.Series
    flight_date: pd.Series


ParquetReader = Callable[..., pd.DataFrame]


def _canonical_paths(repository_root: Path) -> dict[str, Path]:
    processed = (repository_root / "data/processed").resolve()
    return {
        "train": processed / "train.parquet",
        "validation": processed / "validation.parquet",
    }


def require_allowed_v1_path(repository_root: Path, path: Path) -> tuple[str, Path]:
    """Resolve only the two canonical development sources and reject test/arbitrary paths."""

    resolved = path.resolve()
    if resolved.name == "test.parquet":
        raise V1DataGuardError("historical test.parquet access is prohibited")
    for split, canonical in _canonical_paths(repository_root).items():
        if resolved == canonical:
            return split, canonical
    raise V1DataGuardError("only canonical train.parquet and validation.parquet are allowed")


def _validated_context(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    return load_and_validate_v1_protocol(
        repository_root / "configs/v1_experiment_protocol.yaml",
        lock_path=repository_root / "experiments/v1/protocol_lock.json",
        repository_root=repository_root,
    )


def _verify_split_hash(path: Path, split: str, manifest: dict[str, Any]) -> None:
    if not path.is_file():
        raise V1DataGuardError(f"canonical {split} parquet is missing")
    expected = manifest["parquet_files"][split]["sha256"]
    if sha256_file(path) != expected:
        raise V1DataGuardError(f"canonical {split} parquet SHA256 mismatch")


def _period_filters(start: str, end_exclusive: str) -> list[tuple[str, str, datetime]]:
    return [
        ("flight_date", ">=", datetime.fromisoformat(start)),
        ("flight_date", "<", datetime.fromisoformat(end_exclusive)),
    ]


def _read_checked_period(
    *,
    repository_root: Path,
    path: Path,
    split: str,
    manifest: dict[str, Any],
    start: str,
    end_exclusive: str,
    reader: ParquetReader,
) -> pd.DataFrame:
    observed_split, canonical = require_allowed_v1_path(repository_root, path)
    if observed_split != split:
        raise V1DataGuardError("requested split does not match its canonical path")
    _verify_split_hash(canonical, split, manifest)
    frame = reader(canonical, filters=_period_filters(start, end_exclusive))
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise V1DataGuardError("processed parquet schema differs from the canonical contract")
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    if frame.empty or dates.isna().any():
        raise V1DataGuardError("filtered v1 data is empty or contains invalid dates")
    if not (dates.ge(start).all() and dates.lt(end_exclusive).all()):
        raise V1DataGuardError("parquet reader returned rows outside the requested period")
    result = frame.copy()
    result["flight_date"] = dates
    return result.sort_values("flight_date", kind="stable").reset_index(drop=True)


def load_development_data(
    repository_root: Path, *, reader: ParquetReader = pd.read_parquet
) -> DevelopmentData:
    """Read only Jan-Oct training and explicitly filtered November validation rows."""

    root = repository_root.resolve()
    protocol, _lock, protocol_sha = _validated_context(root)
    manifest = read_manifest(root / "data/manifests/processed_manifest.json")
    paths = _canonical_paths(root)
    train = _read_checked_period(
        repository_root=root,
        path=paths["train"],
        split="train",
        manifest=manifest,
        start="2025-01-01",
        end_exclusive="2025-11-01",
        reader=reader,
    )
    november = _read_checked_period(
        repository_root=root,
        path=paths["validation"],
        split="validation",
        manifest=manifest,
        start="2025-11-01",
        end_exclusive="2025-12-01",
        reader=reader,
    )
    return DevelopmentData(train, november, manifest, protocol, protocol_sha)


def load_december_data(
    repository_root: Path, *, reader: ParquetReader = pd.read_parquet
) -> pd.DataFrame:
    """Read only the locked December qualification window from canonical validation."""

    root = repository_root.resolve()
    _validated_context(root)
    manifest = read_manifest(root / "data/manifests/processed_manifest.json")
    return _read_checked_period(
        repository_root=root,
        path=_canonical_paths(root)["validation"],
        split="validation",
        manifest=manifest,
        start="2025-12-01",
        end_exclusive="2026-01-01",
        reader=reader,
    )


def adapt_v1_frame(frame: pd.DataFrame) -> AdaptedV1Frame:
    """Return a copied, chronological, exact-schema CatBoost matrix and binary target."""

    expected = ("flight_date", *V1_FEATURES, "target")
    if tuple(frame.columns) != expected:
        forbidden = set(frame.columns) & FORBIDDEN_FEATURES
        if forbidden:
            raise V1DataGuardError(f"forbidden outcome features entered v1: {sorted(forbidden)}")
        raise V1DataGuardError("v1 input must have the exact canonical columns and order")
    validate_model_features(V1_FEATURES)
    original = frame.copy(deep=True)
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise V1DataGuardError("flight_date must be valid for chronological ordering")
    result = frame.copy(deep=True)
    result["flight_date"] = dates
    result = result.sort_values("flight_date", kind="stable")
    dates = result["flight_date"]
    for column in V1_CATEGORICAL_FEATURES:
        if result[column].isna().any():
            raise V1DataGuardError(f"categorical feature {column} contains missing values")
        values = result[column].astype("string").str.strip()
        if values.eq("").any():
            raise V1DataGuardError(f"categorical feature {column} contains empty values")
        result[column] = values.astype(str)
    numeric_columns = [name for name in V1_FEATURES if name not in V1_CATEGORICAL_FEATURES]
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise V1DataGuardError(f"numeric feature {column} must be finite")
        result[column] = values
    target = pd.to_numeric(result["target"], errors="coerce")
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise V1DataGuardError("target must be binary")
    if not frame.equals(original):
        raise AssertionError("v1 feature adaptation mutated its source frame")
    return AdaptedV1Frame(
        features=result.loc[:, V1_FEATURES],
        target=target.astype(int),
        flight_date=dates.loc[result.index],
    )


def development_period(frame: pd.DataFrame, start: str, end_exclusive: str) -> pd.DataFrame:
    """Select and stably sort an already guarded development period."""

    if start >= "2025-12-01" or end_exclusive > "2025-12-01":
        raise V1DataGuardError("development cannot request December qualification or test data")
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    selected = frame.loc[dates.ge(start) & dates.lt(end_exclusive)].copy()
    if selected.empty:
        raise V1DataGuardError(f"development period [{start}, {end_exclusive}) is empty")
    return selected.sort_values("flight_date", kind="stable")
