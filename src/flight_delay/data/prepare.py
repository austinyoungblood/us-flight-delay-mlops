"""Reproducible BTS archive preparation and chronological Parquet output."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pyarrow

from flight_delay.data.download import YearMonth, inspect_zip, sha256_file
from flight_delay.data.manifest import read_manifest, write_manifest
from flight_delay.data.preprocessing import (
    DataQualityError,
    ExclusionCounts,
    construct_binary_target,
    filter_eligible_flights,
    parse_crs_time,
)
from flight_delay.data.sampling import deterministic_monthly_sample
from flight_delay.data.splitting import chronological_split
from flight_delay.features.engineering import derive_schedule_features
from flight_delay.features.leakage import validate_model_features

REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Cancelled",
    "Diverted",
    "ArrDel15",
)

PROCESSED_FEATURES: tuple[str, ...] = (
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

CANDIDATE_A_FEATURES: tuple[str, ...] = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "scheduled_departure_hour",
    "scheduled_arrival_hour",
    "scheduled_departure_minute_bucket",
    "scheduled_arrival_minute_bucket",
    "is_weekend",
)

OUTPUT_COLUMNS: tuple[str, ...] = ("flight_date", *PROCESSED_FEATURES, "target")


@dataclass(frozen=True)
class MonthlyPreparationStats:
    """Stable row and prevalence accounting for one source month."""

    month: str
    source_rows: int
    cancelled_rows: int
    diverted_rows: int
    missing_target_rows: int
    eligible_rows: int
    invalid_schedule_rows: int
    model_eligible_rows: int
    sampled_rows: int
    eligible_target_prevalence: float
    model_eligible_target_prevalence: float
    sampled_target_prevalence: float


@dataclass(frozen=True)
class PreparationResult:
    """Paths, manifest, and stable statistics from dataset preparation."""

    split_paths: Mapping[str, Path]
    manifest: Mapping[str, Any]
    monthly_stats: tuple[MonthlyPreparationStats, ...]


def _validate_codes(frame: pd.DataFrame) -> None:
    patterns = {
        "Reporting_Airline": re.compile(r"^[A-Z0-9]{2}$"),
        "Origin": re.compile(r"^[A-Z]{3}$"),
        "Dest": re.compile(r"^[A-Z]{3}$"),
    }
    for column, pattern in patterns.items():
        invalid = ~frame[column].astype(str).str.fullmatch(pattern)
        if invalid.any():
            raise DataQualityError(
                f"invalid {column} codes at rows {list(frame.index[invalid])[:10]}"
            )


def _is_valid_crs_time(value: object) -> bool:
    try:
        parse_crs_time(value)
    except DataQualityError:
        return False
    return True


def _valid_schedule_mask(frame: pd.DataFrame) -> pd.Series:
    carrier = frame["Reporting_Airline"].astype("string").str.strip().str.upper()
    origin = frame["Origin"].astype("string").str.strip().str.upper()
    destination = frame["Dest"].astype("string").str.strip().str.upper()
    month = pd.to_numeric(frame["Month"], errors="coerce")
    day_of_month = pd.to_numeric(frame["DayofMonth"], errors="coerce")
    day_of_week = pd.to_numeric(frame["DayOfWeek"], errors="coerce")
    elapsed = pd.to_numeric(frame["CRSElapsedTime"], errors="coerce")
    distance = pd.to_numeric(frame["Distance"], errors="coerce")
    flight_date = pd.to_datetime(frame["FlightDate"], errors="coerce")
    return (
        carrier.str.fullmatch(r"[A-Z0-9]{2}", na=False)
        & origin.str.fullmatch(r"[A-Z]{3}", na=False)
        & destination.str.fullmatch(r"[A-Z]{3}", na=False)
        & month.between(1, 12, inclusive="both")
        & month.mod(1).eq(0)
        & day_of_month.between(1, 31, inclusive="both")
        & day_of_month.mod(1).eq(0)
        & day_of_week.between(1, 7, inclusive="both")
        & day_of_week.mod(1).eq(0)
        & elapsed.gt(0)
        & distance.gt(0)
        & flight_date.notna()
        & frame["CRSDepTime"].map(_is_valid_crs_time)
        & frame["CRSArrTime"].map(_is_valid_crs_time)
    )


def _coerce_processed_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["flight_date"] = pd.to_datetime(result["FlightDate"], errors="raise").dt.normalize()
    for column in ("Reporting_Airline", "Origin", "Dest"):
        result[column] = result[column].astype(str).str.strip().str.upper()
    _validate_codes(result)
    integer_columns = (
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_departure_minute_bucket",
        "scheduled_arrival_minute_bucket",
        "is_weekend",
        "target",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not numeric.mod(1).eq(0).all():
            raise DataQualityError(f"{column} must contain finite integer values")
        result[column] = numeric.astype("int16" if column != "target" else "int8")
    for column in ("CRSElapsedTime", "Distance"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or numeric.le(0).any():
            raise DataQualityError(f"{column} must contain positive numeric values")
        result[column] = numeric.astype("float32")
    for column in (
        "scheduled_departure_sin",
        "scheduled_departure_cos",
        "scheduled_arrival_sin",
        "scheduled_arrival_cos",
    ):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    return result.loc[:, OUTPUT_COLUMNS]


def process_month_archive(
    archive_path: Path,
    csv_member: str,
    month: YearMonth,
    *,
    sample_cap: int | None,
    seed: int,
) -> tuple[pd.DataFrame, MonthlyPreparationStats]:
    """Process one monthly ZIP in memory while retaining only safe output columns."""

    dtype = {
        "Reporting_Airline": "string",
        "Origin": "string",
        "Dest": "string",
        "CRSDepTime": "string",
        "CRSArrTime": "string",
    }
    try:
        with ZipFile(archive_path) as archive, archive.open(csv_member) as source:
            raw = pd.read_csv(source, usecols=list(REQUIRED_SOURCE_COLUMNS), dtype=dtype)
    except (OSError, KeyError, ValueError) as error:
        raise DataQualityError(
            f"cannot read required columns from {archive_path.name}: {error}"
        ) from error
    if raw.empty:
        raise DataQualityError(f"month {month.iso()} has no source rows")

    eligibility = filter_eligible_flights(raw)
    if eligibility.frame.empty:
        raise DataQualityError(f"month {month.iso()} has no eligible rows")
    targeted = construct_binary_target(eligibility.frame)
    schedule_mask = _valid_schedule_mask(targeted)
    model_eligible = targeted.loc[schedule_mask].copy()
    invalid_schedule_rows = len(targeted) - len(model_eligible)
    if model_eligible.empty:
        raise DataQualityError(f"month {month.iso()} has no model-eligible rows")
    classes = sorted(model_eligible["target"].unique())
    if classes != [0, 1]:
        raise DataQualityError(
            f"month {month.iso()} must contain both target classes; got {classes}"
        )
    featured = derive_schedule_features(model_eligible)
    processed = _coerce_processed_types(featured)

    observed_months = processed["flight_date"].dt.to_period("M").unique()
    if list(map(str, observed_months)) != [month.iso()]:
        raise DataQualityError(
            f"archive {archive_path.name} contains dates outside {month.iso()}: {observed_months}"
        )
    sampled = deterministic_monthly_sample(
        processed,
        max_rows_per_month=sample_cap,
        seed=seed,
        date_column="flight_date",
        stratify_column="target",
    ).reset_index(drop=True)
    counts: ExclusionCounts = eligibility.counts
    stats = MonthlyPreparationStats(
        month=month.iso(),
        source_rows=counts.input_rows,
        cancelled_rows=counts.cancelled,
        diverted_rows=counts.diverted,
        missing_target_rows=counts.missing_target,
        eligible_rows=counts.eligible_rows,
        invalid_schedule_rows=invalid_schedule_rows,
        model_eligible_rows=len(processed),
        sampled_rows=len(sampled),
        eligible_target_prevalence=float(targeted["target"].mean()),
        model_eligible_target_prevalence=float(processed["target"].mean()),
        sampled_target_prevalence=float(sampled["target"].mean()),
    )
    return sampled.loc[:, OUTPUT_COLUMNS], stats


def _column_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]


def _split_stats(frame: pd.DataFrame) -> dict[str, int | float]:
    return {"row_count": len(frame), "target_prevalence": float(frame["target"].mean())}


def _write_parquet(frame: pd.DataFrame, path: Path, *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression=compression, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "filename": path.name,
        "byte_size": path.stat().st_size,
        "row_count": len(frame),
        "column_schema": _column_schema(frame),
        "sha256": sha256_file(path),
    }


def prepare_dataset(
    *,
    source_manifest_path: Path,
    raw_directory: Path,
    processed_directory: Path,
    processed_manifest_path: Path,
    sample_cap: int | None,
    seed: int,
    train_start: str,
    validation_start: str,
    test_start: str,
    test_end: str,
    compression: str = "zstd",
) -> PreparationResult:
    """Create deterministic chronological Parquet splits from verified monthly archives."""

    validate_model_features(PROCESSED_FEATURES)
    validate_model_features(CANDIDATE_A_FEATURES)
    source_manifest = read_manifest(source_manifest_path)
    source_records = source_manifest.get("files")
    if not isinstance(source_records, list) or not source_records:
        raise DataQualityError("source manifest must contain non-empty file records")

    monthly_frames: list[pd.DataFrame] = []
    monthly_stats: list[MonthlyPreparationStats] = []
    for record in source_records:
        if not isinstance(record, dict):
            raise DataQualityError("source manifest file records must be objects")
        month = YearMonth(int(record["year"]), int(record["month"]))
        archive_path = raw_directory / str(record["archive_filename"])
        expected_checksum = str(record["sha256"])
        if sha256_file(archive_path) != expected_checksum:
            raise DataQualityError(f"archive checksum mismatch: {archive_path.name}")
        inspection = inspect_zip(archive_path)
        if inspection.selected_csv_member != record["selected_csv_member"]:
            raise DataQualityError(f"selected CSV member changed for {archive_path.name}")
        frame, stats = process_month_archive(
            archive_path,
            inspection.selected_csv_member,
            month,
            sample_cap=sample_cap,
            seed=seed,
        )
        monthly_frames.append(frame)
        monthly_stats.append(stats)

    combined = pd.concat(monthly_frames, ignore_index=True)
    split = chronological_split(
        combined,
        train_start=train_start,
        validation_start=validation_start,
        test_start=test_start,
        test_end=test_end,
        date_column="flight_date",
    )
    splits = {"train": split.train, "validation": split.validation, "test": split.test}
    if sum(len(frame) for frame in splits.values()) != len(combined):
        raise DataQualityError("sampled rows exist outside the declared temporal window")
    if any(frame.empty for frame in splits.values()):
        raise DataQualityError("every chronological split must contain rows")

    split_paths = {
        name: processed_directory / f"{name}.parquet" for name in ("train", "validation", "test")
    }
    for name, frame in splits.items():
        _write_parquet(frame.loc[:, OUTPUT_COLUMNS], split_paths[name], compression=compression)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_manifest_digest": source_manifest["manifest_digest"],
        "preprocessing": {
            "monthly_sample_cap": sample_cap,
            "random_seed": seed,
            "sampling": "deterministic_class_stratified_after_eligibility",
            "parquet_engine": "pyarrow",
            "parquet_compression": compression,
        },
        "split_boundaries": {
            "train": {"start": train_start, "end_exclusive": validation_start},
            "validation": {"start": validation_start, "end_exclusive": test_start},
            "test": {"start": test_start, "end_exclusive": test_end},
        },
        "safe_model_feature_schema": list(PROCESSED_FEATURES),
        "candidate_a_feature_schema": list(CANDIDATE_A_FEATURES),
        "target_name": "target",
        "monthly_counts": [asdict(stats) for stats in monthly_stats],
        "split_counts": {name: _split_stats(frame) for name, frame in splits.items()},
        "parquet_files": {
            name: _parquet_record(split_paths[name], frame) for name, frame in splits.items()
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    manifest = write_manifest(processed_manifest_path, payload)
    return PreparationResult(
        split_paths=split_paths,
        manifest=manifest,
        monthly_stats=tuple(monthly_stats),
    )
