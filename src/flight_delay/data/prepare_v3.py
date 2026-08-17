"""Uncapped v3 dataset preparation that refuses to decode December during development.

V3 needs full eligible prior history for its historical state and all eligible model rows for the
authoritative refit, so no monthly sample cap is applied here. Runtime is controlled downstream by
the 50,000-row-per-month candidate-search cap instead.

The v0/v1/v2 processed dataset and its manifests are never touched; v3 writes beside them.
"""

from __future__ import annotations

import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow

from flight_delay.data.download import YearMonth, inspect_zip, sha256_file
from flight_delay.data.manifest import read_manifest, write_manifest
from flight_delay.data.prepare import (
    OUTPUT_COLUMNS,
    MonthlyPreparationStats,
    process_month_archive,
)
from flight_delay.data.preprocessing import DataQualityError
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.v3.protocol import V3_FEATURES

V3_PROCESSED_DIRECTORY = Path("data/processed_v3")
V3_SOURCE_MANIFEST = Path("data/manifests/v3_source_manifest.json")
V3_PROCESSED_MANIFEST = Path("data/manifests/v3_processed_manifest.json")

HISTORY_SPLIT = "v3_history"
NOVEMBER_SPLIT = "v3_november"
DECEMBER_SPLIT = "v3_december"

SPLIT_BOUNDARIES: dict[str, tuple[str, str]] = {
    HISTORY_SPLIT: ("2024-01-01", "2025-11-01"),
    NOVEMBER_SPLIT: ("2025-11-01", "2025-12-01"),
    DECEMBER_SPLIT: ("2025-12-01", "2026-01-01"),
}
DEVELOPMENT_SPLITS: tuple[str, ...] = (HISTORY_SPLIT, NOVEMBER_SPLIT)
DECEMBER_AUTHORIZATION = "december-2025-qualification-authorized"


class V3PreparationError(DataQualityError):
    """Raised when v3 preparation would violate a temporal or governance boundary."""


@dataclass(frozen=True)
class V3PreparationResult:
    split_paths: dict[str, Path]
    manifest: dict[str, Any]
    monthly_stats: tuple[MonthlyPreparationStats, ...]
    december_decoded: bool


def split_for_month(month: YearMonth, *, include_december: bool) -> str | None:
    """Return the split a source month belongs to, or ``None`` when it must not be decoded."""

    if month.year == 2024 or (month.year == 2025 and month.month <= 10):
        return HISTORY_SPLIT
    if month.year == 2025 and month.month == 11:
        return NOVEMBER_SPLIT
    if month.year == 2025 and month.month == 12:
        return DECEMBER_SPLIT if include_december else None
    raise V3PreparationError(
        f"month {month.iso()} lies outside the v3 window; January-May 2026 is prohibited"
    )


def _prepare_month(payload: tuple[str, str, int, int, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive_path, csv_member, year, month, expected_checksum = payload
    path = Path(archive_path)
    if sha256_file(path) != expected_checksum:
        raise V3PreparationError(f"archive checksum mismatch: {path.name}")
    frame, stats = process_month_archive(
        path, csv_member, YearMonth(year, month), sample_cap=None, seed=42
    )
    if stats.sampled_rows != stats.model_eligible_rows:
        raise V3PreparationError("v3 preparation must retain every model-eligible row")
    return frame, asdict(stats)


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
        "column_schema": [
            {"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns
        ],
        "sha256": sha256_file(path),
    }


def prepare_v3_dataset(
    repository_root: Path,
    *,
    source_manifest_path: Path | None = None,
    raw_directory: Path | None = None,
    processed_directory: Path | None = None,
    processed_manifest_path: Path | None = None,
    compression: str = "zstd",
    max_workers: int = 1,
    december_authorization: str | None = None,
) -> V3PreparationResult:
    """Decode the v3 archives into uncapped splits, leaving December sealed by default."""

    validate_model_features(V3_FEATURES)
    root = repository_root.resolve()
    source_manifest_path = source_manifest_path or root / V3_SOURCE_MANIFEST
    raw_directory = raw_directory or root / "data/raw/bts_reporting_carrier"
    processed_directory = processed_directory or root / V3_PROCESSED_DIRECTORY
    processed_manifest_path = processed_manifest_path or root / V3_PROCESSED_MANIFEST

    include_december = december_authorization == DECEMBER_AUTHORIZATION
    if december_authorization is not None and not include_december:
        raise V3PreparationError("December decoding requires the exact qualification authorization")

    source_manifest = read_manifest(source_manifest_path)
    records = source_manifest.get("files")
    if not isinstance(records, list) or not records:
        raise V3PreparationError("v3 source manifest must contain non-empty file records")
    if any(int(record["year"]) >= 2026 for record in records):
        raise V3PreparationError("the v3 source manifest must not reference any 2026 archive")

    jobs: list[tuple[str, str, int, int, str]] = []
    job_splits: list[str] = []
    for record in sorted(records, key=lambda row: (int(row["year"]), int(row["month"]))):
        month = YearMonth(int(record["year"]), int(record["month"]))
        split = split_for_month(month, include_december=include_december)
        if split is None:
            continue
        archive_path = raw_directory / str(record["archive_filename"])
        inspection = inspect_zip(archive_path)
        if inspection.selected_csv_member != record["selected_csv_member"]:
            raise V3PreparationError(f"selected CSV member changed for {archive_path.name}")
        jobs.append(
            (
                str(archive_path),
                inspection.selected_csv_member,
                month.year,
                month.month,
                str(record["sha256"]),
            )
        )
        job_splits.append(split)

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            outputs = list(pool.map(_prepare_month, jobs))
    else:
        outputs = [_prepare_month(job) for job in jobs]

    frames: dict[str, list[pd.DataFrame]] = {}
    monthly_stats: list[MonthlyPreparationStats] = []
    for split, (frame, stats) in zip(job_splits, outputs, strict=True):
        frames.setdefault(split, []).append(frame)
        monthly_stats.append(MonthlyPreparationStats(**stats))

    split_paths: dict[str, Path] = {}
    split_frames: dict[str, pd.DataFrame] = {}
    for split, parts in frames.items():
        combined = (
            pd.concat(parts, ignore_index=True)
            .sort_values("flight_date", kind="stable")
            .reset_index(drop=True)
        )
        start, end_exclusive = SPLIT_BOUNDARIES[split]
        dates = pd.to_datetime(combined["flight_date"])
        if combined.empty or not (dates.ge(start).all() and dates.lt(end_exclusive).all()):
            raise V3PreparationError(f"{split} contains rows outside {start}/{end_exclusive}")
        path = processed_directory / f"{split}.parquet"
        _write_parquet(combined.loc[:, OUTPUT_COLUMNS], path, compression=compression)
        split_paths[split] = path
        split_frames[split] = combined

    if set(split_paths) < set(DEVELOPMENT_SPLITS):
        raise V3PreparationError("v3 preparation must produce both development splits")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "us-flight-delay-v3-seasonal-temporal-generalization-v1",
        "source_manifest_digest": source_manifest["manifest_digest"],
        "preprocessing": {
            "monthly_sample_cap": None,
            "uncapped_full_eligible_rows": True,
            "random_seed": 42,
            "sampling": "none_all_model_eligible_rows_retained",
            "parquet_engine": "pyarrow",
            "parquet_compression": compression,
        },
        "split_boundaries": {
            split: {"start": start, "end_exclusive": end_exclusive}
            for split, (start, end_exclusive) in SPLIT_BOUNDARIES.items()
            if split in split_paths
        },
        "december_2025_decoded": include_december,
        "january_may_2026_decoded": False,
        "v3_feature_schema": list(V3_FEATURES),
        "safe_model_feature_schema": list(OUTPUT_COLUMNS),
        "target_name": "target",
        "monthly_counts": [asdict(stats) for stats in monthly_stats],
        "split_counts": {
            split: {
                "row_count": len(frame),
                "target_prevalence": float(frame["target"].mean()),
            }
            for split, frame in split_frames.items()
        },
        "parquet_files": {
            split: _parquet_record(split_paths[split], split_frames[split])
            for split in split_paths
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    manifest = write_manifest(processed_manifest_path, payload)
    return V3PreparationResult(
        split_paths=split_paths,
        manifest=manifest,
        monthly_stats=tuple(monthly_stats),
        december_decoded=include_december,
    )
