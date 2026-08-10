"""Prepare deterministic chronological Parquet splits from verified BTS archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from flight_delay.data.prepare import prepare_dataset
from flight_delay.data.preprocessing import DataQualityError


def _load_data_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DataQualityError(f"cannot read configuration {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise DataQualityError("configuration must contain a data mapping")
    return payload["data"]


def build_parser() -> argparse.ArgumentParser:
    """Build the preparation command-line parser."""

    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe BTS train, validation, and sealed test Parquet splits."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run data preparation and print stable non-model summary metadata."""

    args = build_parser().parse_args(argv)
    try:
        config = _load_data_config(args.config)
        boundaries = config["splits"]
        result = prepare_dataset(
            source_manifest_path=Path(str(config["source_manifest"])),
            raw_directory=Path(str(config["raw_directory"])),
            processed_directory=Path(str(config["processed_directory"])),
            processed_manifest_path=Path(str(config["processed_manifest"])),
            sample_cap=config.get("monthly_sample_cap"),
            seed=int(config["random_seed"]),
            train_start=str(boundaries["train_start"]),
            validation_start=str(boundaries["validation_start"]),
            test_start=str(boundaries["test_start"]),
            test_end=str(boundaries["test_end"]),
            compression=str(config.get("parquet_compression", "zstd")),
        )
    except (KeyError, TypeError, ValueError, DataQualityError) as error:
        print(f"data preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "manifest_digest": result.manifest["manifest_digest"],
                "split_counts": result.manifest["split_counts"],
                "parquet_files": result.manifest["parquet_files"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
