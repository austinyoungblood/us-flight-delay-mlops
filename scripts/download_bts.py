"""Download and validate official BTS monthly archives."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from flight_delay.data.download import (
    DownloadBatchError,
    DownloadError,
    YearMonth,
    download_archives,
    inclusive_month_range,
)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DownloadError(f"cannot read configuration {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise DownloadError("configuration must contain a data mapping")
    return payload["data"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Download official BTS Reporting Carrier archives with stable provenance."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing archive instead of requiring a matching manifest checksum.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sequential downloader and emit a gitignored telemetry report."""

    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        start = YearMonth.parse(str(config["start_month"]))
        end = YearMonth.parse(str(config["end_month"]))
        months = inclusive_month_range(start, end)
        download = config.get("download", {})
        summary, manifest = download_archives(
            months,
            Path(str(config["raw_directory"])),
            Path(str(config["source_manifest"])),
            expected_archive_count=int(config["expected_archive_count"]),
            overwrite=args.overwrite,
            timeout=(
                float(download.get("connect_timeout_seconds", 10)),
                float(download.get("read_timeout_seconds", 120)),
            ),
            max_attempts=int(download.get("max_attempts", 3)),
            backoff_seconds=float(download.get("backoff_seconds", 2)),
        )
    except (KeyError, TypeError, ValueError, DownloadError) as error:
        if isinstance(error, DownloadBatchError):
            print(
                json.dumps(
                    {
                        "downloaded": error.summary.downloaded,
                        "skipped": error.summary.skipped,
                        "failed": error.summary.failed,
                        "failures": [
                            result.error
                            for result in error.summary.results
                            if result.status == "failed"
                        ],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        else:
            print(f"download failed: {error}", file=sys.stderr)
        return 1

    telemetry_path = Path(str(config["raw_directory"])) / "download_report.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "downloaded": summary.downloaded,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "aggregate_archive_bytes": summary.total_bytes,
                "manifest_digest": manifest["manifest_digest"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "downloaded": summary.downloaded,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "aggregate_archive_bytes": summary.total_bytes,
                "manifest_digest": manifest["manifest_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
