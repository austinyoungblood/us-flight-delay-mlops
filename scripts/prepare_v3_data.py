#!/usr/bin/env python3
"""Prepare the uncapped v3 processed dataset, leaving December 2025 sealed by default."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flight_delay.data.prepare_v3 import V3PreparationError, prepare_v3_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-workers", type=int, default=1, help="bounded month-level parallelism"
    )
    parser.add_argument(
        "--december-authorization",
        default=None,
        help="exact qualification authorization required to decode December 2025",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    try:
        result = prepare_v3_dataset(
            root,
            max_workers=arguments.max_workers,
            december_authorization=arguments.december_authorization,
        )
    except V3PreparationError as error:
        print(json.dumps({"status": "failed", "detail": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "prepared",
                "december_decoded": result.december_decoded,
                "manifest_digest": result.manifest["manifest_digest"],
                "split_counts": result.manifest["split_counts"],
                "months": len(result.monthly_stats),
                "runtime_seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
