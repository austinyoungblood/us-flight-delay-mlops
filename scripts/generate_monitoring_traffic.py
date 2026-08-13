"""Generate bounded synthetic inference traffic through the real prediction API."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from flight_delay.monitoring.traffic import (
    TrafficPlan,
    run_monitoring_traffic,
    validate_api_base_url,
    write_audit_summary,
)
from flight_delay.ui import FlightDelayApiClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate-per-second", type=float, default=2.0)
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=(datetime.now(UTC) + timedelta(days=1)).date(),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/monitoring-load/summary.json")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Send POST /predict requests. Without this flag, no network request is made.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_url = validate_api_base_url(args.api_base_url)
        plan = TrafficPlan(
            count=args.count,
            seed=args.seed,
            rate_per_second=args.rate_per_second,
            start_date=args.start_date,
        )
    except ValueError as error:
        print(f"traffic generation refused: {error}", file=sys.stderr)
        return 2

    client = None
    try:
        if args.apply:
            client = FlightDelayApiClient(base_url)
        audit = run_monitoring_traffic(
            plan,
            apply=args.apply,
            sender=client.predict if client is not None else None,
        )
        write_audit_summary(args.output, audit)
    finally:
        if client is not None:
            client.close()
    print(json.dumps({**audit.model_dump(), "audit_path": str(args.output)}, sort_keys=True))
    return 1 if audit.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
