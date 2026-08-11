#!/usr/bin/env python3
"""Run the frozen end-to-end smoke sequence in local or live mode."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from flight_delay.deployment import SmokeError, SmokeRunner, load_and_validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "live"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "deploy/deployment_manifest.json")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--traveler-url", required=True)
    parser.add_argument("--monitor-url", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--dynamodb-endpoint-url")
    parser.add_argument("--verify-dynamodb", action="store_true")
    parser.add_argument(
        "--seed-demo-batch",
        help="Explicitly seed a uniquely named labeled monitoring batch after smoke passes.",
    )
    parser.add_argument("--demo-count", type=int, default=30)
    parser.add_argument(
        "--allow-cache-miss",
        action="store_true",
        help="Record cache state without requiring the second prediction to hit cache.",
    )
    return parser.parse_args()


def _dynamodb_table(args: argparse.Namespace, manifest: dict[str, Any]) -> Any:
    if args.mode == "local":
        if not args.dynamodb_endpoint_url or not re.fullmatch(
            r"http://(?:127\.0\.0\.1|localhost):\d+", args.dynamodb_endpoint_url
        ):
            raise SmokeError("local DynamoDB operations require an explicit loopback endpoint")
    elif args.dynamodb_endpoint_url:
        raise SmokeError("live mode forbids a DynamoDB endpoint override")
    import boto3

    credentials = (
        {"aws_access_key_id": "local", "aws_secret_access_key": "local"}
        if args.mode == "local"
        else {}
    )
    resource = boto3.resource(
        "dynamodb",
        region_name=args.region,
        endpoint_url=args.dynamodb_endpoint_url,
        **credentials,
    )
    return resource.Table(manifest["dynamodb"]["table_name"])


def _verify_dynamodb(table: Any, result: dict[str, Any]) -> dict[str, Any]:
    for prediction_id in result["predictions"]["ids"]:
        item = table.get_item(Key={"pk": f"PREDICTION#{prediction_id}"}, ConsistentRead=True).get(
            "Item"
        )
        if not item or item.get("prediction_id") != prediction_id:
            raise SmokeError("direct DynamoDB prediction verification failed")
    feedback_id = result["feedback"]["prediction_id"]
    feedback_item = table.get_item(
        Key={"pk": f"PREDICTION#{feedback_id}"}, ConsistentRead=True
    ).get("Item")
    if not feedback_item or not feedback_item.get("feedback"):
        raise SmokeError("direct DynamoDB feedback verification failed")
    return {"predictions": 2, "feedback": True}


def _seed_demo(table: Any, batch_id: str, count: int) -> dict[str, Any]:
    if not re.fullmatch(r"brief08-[0-9]{8}T[0-9]{6}Z-[a-z0-9-]{1,24}", batch_id):
        raise SmokeError("demo batch ID must be unique and follow the documented format")
    from flight_delay.monitoring.demo import demo_events, seed_events

    events = demo_events(batch_id=batch_id, count=count, start_date=datetime.now(UTC).date())
    return {"batch_id": batch_id, "written": seed_events(table, events), "demo_data": True}


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        print("smoke failed: timeout must be positive", file=sys.stderr)
        return 2
    try:
        manifest = load_and_validate_manifest(
            args.manifest,
            release_decision_path=ROOT / "release/release_decision.json",
            selection_lock_path=ROOT / "release/selection_lock.json",
            repository_root=ROOT,
        )
        with httpx.Client(timeout=args.timeout_seconds, follow_redirects=False) as client:
            result = SmokeRunner(client, manifest).run(
                api_base_url=args.api_url,
                traveler_base_url=args.traveler_url,
                monitor_base_url=args.monitor_url,
                allow_cache_miss=args.allow_cache_miss,
            )
        table = None
        if args.verify_dynamodb or args.seed_demo_batch:
            table = _dynamodb_table(args, manifest)
        if args.verify_dynamodb:
            result["direct_dynamodb"] = _verify_dynamodb(table, result)
        if args.seed_demo_batch:
            result["demo_batch"] = _seed_demo(table, args.seed_demo_batch, args.demo_count)
    except (OSError, httpx.HTTPError, SmokeError, ValueError) as error:
        print(f"smoke failed: {error}", file=sys.stderr)
        return 1

    summary = {"schema_version": 1, "mode": args.mode, **result}
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
