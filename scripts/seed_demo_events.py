"""Dry-run by default; seed or clean one labeled DynamoDB Local demo batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import boto3
from dotenv import load_dotenv

from flight_delay.monitoring.demo import cleanup_demo_batch, demo_events, seed_events


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 8, 1))
    parser.add_argument("--table", default=os.getenv("DYNAMODB_TABLE", "flight-delay-events"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-west-2"))
    parser.add_argument("--endpoint-url", default=os.getenv("DYNAMODB_ENDPOINT_URL"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    events = demo_events(batch_id=args.batch_id, count=args.count, start_date=args.start_date)
    if not args.execute and not args.cleanup:
        print(
            json.dumps(
                {
                    "action": "dry-run",
                    "batch_id": args.batch_id,
                    "count": len(events),
                    "first_pk": events[0]["pk"],
                    "all_demo_labeled": all(item["demo_data"] for item in events),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.endpoint_url:
        print(
            "Demo mutation refused: DYNAMODB_ENDPOINT_URL is required for local seeding; "
            "real AWS is out of scope.",
            file=sys.stderr,
        )
        return 2
    resource = boto3.resource("dynamodb", region_name=args.region, endpoint_url=args.endpoint_url)
    table = resource.Table(args.table)
    if args.cleanup:
        result = {"action": "cleanup", "deleted": cleanup_demo_batch(table, args.batch_id)}
    else:
        result = {"action": "seed", "written": seed_events(table, events)}
    print(json.dumps({**result, "batch_id": args.batch_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
