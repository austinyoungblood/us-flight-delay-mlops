"""Idempotently create or validate the DynamoDB table contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

INDEX_NAME = "event-date-created-at-index"


class TableContractError(RuntimeError):
    """The existing table is incompatible with the required schema."""


def desired_request(table_name: str) -> dict[str, Any]:
    """Return the non-destructive table creation contract."""

    return {
        "TableName": table_name,
        "BillingMode": "PAY_PER_REQUEST",
        "KeySchema": [{"AttributeName": "pk", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "event_date", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": INDEX_NAME,
                "KeySchema": [
                    {"AttributeName": "event_date", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    }


def validate_description(description: dict[str, Any], table_name: str) -> None:
    """Reject an existing table whose key, billing mode, or GSI differs."""

    table = description["Table"]
    expected = desired_request(table_name)
    if table.get("KeySchema") != expected["KeySchema"]:
        raise TableContractError("existing table partition key is incompatible")
    if table.get("BillingModeSummary", {}).get("BillingMode") != "PAY_PER_REQUEST":
        raise TableContractError("existing table billing mode is incompatible")
    attributes = {
        item["AttributeName"]: item["AttributeType"] for item in table["AttributeDefinitions"]
    }
    if attributes != {"pk": "S", "event_date": "S", "created_at": "S"}:
        raise TableContractError("existing table attribute definitions are incompatible")
    indexes = {item["IndexName"]: item for item in table.get("GlobalSecondaryIndexes", [])}
    index = indexes.get(INDEX_NAME)
    if not index or index.get("KeySchema") != expected["GlobalSecondaryIndexes"][0]["KeySchema"]:
        raise TableContractError("existing table GSI key schema is incompatible")
    if index.get("Projection", {}).get("ProjectionType") != "ALL":
        raise TableContractError("existing table GSI projection is incompatible")


def provision(*, client: Any, table_name: str, dry_run: bool = False) -> dict[str, Any]:
    """Create or validate the table, waiting for ACTIVE after a real creation."""

    request = desired_request(table_name)
    if dry_run:
        return {"action": "dry-run", "request": request}
    try:
        description = client.describe_table(TableName=table_name)
        action = "validated"
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        client.create_table(**request)
        client.get_waiter("table_exists").wait(TableName=table_name)
        description = client.describe_table(TableName=table_name)
        action = "created"
    validate_description(description, table_name)
    if description["Table"].get("TableStatus") != "ACTIVE":
        client.get_waiter("table_exists").wait(TableName=table_name)
        description = client.describe_table(TableName=table_name)
    if description["Table"].get("TableStatus") != "ACTIVE":
        raise TableContractError("table did not become ACTIVE")
    return {
        "action": action,
        "table_name": table_name,
        "status": "ACTIVE",
        "billing_mode": "PAY_PER_REQUEST",
        "gsi": INDEX_NAME,
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=os.getenv("DYNAMODB_TABLE", "flight-delay-events"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-west-2"))
    parser.add_argument("--endpoint-url", default=os.getenv("DYNAMODB_ENDPOINT_URL"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        client = (
            boto3.client("dynamodb", region_name=args.region, endpoint_url=args.endpoint_url)
            if not args.dry_run
            else None
        )
        result = provision(client=client, table_name=args.table, dry_run=args.dry_run)
    except (ClientError, TableContractError) as error:
        print(f"DynamoDB provisioning failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
