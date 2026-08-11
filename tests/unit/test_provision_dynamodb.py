from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from infra.provision_dynamodb import (
    INDEX_NAME,
    TableContractError,
    desired_request,
    provision,
    validate_description,
)


def active_description(name: str = "flight-delay-events") -> dict[str, Any]:
    request = desired_request(name)
    return {
        "Table": {
            **request,
            "TableStatus": "ACTIVE",
            "BillingModeSummary": {"BillingMode": "PAY_PER_REQUEST"},
            "GlobalSecondaryIndexes": request["GlobalSecondaryIndexes"],
        }
    }


class FakeClient:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.created: dict[str, Any] | None = None

    def describe_table(self, **kwargs: Any) -> dict[str, Any]:
        if not self.exists:
            self.exists = True
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "DescribeTable",
            )
        return active_description(kwargs["TableName"])

    def create_table(self, **kwargs: Any) -> None:
        self.created = kwargs

    def get_waiter(self, name: str) -> Any:
        assert name == "table_exists"
        return type("Waiter", (), {"wait": lambda self, **kwargs: None})()


def test_dry_run_is_safe_and_exact() -> None:
    result = provision(client=None, table_name="flight-delay-events", dry_run=True)
    assert result["action"] == "dry-run"
    assert result["request"]["BillingMode"] == "PAY_PER_REQUEST"
    assert result["request"]["GlobalSecondaryIndexes"][0]["IndexName"] == INDEX_NAME


def test_provision_creates_missing_table_idempotently() -> None:
    client = FakeClient(exists=False)
    result = provision(client=client, table_name="flight-delay-events")
    assert result["action"] == "created"
    assert client.created == desired_request("flight-delay-events")
    assert provision(client=client, table_name="flight-delay-events")["action"] == "validated"


def test_validation_rejects_incompatible_table() -> None:
    description = active_description()
    description["Table"]["KeySchema"] = [{"AttributeName": "wrong", "KeyType": "HASH"}]
    with pytest.raises(TableContractError, match="partition key"):
        validate_description(description, "flight-delay-events")
