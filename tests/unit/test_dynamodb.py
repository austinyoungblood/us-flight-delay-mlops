from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from botocore.exceptions import ClientError

from flight_delay.persistence.dynamodb import (
    DynamoDBRepository,
    PersistenceConflict,
    from_dynamodb,
    to_dynamodb,
)


def conditional_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "conflict"}},
        "UpdateItem",
    )


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.puts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.loaded = False
        self.fail_condition = False

    def load(self) -> None:
        self.loaded = True

    def put_item(self, **kwargs: Any) -> None:
        if self.fail_condition:
            raise conditional_error()
        self.puts.append(kwargs)
        self.items[kwargs["Item"]["pk"]] = kwargs["Item"]

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        item = self.items.get(kwargs["Key"]["pk"])
        return {"Item": item} if item else {}

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_condition:
            raise conditional_error()
        self.updates.append(kwargs)
        key = kwargs["Key"]["pk"]
        if key.startswith("PREDICTION#"):
            item = self.items[key]
            values = kwargs["ExpressionAttributeValues"]
            item["feedback"] = values[":feedback"]
            item["feedback_revision"] = values[":revision"]
            return {"Attributes": item}
        return {"Attributes": {}}


class FakeResource:
    def __init__(self, table: FakeTable) -> None:
        self.table = table

    def Table(self, name: str) -> FakeTable:
        assert name == "flight-delay-events"
        return self.table


def test_decimal_serialization_preserves_bool_and_round_trips() -> None:
    source = {
        "probability": 0.2,
        "count": 2,
        "flag": True,
        "nested": [1.5, False],
        "at": datetime(2026, 8, 10, tzinfo=UTC),
    }
    encoded = to_dynamodb(source)
    assert encoded["probability"] == Decimal("0.2")
    assert encoded["flag"] is True
    assert encoded["nested"] == [Decimal("1.5"), False]
    assert from_dynamodb(encoded)["probability"] == 0.2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_decimal_serialization_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        to_dynamodb({"value": value})


def test_prediction_item_uses_conditional_put_and_consistent_read() -> None:
    table = FakeTable()
    repository = DynamoDBRepository(resource=FakeResource(table))
    repository.connect()
    repository.put_prediction({"pk": "PREDICTION#one", "probability": 0.2})
    assert table.loaded
    assert table.puts[0]["ConditionExpression"] == "attribute_not_exists(pk)"
    assert repository.get_prediction("one") == {
        "pk": "PREDICTION#one",
        "probability": 0.2,
    }


def test_prediction_collision_maps_to_conflict() -> None:
    table = FakeTable()
    table.fail_condition = True
    repository = DynamoDBRepository(resource=FakeResource(table))
    with pytest.raises(PersistenceConflict, match="already exists"):
        repository.put_prediction({"pk": "PREDICTION#one"})


def test_feedback_requires_existing_item_and_increments_revision() -> None:
    table = FakeTable()
    table.items["PREDICTION#one"] = {
        "pk": "PREDICTION#one",
        "predicted_delayed": True,
    }
    repository = DynamoDBRepository(resource=FakeResource(table))
    updated = repository.update_feedback(
        "one",
        {
            "actual_delayed": True,
            "feedback_correct": True,
            "feedback_at": datetime(2026, 8, 10, tzinfo=UTC),
        },
    )
    assert updated is not None
    assert updated["feedback_revision"] == 1
    assert updated["feedback"]["feedback_revision"] == 1
    assert repository.update_feedback("missing", {}) is None


def test_feedback_stale_revision_maps_to_conflict() -> None:
    table = FakeTable()
    table.items["PREDICTION#one"] = {"pk": "PREDICTION#one"}
    table.fail_condition = True
    repository = DynamoDBRepository(resource=FakeResource(table))
    with pytest.raises(PersistenceConflict, match="revision conflict"):
        repository.update_feedback("one", {"actual_delayed": False})


def test_model_metadata_uses_immutable_identity_condition() -> None:
    table = FakeTable()
    repository = DynamoDBRepository(resource=FakeResource(table))
    repository.put_model_metadata(
        {
            "pk": "MODEL#v0",
            "registry_version": "v0",
            "registry_digest": "digest",
            "bundle_digest": "bundle",
            "last_loaded_at": datetime(2026, 8, 10, tzinfo=UTC),
        }
    )
    call = table.updates[0]
    assert "attribute_not_exists(pk) OR" in call["ConditionExpression"]
    assert "if_not_exists(first_loaded_at" in call["UpdateExpression"]
