from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import pytest
from botocore.exceptions import ClientError

from flight_delay.contracts import RiskBand, RouteReliability
from flight_delay.persistence.dynamodb import (
    DynamoDBRepository,
    PersistenceConflict,
    PersistenceError,
    from_dynamodb,
    to_dynamodb,
)


def test_to_dynamodb_serializes_contract_models_and_enums() -> None:
    payload = {
        "risk_band": RiskBand.HIGH,
        "route": RouteReliability(
            scope="all_carriers",
            origin="DEN",
            destination="LAX",
            eligible_flights=100,
            on_time_count=70,
            on_time_rate=0.7,
            delayed_count=30,
            delayed_rate=0.3,
            meets_minimum_support=True,
        ),
    }

    serialized = to_dynamodb(payload)

    assert serialized["risk_band"] == "high"
    assert serialized["route"]["origin"] == "DEN"
    assert serialized["route"]["on_time_rate"] == Decimal("0.7")


def conditional_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "conflict"}},
        "UpdateItem",
    )


def service_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "InternalServerError", "Message": "private service detail"}},
        "DynamoDBOperation",
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


def test_serialization_rejects_naive_datetime_and_unknown_types() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        to_dynamodb(datetime(2026, 8, 10))
    with pytest.raises(TypeError, match="unsupported DynamoDB value type"):
        to_dynamodb({1, 2})
    assert to_dynamodb(date(2026, 8, 10)) == "2026-08-10"
    assert to_dynamodb(time(8, 30)) == "08:30:00"
    assert to_dynamodb((1, 2)) == [1, 2]
    assert from_dynamodb(Decimal("2")) == 2


def test_connect_and_prediction_failures_are_sanitized() -> None:
    table = FakeTable()

    def fail_load() -> None:
        raise RuntimeError("private endpoint detail")

    table.load = fail_load
    repository = DynamoDBRepository(resource=FakeResource(table))
    with pytest.raises(PersistenceError, match="table is unavailable"):
        repository.connect()
    repository.close()

    def fail_put(**kwargs: Any) -> None:
        raise service_error()

    table.put_item = fail_put
    with pytest.raises(PersistenceError, match="prediction persistence failed"):
        repository.put_prediction({"pk": "PREDICTION#one"})


def test_error_write_is_best_effort_for_success_and_failure() -> None:
    table = FakeTable()
    repository = DynamoDBRepository(resource=FakeResource(table))
    repository.put_error({"pk": "ERROR#one", "detail": "sanitized"})
    assert table.items["ERROR#one"]["detail"] == "sanitized"

    def fail_put(**kwargs: Any) -> None:
        raise service_error()

    table.put_item = fail_put
    assert repository.put_error({"pk": "ERROR#two"}) is None


def test_retrieval_and_model_metadata_service_failures_are_sanitized() -> None:
    table = FakeTable()
    repository = DynamoDBRepository(resource=FakeResource(table))

    def fail_get(**kwargs: Any) -> dict[str, Any]:
        raise service_error()

    table.get_item = fail_get
    with pytest.raises(PersistenceError, match="prediction retrieval failed"):
        repository.get_prediction("one")

    model = {
        "pk": "MODEL#v0",
        "registry_version": "v0",
        "registry_digest": "digest",
        "bundle_digest": "bundle",
        "last_loaded_at": datetime(2026, 8, 10, tzinfo=UTC),
    }

    def fail_update(**kwargs: Any) -> dict[str, Any]:
        raise service_error()

    table.update_item = fail_update
    with pytest.raises(PersistenceError, match="model metadata persistence failed"):
        repository.put_model_metadata(model)

    table.fail_condition = True
    table.update_item = FakeTable.update_item.__get__(table)
    with pytest.raises(PersistenceConflict, match="metadata identity conflict"):
        repository.put_model_metadata(model)


def test_feedback_service_failure_is_sanitized() -> None:
    table = FakeTable()
    table.items["PREDICTION#one"] = {"pk": "PREDICTION#one"}
    repository = DynamoDBRepository(resource=FakeResource(table))

    def fail_update(**kwargs: Any) -> dict[str, Any]:
        raise service_error()

    table.update_item = fail_update
    with pytest.raises(PersistenceError, match="feedback persistence failed"):
        repository.update_feedback("one", {"actual_delayed": False})
