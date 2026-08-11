"""DynamoDB event repository with strict serialization and conditional writes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel


class PersistenceError(RuntimeError):
    """Sanitized persistence failure."""


class PersistenceConflict(PersistenceError):
    """A conditional write rejected a collision or stale revision."""


def to_dynamodb(value: Any) -> Any:
    """Recursively convert Python values to DynamoDB-safe values."""

    if isinstance(value, BaseModel):
        return to_dynamodb(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return to_dynamodb(value.value)
    if isinstance(value, bool) or value is None or isinstance(value, str | Decimal):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers cannot be persisted")
        return Decimal(str(value))
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("persisted datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_dynamodb(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_dynamodb(item) for item in value]
    raise TypeError(f"unsupported DynamoDB value type: {type(value).__name__}")


def from_dynamodb(value: Any) -> Any:
    """Recursively convert Decimal values to ordinary API JSON numbers."""

    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Mapping):
        return {str(key): from_dynamodb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_dynamodb(item) for item in value]
    return value


def _conditional(error: ClientError) -> bool:
    return error.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


class DynamoDBRepository:
    """Production prediction, feedback, and active-model persistence adapter."""

    def __init__(
        self,
        *,
        table_name: str = "flight-delay-events",
        region_name: str = "us-west-2",
        endpoint_url: str | None = None,
        resource: Any | None = None,
    ) -> None:
        resource = resource or boto3.resource(
            "dynamodb", region_name=region_name, endpoint_url=endpoint_url
        )
        self.table = resource.Table(table_name)

    def connect(self) -> None:
        """Verify table access during application startup."""

        try:
            self.table.load()
        except Exception as error:
            raise PersistenceError("DynamoDB table is unavailable") from error

    def close(self) -> None:
        """No-op hook matching the injected repository lifecycle contract."""

    def put_prediction(self, item: Mapping[str, Any]) -> None:
        """Create a unique prediction event without allowing overwrite."""

        try:
            self.table.put_item(
                Item=to_dynamodb(dict(item)),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except ClientError as error:
            if _conditional(error):
                raise PersistenceConflict("prediction identifier already exists") from error
            raise PersistenceError("prediction persistence failed") from error

    def put_error(self, item: Mapping[str, Any]) -> None:
        """Best-effort write of a sanitized inference failure event."""

        try:
            self.table.put_item(
                Item=to_dynamodb(dict(item)),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception:
            return

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        """Strongly consistently read one prediction event."""

        try:
            response = self.table.get_item(
                Key={"pk": f"PREDICTION#{prediction_id}"}, ConsistentRead=True
            )
        except ClientError as error:
            raise PersistenceError("prediction retrieval failed") from error
        item = response.get("Item")
        return from_dynamodb(item) if item else None

    def put_model_metadata(self, item: Mapping[str, Any]) -> None:
        """Upsert load timestamps only while immutable model identity matches."""

        model = dict(item)
        now = model["last_loaded_at"]
        values = to_dynamodb(
            {
                ":version": model["registry_version"],
                ":digest": model["registry_digest"],
                ":bundle": model["bundle_digest"],
                ":now": now,
                ":payload": model,
            }
        )
        try:
            self.table.update_item(
                Key={"pk": model["pk"]},
                UpdateExpression=(
                    "SET first_loaded_at = if_not_exists(first_loaded_at, :now), "
                    "last_loaded_at = :now, model_metadata = :payload, "
                    "registry_version = :version, registry_digest = :digest, "
                    "bundle_digest = :bundle"
                ),
                ConditionExpression=(
                    "attribute_not_exists(pk) OR "
                    "(registry_version = :version AND registry_digest = :digest "
                    "AND bundle_digest = :bundle)"
                ),
                ExpressionAttributeValues=values,
            )
        except ClientError as error:
            if _conditional(error):
                raise PersistenceConflict("active model metadata identity conflict") from error
            raise PersistenceError("model metadata persistence failed") from error

    def update_feedback(
        self, prediction_id: str, feedback: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Revision feedback only on an existing prediction record."""

        current = self.get_prediction(prediction_id)
        if current is None:
            return None
        expected = int(current.get("feedback_revision", 0))
        revision = expected + 1
        feedback_record = {**feedback, "feedback_revision": revision}
        values = to_dynamodb(
            {":expected": expected, ":revision": revision, ":feedback": feedback_record}
        )
        try:
            response = self.table.update_item(
                Key={"pk": f"PREDICTION#{prediction_id}"},
                UpdateExpression="SET feedback = :feedback, feedback_revision = :revision",
                ConditionExpression=(
                    "attribute_exists(pk) AND "
                    "(attribute_not_exists(feedback_revision) OR feedback_revision = :expected)"
                ),
                ExpressionAttributeValues=values,
                ReturnValues="ALL_NEW",
            )
        except ClientError as error:
            if _conditional(error):
                raise PersistenceConflict("feedback revision conflict") from error
            raise PersistenceError("feedback persistence failed") from error
        return from_dynamodb(response["Attributes"])
