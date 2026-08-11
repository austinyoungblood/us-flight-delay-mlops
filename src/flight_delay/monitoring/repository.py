"""Read-optimized DynamoDB monitoring repository."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from flight_delay.persistence import DynamoDBRepository, PersistenceError, from_dynamodb

EVENT_INDEX = "event-date-created-at-index"


def date_partitions(start_date: date, end_date: date, *, max_days: int = 31) -> list[str]:
    """Return inclusive UTC date partitions with an enforced interactive bound."""

    if end_date < start_date:
        raise ValueError("end date must not precede start date")
    count = (end_date - start_date).days + 1
    if count > max_days:
        raise ValueError(f"date range cannot exceed {max_days} days")
    return [(start_date + timedelta(days=offset)).isoformat() for offset in range(count)]


class MonitoringRepository:
    """DynamoDB-only data plane for monitoring and feedback adjudication."""

    def __init__(
        self,
        *,
        table_name: str = "flight-delay-events",
        region_name: str = "us-west-2",
        endpoint_url: str | None = None,
        resource: Any | None = None,
        max_days: int = 31,
    ) -> None:
        resource = resource or boto3.resource(
            "dynamodb", region_name=region_name, endpoint_url=endpoint_url
        )
        self._events = DynamoDBRepository(
            table_name=table_name, region_name=region_name, resource=resource
        )
        self.table = self._events.table
        self.max_days = max_days

    def connect(self) -> None:
        self._events.connect()

    def close(self) -> None:
        self._events.close()

    def query_predictions(
        self,
        start_date: date,
        end_date: date,
        *,
        carrier: str | None = None,
        route: str | None = None,
        model_version: str | None = None,
        request_status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query every bounded UTC GSI partition and consume every page."""

        items: list[dict[str, Any]] = []
        try:
            for partition in date_partitions(start_date, end_date, max_days=self.max_days):
                kwargs: dict[str, Any] = {
                    "IndexName": EVENT_INDEX,
                    "KeyConditionExpression": Key("event_date").eq(partition),
                }
                while True:
                    response = self.table.query(**kwargs)
                    items.extend(from_dynamodb(item) for item in response.get("Items", []))
                    key = response.get("LastEvaluatedKey")
                    if not key:
                        break
                    kwargs["ExclusiveStartKey"] = key
        except ClientError as error:
            raise PersistenceError("monitoring query failed") from error

        def selected(item: dict[str, Any]) -> bool:
            request = item.get("request") or {}
            item_route = f"{request.get('origin', '')}-{request.get('destination', '')}"
            return all(
                (
                    carrier is None or request.get("carrier") == carrier,
                    route is None or item_route == route,
                    model_version is None or item.get("model_version") == model_version,
                    request_status is None or item.get("request_status") == request_status,
                )
            )

        return sorted(
            (item for item in items if selected(item)), key=lambda item: item.get("created_at", "")
        )

    def get_model_metadata(self, model_version: str | None = None) -> dict[str, Any] | None:
        """Read exact model metadata, or perform one bounded metadata-only lookup."""

        try:
            if model_version:
                response = self.table.get_item(
                    Key={"pk": f"MODEL#{model_version}"}, ConsistentRead=True
                )
                item = response.get("Item")
                return from_dynamodb(item) if item else None
            response = self.table.scan(FilterExpression=Attr("pk").begins_with("MODEL#"), Limit=100)
        except ClientError as error:
            raise PersistenceError("model metadata retrieval failed") from error
        models = [from_dynamodb(item) for item in response.get("Items", [])]
        if not models:
            return None
        return max(models, key=lambda item: item.get("last_loaded_at", ""))

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return self._events.get_prediction(prediction_id)

    def update_feedback(
        self, prediction_id: str, feedback: dict[str, Any]
    ) -> dict[str, Any] | None:
        return self._events.update_feedback(prediction_id, feedback)
