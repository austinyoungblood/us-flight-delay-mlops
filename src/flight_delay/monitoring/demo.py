"""Deterministic, explicitly labeled demo prediction events."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from botocore.exceptions import ClientError

from flight_delay.persistence import PersistenceConflict, to_dynamodb

_NAMESPACE = uuid.UUID("9bd5938d-145e-4c0e-84c9-2f54210c9b31")


def demo_events(
    *,
    batch_id: str,
    count: int = 30,
    start_date: date | None = None,
    model_version: str = "v0",
    model_alias: str = "production",
) -> list[dict[str, Any]]:
    """Generate schema-compatible demo events without invoking a model."""

    if not batch_id.strip():
        raise ValueError("demo batch ID is required")
    if count <= 0 or count > 10_000:
        raise ValueError("demo event count must be in 1..10000")
    start_date = start_date or date(2026, 8, 1)
    routes = [("UA", "DEN", "LAX"), ("WN", "DEN", "PHX"), ("AA", "DFW", "ORD")]
    events: list[dict[str, Any]] = []
    for index in range(count):
        carrier, origin, destination = routes[index % len(routes)]
        created_at = datetime.combine(
            start_date + timedelta(days=index % 7), time(8 + index % 12, index % 60), tzinfo=UTC
        )
        prediction_id = str(uuid.uuid5(_NAMESPACE, f"{batch_id}:{index}"))
        probability = round(0.05 + 0.9 * ((index * 37) % 100) / 100, 6)
        threshold = 0.1840285229739868
        predicted = probability >= threshold
        event: dict[str, Any] = {
            "pk": f"PREDICTION#{prediction_id}",
            "prediction_id": prediction_id,
            "event_date": created_at.date().isoformat(),
            "created_at": created_at,
            "request": {
                "carrier": carrier,
                "origin": origin,
                "destination": destination,
                "flight_date": (created_at.date() + timedelta(days=14)).isoformat(),
                "scheduled_departure": f"{8 + index % 12:02d}:00:00",
                "scheduled_arrival": f"{10 + index % 12:02d}:00:00",
                "scheduled_elapsed_minutes": 90 + index % 180,
                "distance_miles": 300.0 + (index * 73) % 1800,
            },
            "delay_probability": probability,
            "predicted_delayed": predicted,
            "risk_band": _risk_band(probability, threshold),
            "classification_threshold": threshold,
            "route_reliability": [],
            "support_warning": None,
            "model_alias": model_alias,
            "model_version": model_version,
            "model_digest": "demo-provenance-not-a-real-inference",
            "bundle_digest": "demo-provenance-not-a-real-inference",
            "cache_hit": bool(index % 4 == 0),
            "latency_ms": float(15 + index % 40),
            "inference_latency_ms": float(3 + index % 10),
            "persistence_latency_ms": float(4 + index % 12),
            "total_latency_ms": float(15 + index % 40),
            "request_status": "success",
            "feedback": None,
            "demo_data": True,
            "demo_batch_id": batch_id,
        }
        if index % 3 == 0:
            actual = bool((index // 3) % 2)
            event["feedback_revision"] = 1
            event["feedback"] = {
                "actual_delayed": actual,
                "arrival_delay_minutes": 25.0 if actual else -2.0,
                "notes": "Deterministic demo outcome",
                "source": "demo-seeder",
                "feedback_correct": actual == predicted,
                "feedback_at": created_at + timedelta(hours=4),
                "feedback_revision": 1,
            }
        events.append(event)
    return events


def _risk_band(probability: float, threshold: float) -> str:
    if probability < 0.75 * threshold:
        return "low"
    if probability < 1.25 * threshold:
        return "medium"
    return "high"


def seed_events(table: Any, events: list[dict[str, Any]]) -> int:
    """Conditionally write demo events without overwriting any prediction."""

    written = 0
    for event in events:
        try:
            table.put_item(Item=to_dynamodb(event), ConditionExpression="attribute_not_exists(pk)")
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise PersistenceConflict("demo prediction identifier already exists") from error
            raise
        written += 1
    return written


def cleanup_demo_batch(table: Any, batch_id: str) -> int:
    """Delete only items matching one explicitly named demo batch."""

    if not batch_id.strip():
        raise ValueError("demo batch ID is required")
    deleted = 0
    kwargs: dict[str, Any] = {
        "FilterExpression": "demo_batch_id = :batch",
        "ExpressionAttributeValues": {":batch": batch_id},
        "ProjectionExpression": "pk, demo_batch_id",
    }
    while True:
        response = table.scan(**kwargs)
        for item in response.get("Items", []):
            if item.get("demo_batch_id") != batch_id:
                continue
            table.delete_item(
                Key={"pk": item["pk"]},
                ConditionExpression="demo_batch_id = :batch",
                ExpressionAttributeValues={":batch": batch_id},
            )
            deleted += 1
        key = response.get("LastEvaluatedKey")
        if not key:
            break
        kwargs["ExclusiveStartKey"] = key
    return deleted
