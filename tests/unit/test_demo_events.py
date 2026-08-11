from __future__ import annotations

from datetime import date
from typing import Any

from flight_delay.monitoring.demo import cleanup_demo_batch, demo_events, seed_events


class FakeTable:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []

    def put_item(self, **kwargs: Any) -> None:
        self.puts.append(kwargs)

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "Items": [
                {"pk": "PREDICTION#one", "demo_batch_id": "batch-a"},
                {"pk": "PREDICTION#two", "demo_batch_id": "batch-b"},
            ]
        }

    def delete_item(self, **kwargs: Any) -> None:
        self.deletes.append(kwargs)


def test_demo_generation_is_deterministic_labeled_and_schema_shaped() -> None:
    first = demo_events(batch_id="batch-a", count=5, start_date=date(2026, 8, 1))
    second = demo_events(batch_id="batch-a", count=5, start_date=date(2026, 8, 1))
    assert first == second
    assert all(item["demo_data"] is True for item in first)
    assert all(item["demo_batch_id"] == "batch-a" for item in first)
    assert all(item["pk"].startswith("PREDICTION#") for item in first)
    assert all(item["model_digest"] == "demo-provenance-not-a-real-inference" for item in first)


def test_seed_is_conditional_and_cleanup_is_batch_scoped() -> None:
    table = FakeTable()
    events = demo_events(batch_id="batch-a", count=2)
    assert seed_events(table, events) == 2
    assert all(call["ConditionExpression"] == "attribute_not_exists(pk)" for call in table.puts)
    assert cleanup_demo_batch(table, "batch-a") == 1
    assert table.deletes[0]["Key"] == {"pk": "PREDICTION#one"}
    assert table.deletes[0]["ExpressionAttributeValues"] == {":batch": "batch-a"}
