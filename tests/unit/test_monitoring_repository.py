from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from flight_delay.monitoring.repository import MonitoringRepository, date_partitions


class FakeTable:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def load(self) -> None:
        return None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        if "ExclusiveStartKey" not in kwargs:
            return {
                "Items": [
                    {
                        "pk": "PREDICTION#one",
                        "created_at": "2026-08-01T08:00:00Z",
                        "request_status": "success",
                        "model_version": "v0",
                        "request": {
                            "carrier": "UA",
                            "origin": "DEN",
                            "destination": "LAX",
                        },
                    }
                ],
                "LastEvaluatedKey": {"pk": "PREDICTION#one"},
            }
        return {
            "Items": [
                {
                    "pk": "PREDICTION#two",
                    "created_at": "2026-08-01T09:00:00Z",
                    "request_status": "error",
                    "model_version": "v0",
                    "request": {
                        "carrier": "WN",
                        "origin": "DEN",
                        "destination": "PHX",
                    },
                }
            ]
        }

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.gets.append(kwargs)
        return {"Item": {"pk": kwargs["Key"]["pk"], "registry_version": "v0"}}

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        return {"Items": [{"pk": "MODEL#v0", "last_loaded_at": "2026-08-10"}]}


class FakeResource:
    def __init__(self, table: FakeTable) -> None:
        self.value = table

    def Table(self, name: str) -> FakeTable:
        assert name == "flight-delay-events"
        return self.value


def test_date_partitions_are_inclusive_and_bounded() -> None:
    assert date_partitions(date(2026, 8, 1), date(2026, 8, 3)) == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
    ]
    with pytest.raises(ValueError, match="31"):
        date_partitions(date(2026, 1, 1), date(2026, 2, 1))


def test_query_consumes_each_page_and_applies_filters() -> None:
    table = FakeTable()
    repository = MonitoringRepository(resource=FakeResource(table))
    items = repository.query_predictions(
        date(2026, 8, 1), date(2026, 8, 1), carrier="UA", route="DEN-LAX"
    )
    assert [item["pk"] for item in items] == ["PREDICTION#one"]
    assert len(table.queries) == 2
    assert table.queries[0]["IndexName"] == "event-date-created-at-index"
    assert table.queries[1]["ExclusiveStartKey"] == {"pk": "PREDICTION#one"}
    assert "ConsistentRead" not in table.queries[0]


def test_model_metadata_exact_and_bounded_fallback() -> None:
    table = FakeTable()
    repository = MonitoringRepository(resource=FakeResource(table))
    assert repository.get_model_metadata("v0")["pk"] == "MODEL#v0"
    assert table.gets[0]["ConsistentRead"] is True
    assert repository.get_model_metadata()["pk"] == "MODEL#v0"
