from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from botocore.exceptions import ClientError

from flight_delay.monitoring.repository import MonitoringRepository, date_partitions
from flight_delay.persistence import PersistenceError


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
    with pytest.raises(ValueError, match="must not precede"):
        date_partitions(date(2026, 8, 2), date(2026, 8, 1))


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


def client_error(operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "InternalServerError", "Message": "private detail"}}, operation
    )


def test_repository_lifecycle_and_delegation() -> None:
    table = FakeTable()
    repository = MonitoringRepository(resource=FakeResource(table))
    repository.connect()
    repository.close()
    table.items = {"PREDICTION#one": {"pk": "PREDICTION#one"}}
    assert repository.get_prediction("one")["pk"] == "PREDICTION#one"


def test_query_filters_model_status_and_sanitizes_service_errors() -> None:
    table = FakeTable()
    repository = MonitoringRepository(resource=FakeResource(table))
    assert (
        repository.query_predictions(
            date(2026, 8, 1),
            date(2026, 8, 1),
            model_version="v0",
            request_status="error",
        )[0]["pk"]
        == "PREDICTION#two"
    )
    assert (
        repository.query_predictions(date(2026, 8, 1), date(2026, 8, 1), model_version="v9") == []
    )

    def fail_query(**kwargs: Any) -> dict[str, Any]:
        raise client_error("Query")

    table.query = fail_query
    with pytest.raises(PersistenceError, match="monitoring query failed"):
        repository.query_predictions(date(2026, 8, 1), date(2026, 8, 1))


def test_model_metadata_handles_missing_latest_and_service_failure() -> None:
    table = FakeTable()
    repository = MonitoringRepository(resource=FakeResource(table))
    table.get_item = lambda **kwargs: {}
    assert repository.get_model_metadata("missing") is None
    table.scan = lambda **kwargs: {"Items": []}
    assert repository.get_model_metadata() is None
    table.scan = lambda **kwargs: {
        "Items": [
            {"pk": "MODEL#v0", "last_loaded_at": "2026-08-09"},
            {"pk": "MODEL#v1", "last_loaded_at": "2026-08-10"},
        ]
    }
    assert repository.get_model_metadata()["pk"] == "MODEL#v1"

    def fail_get(**kwargs: Any) -> dict[str, Any]:
        raise client_error("GetItem")

    table.get_item = fail_get
    with pytest.raises(PersistenceError, match="metadata retrieval failed"):
        repository.get_model_metadata("v0")
