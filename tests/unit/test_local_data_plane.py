from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from flight_delay.persistence.dynamodb import DynamoDBRepository
from services.api.app.main import Settings

ROOT = Path(__file__).resolve().parents[2]


class Resource:
    def Table(self, name: str) -> object:
        return object()


def test_repository_forwards_development_endpoint_without_changing_default(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def resource(service: str, **kwargs: Any) -> Resource:
        calls.append({"service": service, **kwargs})
        return Resource()

    monkeypatch.setattr("flight_delay.persistence.dynamodb.boto3.resource", resource)
    DynamoDBRepository(endpoint_url="http://dynamodb-local:8000")
    DynamoDBRepository()
    assert calls[0]["endpoint_url"] == "http://dynamodb-local:8000"
    assert calls[1]["endpoint_url"] is None


def test_api_settings_use_only_explicit_local_endpoint(monkeypatch: Any) -> None:
    monkeypatch.setenv("DYNAMODB_ENDPOINT_URL", "http://dynamodb-local:8000")
    assert Settings.from_environment().dynamodb_endpoint_url == "http://dynamodb-local:8000"
    monkeypatch.delenv("DYNAMODB_ENDPOINT_URL")
    assert Settings.from_environment().dynamodb_endpoint_url is None


def test_compose_local_plane_uses_dummy_local_credentials_and_exact_endpoint() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert services["dynamodb-local"]["image"].startswith("amazon/dynamodb-local:")
    environment = services["table-init"]["environment"]
    assert environment["AWS_ACCESS_KEY_ID"] == "local"
    assert environment["AWS_SECRET_ACCESS_KEY"] == "local"
    assert environment["DYNAMODB_ENDPOINT_URL"] == "http://dynamodb-local:8000"
    assert (
        services["api"]["environment"]["DYNAMODB_ENDPOINT_URL"]
        == environment["DYNAMODB_ENDPOINT_URL"]
    )
    assert (
        services["monitor-ui"]["environment"]["DYNAMODB_ENDPOINT_URL"]
        == environment["DYNAMODB_ENDPOINT_URL"]
    )
