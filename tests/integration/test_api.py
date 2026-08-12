from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime
from typing import Any

import httpx

from flight_delay.contracts import RouteReliability
from flight_delay.persistence import PersistenceError
from services.api.app.main import Settings, create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.threshold = 0.2
        self.calls = 0
        self.training_baseline = {"row_count": 100}
        self.identity = {
            "registry_path": "wandb-registry-Model/us-flight-arrival-delay-15m",
            "serving_alias": "production",
            "registry_version": "v0",
            "registry_digest": "digest",
            "source_artifact_digest": "source",
            "bundle_digest": "bundle",
            "selection_lock_sha256": "lock",
            "route_asset_sha256": "route",
            "classification_threshold": 0.2,
            "feature_schema": ["Origin"],
            "training_partitions": {"base_fit": "2025-01-01/2025-10-31"},
            "release_decision": {
                "serving_alias": "production",
                "final_test_passed": False,
                "internal_production_gate_passed": False,
                "deployment_purpose": "academic_demo",
            },
            "release_git_sha": "abc123",
            "loaded_at": datetime(2026, 8, 10, tzinfo=UTC),
            "internal_production_gate_passed": False,
            "deployment_purpose": "academic_demo",
            "governance_notice": "Academic demonstration — internal production gate failed.",
            "serving_stage_notice": "Academic demonstration — internal production gate failed.",
        }

    def route_reliability(
        self, origin: str, destination: str, carrier: str | None = None
    ) -> list[RouteReliability]:
        if (origin, destination) != ("DEN", "LAX"):
            return []
        return [
            RouteReliability(
                scope="all_carriers",
                origin="DEN",
                destination="LAX",
                eligible_flights=100,
                on_time_count=70,
                on_time_rate=0.7,
                delayed_count=30,
                delayed_rate=0.3,
                meets_minimum_support=True,
            )
        ]

    def predict(self, request: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "delay_probability": 0.3,
            "predicted_delayed": True,
            "risk_band": "high",
            "route_reliability": self.route_reliability(
                request.origin, request.destination, request.carrier
            ),
            "support_warning": None,
        }


class Loader:
    def __init__(self, runtime: FakeRuntime | None = None, error: Exception | None = None) -> None:
        self.runtime = runtime
        self.error = error

    def load(self) -> FakeRuntime:
        if self.error:
            raise self.error
        assert self.runtime is not None
        return self.runtime


class MemoryRepository:
    def __init__(self, *, fail_put: bool = False) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.models: list[dict[str, Any]] = []
        self.fail_put = fail_put
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def put_model_metadata(self, item: dict[str, Any]) -> None:
        self.models.append(copy.deepcopy(item))

    def put_prediction(self, item: dict[str, Any]) -> None:
        if self.fail_put:
            raise PersistenceError("DynamoDB unavailable")
        self.items[item["pk"]] = copy.deepcopy(item)

    def put_error(self, item: dict[str, Any]) -> None:
        self.items[item["pk"]] = copy.deepcopy(item)

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        item = self.items.get(f"PREDICTION#{prediction_id}")
        return copy.deepcopy(item) if item else None

    def update_feedback(
        self, prediction_id: str, feedback: dict[str, Any]
    ) -> dict[str, Any] | None:
        item = self.items.get(f"PREDICTION#{prediction_id}")
        if item is None:
            return None
        revision = int(item.get("feedback_revision", 0)) + 1
        item["feedback_revision"] = revision
        item["feedback"] = {**feedback, "feedback_revision": revision}
        return copy.deepcopy(item)


def prediction_payload() -> dict[str, Any]:
    return {
        "carrier": "UA",
        "origin": "DEN",
        "destination": "LAX",
        "flight_date": "2026-08-18",
        "scheduled_departure": "07:30:00",
        "scheduled_arrival": "09:00:00",
        "scheduled_elapsed_minutes": 150,
        "distance_miles": 862,
    }


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)


def configured_app(
    runtime: FakeRuntime | None = None, repository: MemoryRepository | None = None
) -> Any:
    runtime = runtime or FakeRuntime()
    repository = repository or MemoryRepository()
    return create_app(
        runtime_loader=Loader(runtime),
        repository_factory=lambda: repository,
        settings=Settings(),
    )


def test_ready_health_and_model_info_are_exact() -> None:
    app = configured_app()

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/health"), await client.get("/model-info")

    health, info = run(scenario())
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert info.json()["serving_alias"] == "production"
    assert info.json()["internal_production_gate_passed"] is False
    assert info.json()["deployment_purpose"] == "academic_demo"
    assert "Academic demonstration" in info.json()["governance_notice"]


def test_degraded_health_and_predict_503() -> None:
    app = create_app(
        runtime_loader=Loader(error=RuntimeError("Registry denied")),
        repository_factory=MemoryRepository,
        settings=Settings(),
    )

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/health"), await client.post(
                    "/predict", json=prediction_payload()
                )

    health, prediction = run(scenario())
    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert health.json()["dependencies"]["registry"]["detail"] == "Registry denied"
    assert prediction.status_code == 503


def test_predict_cache_keeps_unique_persisted_events_and_feedback_round_trip() -> None:
    runtime = FakeRuntime()
    repository = MemoryRepository()
    app = configured_app(runtime, repository)

    async def scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = await client.post("/predict", json=prediction_payload())
                second = await client.post("/predict", json=prediction_payload())
                prediction_id = first.json()["prediction_id"]
                feedback = await client.post(
                    f"/feedback/{prediction_id}",
                    json={
                        "actual_delayed": True,
                        "arrival_delay_minutes": 22,
                        "notes": "landed late",
                        "source": "traveler",
                    },
                )
                retrieved = await client.get(f"/predictions/{prediction_id}")
                return first, second, feedback, retrieved

    first, second, feedback, retrieved = run(scenario())
    assert first.status_code == second.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    assert runtime.calls == 1
    assert len([key for key in repository.items if key.startswith("PREDICTION#")]) == 2
    assert feedback.status_code == 200
    assert feedback.json()["feedback_correct"] is True
    assert feedback.json()["feedback_revision"] == 1
    assert retrieved.json()["feedback"]["source"] == "traveler"


def test_endpoint_errors_and_route_fallback() -> None:
    app = configured_app()

    async def scenario() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                invalid = await client.post("/predict", json={"carrier": "bad"})
                missing_route = await client.get("/route-reliability?origin=DEN&destination=JFK")
                route = await client.get("/route-reliability?origin=den&destination=lax&carrier=ua")
                missing_prediction = await client.get("/predictions/missing")
                missing_feedback = await client.post(
                    "/feedback/missing", json={"actual_delayed": False}
                )
                return [invalid, missing_route, route, missing_prediction, missing_feedback]

    invalid, missing_route, route, missing_prediction, missing_feedback = run(scenario())
    assert invalid.status_code == 422
    assert missing_route.status_code == 404
    assert route.status_code == 200
    assert route.json()[0]["scope"] == "all_carriers"
    assert missing_prediction.status_code == 404
    assert missing_feedback.status_code == 404


def test_persistence_failure_never_returns_prediction_success() -> None:
    app = configured_app(repository=MemoryRepository(fail_put=True))
    response = run(request(app, "POST", "/predict", json=prediction_payload()))
    assert response.status_code == 503
    assert response.json()["detail"] == "prediction persistence is unavailable"
