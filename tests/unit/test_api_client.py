from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from flight_delay.contracts import FeedbackRequest, FlightPredictionRequest
from flight_delay.ui import ApiClientError, FlightDelayApiClient


def client(handler: object) -> FlightDelayApiClient:
    return FlightDelayApiClient(
        "http://api",
        transport=httpx.MockTransport(handler),
    )


def test_health_preserves_structured_degraded_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            503,
            json={
                "service": "flight-delay-api",
                "status": "degraded",
                "model_loaded": False,
                "database_connected": False,
                "dependencies": {
                    "registry": {"status": "unavailable", "detail": "not configured"},
                    "dynamodb": {"status": "ready", "detail": "connected"},
                },
            },
        )

    result = client(handler).health()
    assert result.status == "degraded"
    assert result.dependencies["registry"].detail == "not configured"


def test_error_normalization_is_safe() -> None:
    api = client(lambda request: httpx.Response(422, json={"detail": [{"msg": "bad route"}]}))
    with pytest.raises(ApiClientError) as captured:
        api.model_info()
    assert captured.value.status_code == 422
    assert captured.value.detail == "bad route"


def test_prediction_and_feedback_post_once_with_typed_results() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/predict":
            return httpx.Response(
                200,
                json={
                    "prediction_id": "one",
                    "delay_probability": 0.3,
                    "predicted_delayed": True,
                    "risk_band": "high",
                    "classification_threshold": 0.18,
                    "route_reliability": [],
                    "support_warning": None,
                    "model_alias": "staging",
                    "model_version": "v0",
                    "model_digest": "digest",
                    "cache_hit": False,
                    "latency_ms": 12.0,
                    "created_at": datetime(2026, 8, 10, tzinfo=UTC).isoformat(),
                },
            )
        return httpx.Response(
            200,
            json={
                "actual_delayed": True,
                "arrival_delay_minutes": 20,
                "notes": None,
                "source": "traveler-ui",
                "feedback_correct": True,
                "feedback_at": datetime(2026, 8, 10, tzinfo=UTC).isoformat(),
                "feedback_revision": 1,
            },
        )

    api = client(handler)
    prediction = api.predict(
        FlightPredictionRequest.model_validate(
            {
                "carrier": "UA",
                "origin": "DEN",
                "destination": "LAX",
                "flight_date": "2026-08-18",
                "scheduled_departure": "08:00",
                "scheduled_arrival": "09:30",
                "scheduled_elapsed_minutes": 150,
                "distance_miles": 862,
            }
        )
    )
    feedback = api.submit_feedback(
        prediction.prediction_id,
        FeedbackRequest(actual_delayed=True, arrival_delay_minutes=20, source="traveler-ui"),
    )
    assert prediction.model_alias == "staging"
    assert feedback.feedback_revision == 1
    assert calls == ["/predict", "/feedback/one"]


def test_transport_timeout_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiClientError, match="timed out"):
        client(handler).model_info()


def test_invalid_success_contract_is_normalized() -> None:
    api = client(lambda request: httpx.Response(200, json={"unexpected": "payload"}))

    with pytest.raises(ApiClientError, match="invalid model information response"):
        api.model_info()
