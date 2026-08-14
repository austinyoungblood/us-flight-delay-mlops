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


def test_client_rejects_nonpositive_timeouts_and_closes() -> None:
    with pytest.raises(ValueError, match="timeouts"):
        FlightDelayApiClient("http://api", connect_timeout_seconds=0)
    api = client(lambda request: httpx.Response(200, json={}))
    api.close()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="not-json"), "HTTP 500"),
        (httpx.Response(500, json={"detail": "safe failure"}), "safe failure"),
        (httpx.Response(500, json={"detail": []}), "validation failed"),
        (httpx.Response(500, json={"detail": 42}), "dependency is unavailable"),
    ],
)
def test_error_payload_shapes_are_safely_normalized(response: httpx.Response, message: str) -> None:
    with pytest.raises(ApiClientError, match=message):
        client(lambda request: response).model_info()


def test_transport_connection_error_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection details", request=request)

    with pytest.raises(ApiClientError, match="unreachable"):
        client(handler).model_info()


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (httpx.ReadTimeout, "health check timed out"),
        (httpx.ConnectError, "unreachable"),
    ],
)
def test_health_transport_failures_are_normalized(
    error_type: type[httpx.RequestError], message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("transport failure", request=request)

    with pytest.raises(ApiClientError, match=message):
        client(handler).health()


def test_health_rejects_unexpected_status_and_invalid_contract() -> None:
    with pytest.raises(ApiClientError, match="maintenance") as captured:
        client(lambda request: httpx.Response(429, json={"detail": "maintenance"})).health()
    assert captured.value.status_code == 429
    with pytest.raises(ApiClientError, match="invalid health"):
        client(lambda request: httpx.Response(200, json={})).health()


def test_route_reliability_and_prediction_retrieval_are_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/route-reliability":
            assert dict(request.url.params) == {"origin": "DEN", "destination": "LAX"}
            return httpx.Response(
                200,
                json=[
                    {
                        "scope": "all_carriers",
                        "carrier": None,
                        "origin": "DEN",
                        "destination": "LAX",
                        "eligible_flights": 100,
                        "on_time_count": 70,
                        "on_time_rate": 0.7,
                        "delayed_count": 30,
                        "delayed_rate": 0.3,
                        "mean_arrival_delay_minutes": 8.0,
                        "median_arrival_delay_minutes": 2.0,
                        "meets_minimum_support": True,
                    }
                ],
            )
        assert request.url.path == "/predictions/one"
        return httpx.Response(
            200,
            json={
                "prediction_id": "one",
                "delay_probability": 0.2,
                "predicted_delayed": True,
                "risk_band": "medium",
                "classification_threshold": 0.18,
                "route_reliability": [],
                "support_warning": None,
                "model_alias": "production",
                "model_version": "v0",
                "model_digest": "digest",
                "cache_hit": False,
                "latency_ms": 12.0,
                "created_at": "2026-08-10T00:00:00Z",
                "request": {
                    "carrier": "UA",
                    "origin": "DEN",
                    "destination": "LAX",
                    "flight_date": "2026-08-10",
                    "scheduled_departure": "08:00:00",
                    "scheduled_arrival": "09:30:00",
                    "scheduled_elapsed_minutes": 150,
                    "distance_miles": 862.0,
                },
                "event_date": "2026-08-10",
                "request_status": "success",
                "inference_latency_ms": 5.0,
                "persistence_latency_ms": 2.0,
                "total_latency_ms": 12.0,
                "bundle_digest": "bundle",
                "feedback": None,
            },
        )

    api = client(handler)
    assert api.route_reliability(origin="DEN", destination="LAX")[0].on_time_rate == 0.7
    assert api.get_prediction("one").request.carrier == "UA"


def test_route_reliability_rejects_non_list_payload() -> None:
    api = client(lambda request: httpx.Response(200, json={"route": "DEN-LAX"}))
    with pytest.raises(ApiClientError, match="invalid route reliability"):
        api.route_reliability(origin="DEN", destination="LAX", carrier="UA")
