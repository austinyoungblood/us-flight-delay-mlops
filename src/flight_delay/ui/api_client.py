"""Typed, non-retrying HTTP client for the traveler Streamlit application."""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from flight_delay.contracts import (
    FeedbackRecord,
    FeedbackRequest,
    FlightPredictionRequest,
    FlightPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecord,
    RouteReliability,
    TrafficSource,
)

ContractT = TypeVar("ContractT", bound=BaseModel)


class ApiClientError(RuntimeError):
    """Normalized API or transport error safe for presentation."""

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _safe_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return f"API request failed with HTTP {response.status_code}."
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, str):
        return detail[:500]
    if isinstance(detail, list):
        messages = [
            str(item.get("msg", "invalid value")) for item in detail if isinstance(item, dict)
        ]
        return "; ".join(messages)[:500] or "Request validation failed."
    return "API dependency is unavailable."


def _contract(model: type[ContractT], payload: object, label: str) -> ContractT:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise ApiClientError(f"The API returned an invalid {label} response.") from error


class FlightDelayApiClient:
    """Traveler client; all prediction and feedback operations go through FastAPI."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout_seconds: float = 5,
        read_timeout_seconds: float = 15,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ValueError("API timeouts must be positive")
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise ApiClientError("The API request timed out. Please try again.") from error
        except httpx.RequestError as error:
            raise ApiClientError("The API is unreachable. Check its health and URL.") from error
        if response.is_error:
            raise ApiClientError(_safe_detail(response), status_code=response.status_code)
        return response

    def health(self) -> HealthResponse:
        """Return structured readiness, including a deliberate HTTP 503 response."""

        try:
            response = self._client.get("/health")
        except httpx.TimeoutException as error:
            raise ApiClientError("The API health check timed out.") from error
        except httpx.RequestError as error:
            raise ApiClientError("The API is unreachable. Check its health and URL.") from error
        if response.status_code not in {200, 503}:
            raise ApiClientError(_safe_detail(response), status_code=response.status_code)
        try:
            return HealthResponse.model_validate(response.json())
        except ValueError as error:
            raise ApiClientError("The API returned an invalid health response.") from error

    def model_info(self) -> ModelInfoResponse:
        return _contract(
            ModelInfoResponse, self._request("GET", "/model-info").json(), "model information"
        )

    def predict(
        self,
        request: FlightPredictionRequest,
        *,
        traffic_source: TrafficSource = TrafficSource.API_UNSPECIFIED,
    ) -> FlightPredictionResponse:
        if traffic_source is TrafficSource.LEGACY_UNATTRIBUTED:
            raise ValueError(
                "legacy_unattributed is historical-only and cannot identify "
                "a new prediction request"
            )
        return _contract(
            FlightPredictionResponse,
            self._request(
                "POST",
                "/predict",
                json=request.model_dump(mode="json"),
                headers={"X-Traffic-Source": traffic_source.value},
            ).json(),
            "prediction",
        )

    def route_reliability(
        self, *, origin: str, destination: str, carrier: str | None = None
    ) -> list[RouteReliability]:
        params = {"origin": origin, "destination": destination}
        if carrier:
            params["carrier"] = carrier
        payload = self._request("GET", "/route-reliability", params=params).json()
        if not isinstance(payload, list):
            raise ApiClientError("The API returned an invalid route reliability response.")
        return [_contract(RouteReliability, item, "route reliability") for item in payload]

    def get_prediction(self, prediction_id: str) -> PredictionRecord:
        return _contract(
            PredictionRecord,
            self._request("GET", f"/predictions/{prediction_id}").json(),
            "prediction retrieval",
        )

    def submit_feedback(self, prediction_id: str, feedback: FeedbackRequest) -> FeedbackRecord:
        return _contract(
            FeedbackRecord,
            self._request(
                "POST",
                f"/feedback/{prediction_id}",
                json=feedback.model_dump(mode="json"),
            ).json(),
            "feedback",
        )
