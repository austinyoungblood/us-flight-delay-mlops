"""Deterministic HTTP smoke sequence for local rehearsal and the live runbook."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from flight_delay.contracts import (
    FeedbackRecord,
    FlightPredictionRequest,
    FlightPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecord,
)


class SmokeError(RuntimeError):
    """A smoke gate failed with a sanitized, actionable message."""


def _json(response: httpx.Response, label: str) -> Any:
    if response.status_code != 200:
        raise SmokeError(f"{label} returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as error:
        raise SmokeError(f"{label} returned invalid JSON") from error


class SmokeRunner:
    """Run the accepted application path without retrying mutating requests."""

    def __init__(self, client: httpx.Client, manifest: dict[str, Any]) -> None:
        self._client = client
        self._manifest = manifest

    def _validated(self, model: type[Any], response: httpx.Response, label: str) -> Any:
        try:
            return model.model_validate(_json(response, label))
        except ValidationError as error:
            raise SmokeError(f"{label} violated its response contract") from error

    def _verify_model_identity(self, info: ModelInfoResponse) -> None:
        expected = self._manifest["model"]
        comparisons = {
            "registry_path": expected["registry_collection"],
            "serving_alias": expected["serving_alias"],
            "registry_version": expected["registry_version"],
            "registry_digest": expected["registry_digest"],
            "bundle_digest": expected["release_bundle_digest"],
            "classification_threshold": expected["classification_threshold"],
            "internal_production_gate_passed": expected["internal_production_gate_passed"],
            "deployment_purpose": expected["deployment_purpose"],
        }
        for field, value in comparisons.items():
            if getattr(info, field) != value:
                raise SmokeError(f"model identity mismatch: {field}")

    def run(
        self,
        *,
        api_base_url: str,
        traveler_base_url: str,
        monitor_base_url: str,
        allow_cache_miss: bool = False,
    ) -> dict[str, Any]:
        """Exercise health, identity, prediction, retrieval, feedback, and UI health."""

        api = api_base_url.rstrip("/")
        traveler = traveler_base_url.rstrip("/")
        monitor = monitor_base_url.rstrip("/")

        health = self._validated(HealthResponse, self._client.get(f"{api}/health"), "health")
        if health.status != "ready" or not health.model_loaded or not health.database_connected:
            raise SmokeError("API health is not ready")

        model_info = self._validated(
            ModelInfoResponse, self._client.get(f"{api}/model-info"), "model-info"
        )
        self._verify_model_identity(model_info)

        request = FlightPredictionRequest(
            carrier="UA",
            origin="DEN",
            destination="LAX",
            flight_date="2026-08-18",
            scheduled_departure="07:30:00",
            scheduled_arrival="09:00:00",
            scheduled_elapsed_minutes=150,
            distance_miles=862,
        )
        payload = request.model_dump(mode="json")
        first = self._validated(
            FlightPredictionResponse,
            self._client.post(f"{api}/predict", json=payload),
            "first prediction",
        )
        second = self._validated(
            FlightPredictionResponse,
            self._client.post(f"{api}/predict", json=payload),
            "second prediction",
        )
        if first.prediction_id == second.prediction_id:
            raise SmokeError("prediction identifiers are not unique")
        if second.cache_hit is not True and not allow_cache_miss:
            raise SmokeError("second identical prediction did not prove a cache hit")

        records = []
        for index, prediction in enumerate((first, second), start=1):
            record = self._validated(
                PredictionRecord,
                self._client.get(f"{api}/predictions/{prediction.prediction_id}"),
                f"prediction {index} retrieval",
            )
            if record.prediction_id != prediction.prediction_id:
                raise SmokeError(f"prediction {index} retrieval identity mismatch")
            records.append(record)

        feedback = self._validated(
            FeedbackRecord,
            self._client.post(
                f"{api}/feedback/{first.prediction_id}",
                json={
                    "actual_delayed": False,
                    "arrival_delay_minutes": 4,
                    "notes": "Brief 08 smoke test",
                    "source": "brief-08-smoke",
                },
            ),
            "feedback",
        )
        after_feedback = self._validated(
            PredictionRecord,
            self._client.get(f"{api}/predictions/{first.prediction_id}"),
            "feedback retrieval",
        )
        if after_feedback.feedback is None:
            raise SmokeError("retrieved prediction does not contain feedback")
        if after_feedback.feedback.feedback_revision != feedback.feedback_revision:
            raise SmokeError("feedback revision was not persisted")

        for label, base_url in (("traveler UI", traveler), ("monitor UI", monitor)):
            response = self._client.get(f"{base_url}/_stcore/health")
            if response.status_code != 200 or response.text.strip().lower() != "ok":
                raise SmokeError(f"{label} health check failed")

        return {
            "status": "passed",
            "model": {
                "registry_collection": model_info.registry_path,
                "serving_alias": model_info.serving_alias,
                "registry_version": model_info.registry_version,
                "registry_digest": model_info.registry_digest,
                "release_bundle_digest": model_info.bundle_digest,
                "internal_production_gate_passed": model_info.internal_production_gate_passed,
                "deployment_purpose": model_info.deployment_purpose,
                "governance_notice": model_info.governance_notice,
            },
            "predictions": {
                "count": 2,
                "ids": [first.prediction_id, second.prediction_id],
                "unique_ids": True,
                "second_cache_hit": second.cache_hit,
                "retrieved_count": len(records),
            },
            "feedback": {
                "persisted": True,
                "prediction_id": first.prediction_id,
                "revision": feedback.feedback_revision,
            },
            "interfaces": {"api": "ready", "traveler": "ready", "monitor": "ready"},
        }
