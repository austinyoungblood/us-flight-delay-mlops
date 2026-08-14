"""Registry-backed FastAPI service with required DynamoDB event persistence."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from flight_delay.contracts import (
    DependencyHealth,
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
from flight_delay.persistence import (
    DynamoDBRepository,
    PersistenceConflict,
    PersistenceError,
)
from flight_delay.serving import RegistryRuntimeError, ServingRuntime, VerifiedRegistryLoader
from flight_delay.serving.cache import PredictionCache


@dataclass(frozen=True)
class Settings:
    """Non-secret runtime configuration."""

    aws_region: str = "us-west-2"
    dynamodb_table: str = "flight-delay-events"
    dynamodb_endpoint_url: str | None = None
    model_download_dir: Path = Path("/tmp/flight-delay-model")
    prediction_cache_maxsize: int = 1_024
    prediction_cache_ttl_seconds: float = 300

    @classmethod
    def from_environment(cls) -> Settings:
        load_dotenv()
        try:
            return cls(
                aws_region=os.getenv("AWS_REGION", "us-west-2"),
                dynamodb_table=os.getenv("DYNAMODB_TABLE", "flight-delay-events"),
                dynamodb_endpoint_url=os.getenv("DYNAMODB_ENDPOINT_URL") or None,
                model_download_dir=Path(os.getenv("MODEL_DOWNLOAD_DIR", "/tmp/flight-delay-model")),
                prediction_cache_maxsize=int(os.getenv("PREDICTION_CACHE_MAXSIZE", "1024")),
                prediction_cache_ttl_seconds=float(
                    os.getenv("PREDICTION_CACHE_TTL_SECONDS", "300")
                ),
            )
        except ValueError as error:
            raise RuntimeError("invalid API runtime configuration") from error


@dataclass
class ApplicationState:
    """Dependencies initialized exactly once by the application lifespan."""

    runtime: ServingRuntime | None = None
    repository: Any | None = None
    cache: PredictionCache | None = None
    dependency_errors: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.runtime is not None and self.repository is not None


def _cache_key(payload: FlightPredictionRequest) -> str:
    encoded = json.dumps(
        payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _model_item(runtime: ServingRuntime) -> dict[str, Any]:
    identity = runtime.identity
    loaded_at = identity["loaded_at"]
    return {
        "pk": f"MODEL#{identity['registry_version']}",
        "registry_path": identity["registry_path"],
        "serving_alias": identity["serving_alias"],
        "registry_version": identity["registry_version"],
        "registry_digest": identity["registry_digest"],
        "source_artifact_digest": identity["source_artifact_digest"],
        "bundle_digest": identity["bundle_digest"],
        "internal_production_gate_passed": identity["internal_production_gate_passed"],
        "deployment_purpose": identity["deployment_purpose"],
        "governance_notice": identity["governance_notice"],
        "selection_lock_sha256": identity["selection_lock_sha256"],
        "route_asset_sha256": identity["route_asset_sha256"],
        "classification_threshold": identity["classification_threshold"],
        "training_partitions": identity["training_partitions"],
        "training_baseline": runtime.training_baseline,
        "release_decision": identity["release_decision"],
        "first_loaded_at": loaded_at,
        "last_loaded_at": loaded_at,
    }


def _public_record(item: dict[str, Any]) -> PredictionRecord:
    payload = dict(item)
    payload.pop("pk", None)
    feedback = payload.get("feedback")
    if feedback is not None and "feedback_revision" not in feedback:
        feedback = {**feedback, "feedback_revision": payload.get("feedback_revision", 1)}
        payload["feedback"] = feedback
    payload.pop("feedback_revision", None)
    return PredictionRecord.model_validate(payload)


def create_app(
    *,
    runtime_loader: Any | None = None,
    repository_factory: Callable[[], Any] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create an application with injectable external dependencies for hermetic tests."""

    settings = settings or Settings.from_environment()
    runtime_loader = runtime_loader or VerifiedRegistryLoader(
        download_root=settings.model_download_dir
    )
    repository_factory = repository_factory or (
        lambda: DynamoDBRepository(
            table_name=settings.dynamodb_table,
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        )
    )
    state_holder = ApplicationState()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        state_holder.dependency_errors.clear()
        try:
            state_holder.runtime = runtime_loader.load()
        except Exception as error:
            state_holder.dependency_errors["registry"] = _safe_dependency_error(error)
        try:
            repository = repository_factory()
            repository.connect()
            state_holder.repository = repository
        except Exception as error:
            state_holder.dependency_errors["dynamodb"] = _safe_dependency_error(error)
        if state_holder.runtime is not None and state_holder.repository is not None:
            try:
                state_holder.repository.put_model_metadata(_model_item(state_holder.runtime))
            except Exception as error:
                state_holder.dependency_errors["dynamodb"] = _safe_dependency_error(error)
                state_holder.repository = None
        state_holder.cache = PredictionCache(
            maxsize=settings.prediction_cache_maxsize,
            ttl_seconds=settings.prediction_cache_ttl_seconds,
        )
        application.state.dependencies = state_holder
        yield
        state_holder.cache.clear()
        if state_holder.repository is not None:
            state_holder.repository.close()
        state_holder.runtime = None
        state_holder.repository = None

    application = FastAPI(
        title="U.S. Flight Delay API",
        description=(
            "Pre-departure flight-delay inference from the exact W&B Registry release selected "
            "by release/release_decision.json, with required DynamoDB event persistence."
        ),
        version="0.6.0",
        lifespan=lifespan,
    )

    def dependencies() -> ApplicationState:
        return state_holder

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def health() -> HealthResponse | JSONResponse:
        """Report readiness and sanitized external dependency state."""

        state = dependencies()
        ready = state.ready
        response = HealthResponse(
            status="ready" if ready else "degraded",
            model_loaded=state.runtime is not None,
            database_connected=state.repository is not None,
            dependencies={
                "registry": DependencyHealth(
                    status="ready" if state.runtime is not None else "unavailable",
                    detail=(
                        "verified Registry runtime loaded"
                        if state.runtime is not None
                        else state.dependency_errors.get("registry", "not initialized")
                    ),
                ),
                "dynamodb": DependencyHealth(
                    status="ready" if state.repository is not None else "unavailable",
                    detail=(
                        "DynamoDB table connected"
                        if state.repository is not None
                        else state.dependency_errors.get("dynamodb", "not initialized")
                    ),
                ),
            },
        )
        if not ready:
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
        return response

    @application.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Return the exact verified serving identity."""

        state = _require_ready(dependencies())
        return ModelInfoResponse.model_validate(state.runtime.identity)

    @application.post(
        "/predict",
        response_model=FlightPredictionResponse,
    )
    async def predict(
        payload: FlightPredictionRequest,
        traffic_source: Annotated[TrafficSource, Header(alias="X-Traffic-Source")] = (
            TrafficSource.API_UNSPECIFIED
        ),
    ) -> FlightPredictionResponse:
        """Infer and persist one uniquely observable prediction event."""

        if traffic_source is TrafficSource.LEGACY_UNATTRIBUTED:
            raise HTTPException(
                status_code=422,
                detail="legacy_unattributed is reserved for historical persisted records",
            )
        state = _require_ready(dependencies())
        started = time.perf_counter()
        created_at = _utc_now()
        prediction_id = str(uuid.uuid4())
        key = _cache_key(payload)
        cached = state.cache.get(key)
        cache_hit = cached is not None
        inference_started = time.perf_counter()
        try:
            result = cached if cached is not None else state.runtime.predict(payload)
            if cached is None:
                state.cache.put(key, result)
        except Exception as error:
            state.repository.put_error(
                {
                    "pk": f"ERROR#{prediction_id}",
                    "event_date": created_at.date().isoformat(),
                    "created_at": created_at,
                    "request_status": "inference_error",
                    "error_type": type(error).__name__,
                    "traffic_source": traffic_source.value,
                }
            )
            raise HTTPException(status_code=503, detail="model inference is unavailable") from error
        inference_ms = (time.perf_counter() - inference_started) * 1_000
        identity = state.runtime.identity
        persistence_started = time.perf_counter()
        preliminary_persistence_ms = max((time.perf_counter() - persistence_started) * 1_000, 0.0)
        preliminary_total_ms = (time.perf_counter() - started) * 1_000
        record = {
            "pk": f"PREDICTION#{prediction_id}",
            "prediction_id": prediction_id,
            "event_date": created_at.date(),
            "created_at": created_at,
            "traffic_source": traffic_source.value,
            "request": payload.model_dump(mode="python"),
            **result,
            "classification_threshold": state.runtime.threshold,
            "model_alias": identity["serving_alias"],
            "model_version": identity["registry_version"],
            "model_digest": identity["registry_digest"],
            "bundle_digest": identity["bundle_digest"],
            "cache_hit": cache_hit,
            "latency_ms": preliminary_total_ms,
            "inference_latency_ms": inference_ms,
            "persistence_latency_ms": preliminary_persistence_ms,
            "total_latency_ms": preliminary_total_ms,
            "request_status": "success",
            "feedback": None,
        }
        try:
            state.repository.put_prediction(record)
        except PersistenceConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="prediction persistence is unavailable"
            ) from error
        persistence_ms = (time.perf_counter() - persistence_started) * 1_000
        total_ms = (time.perf_counter() - started) * 1_000
        record["persistence_latency_ms"] = persistence_ms
        record["total_latency_ms"] = total_ms
        record["latency_ms"] = total_ms
        public = {
            name: record[name] for name in FlightPredictionResponse.model_fields if name in record
        }
        return FlightPredictionResponse.model_validate(public)

    @application.get("/route-reliability", response_model=list[RouteReliability])
    async def route_reliability(
        origin: str = Query(pattern=r"^[A-Za-z]{3}$"),
        destination: str = Query(pattern=r"^[A-Za-z]{3}$"),
        carrier: str | None = Query(default=None, pattern=r"^[A-Za-z0-9]{2}$"),
    ) -> list[RouteReliability]:
        """Return route evidence independently from prediction inference."""

        state = _require_ready(dependencies())
        evidence = state.runtime.route_reliability(
            origin.upper(), destination.upper(), carrier.upper() if carrier else None
        )
        if not evidence:
            raise HTTPException(status_code=404, detail="route evidence not found")
        return evidence

    @application.get("/predictions/{prediction_id}", response_model=PredictionRecord)
    async def get_prediction(prediction_id: str) -> PredictionRecord:
        """Strongly consistently retrieve one prediction event."""

        state = _require_ready(dependencies())
        try:
            item = state.repository.get_prediction(prediction_id)
        except PersistenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if item is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        return _public_record(item)

    @application.post("/feedback/{prediction_id}", response_model=FeedbackRecord)
    async def submit_feedback(prediction_id: str, feedback: FeedbackRequest) -> FeedbackRecord:
        """Conditionally revision observed outcome data on an existing event."""

        state = _require_ready(dependencies())
        try:
            current = state.repository.get_prediction(prediction_id)
            if current is None:
                raise HTTPException(status_code=404, detail="prediction not found")
            feedback_at = _utc_now()
            payload = {
                **feedback.model_dump(mode="python"),
                "feedback_correct": feedback.actual_delayed == current["predicted_delayed"],
                "feedback_at": feedback_at,
            }
            updated = state.repository.update_feedback(prediction_id, payload)
        except PersistenceConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except PersistenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if updated is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        stored = dict(updated["feedback"])
        stored.setdefault("feedback_revision", updated["feedback_revision"])
        return FeedbackRecord.model_validate(stored)

    return application


def _safe_dependency_error(error: Exception) -> str:
    if isinstance(error, RegistryRuntimeError | PersistenceError | RuntimeError | ValueError):
        return str(error)
    return f"{type(error).__name__}: dependency unavailable"


def _require_ready(state: ApplicationState) -> ApplicationState:
    if not state.ready:
        raise HTTPException(status_code=503, detail="service dependencies are unavailable")
    return state


app = create_app()
