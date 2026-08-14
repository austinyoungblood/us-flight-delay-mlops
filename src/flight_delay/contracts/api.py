"""Pydantic contracts shared by the API and future presentation clients."""

from datetime import date, datetime, time
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

CarrierCode = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{2}$")]
AirportCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class StrictContract(BaseModel):
    """Base contract that rejects unknown fields and validates assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RiskBand(StrEnum):
    """Human-readable probability category."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrafficSource(StrEnum):
    """Controlled origin for one persisted prediction request."""

    TRAVELER_UI = "traveler_ui"
    SYNTHETIC_LOAD_TEST = "synthetic_load_test"
    API_UNSPECIFIED = "api_unspecified"
    LEGACY_UNATTRIBUTED = "legacy_unattributed"


class FlightPredictionRequest(StrictContract):
    """Scheduled flight facts available before departure."""

    carrier: CarrierCode
    origin: AirportCode
    destination: AirportCode
    flight_date: date
    scheduled_departure: time
    scheduled_arrival: time
    scheduled_elapsed_minutes: int = Field(gt=0, le=1_500)
    distance_miles: float = Field(gt=0, le=10_000)

    @field_validator("carrier", "origin", "destination", mode="before")
    @classmethod
    def normalize_codes(cls, value: object) -> object:
        """Normalize string codes before their constrained patterns run."""

        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("scheduled_departure", "scheduled_arrival")
    @classmethod
    def reject_timezone_aware_times(cls, value: time) -> time:
        """Keep schedule inputs local and timezone-naive in this phase."""

        if value.tzinfo is not None:
            raise ValueError("scheduled times must not include a timezone")
        return value

    @field_validator("destination")
    @classmethod
    def destination_must_differ_from_origin(cls, value: str, info: object) -> str:
        """Reject a route whose origin and destination are identical."""

        origin = getattr(info, "data", {}).get("origin")
        if origin == value:
            raise ValueError("destination must differ from origin")
        return value


class RouteReliability(StrictContract):
    """Historical reliability summary; it is not a live flight-status guarantee."""

    scope: Literal["carrier_route", "all_carriers"]
    carrier: CarrierCode | None = None
    origin: AirportCode
    destination: AirportCode
    eligible_flights: int = Field(ge=0)
    on_time_count: int = Field(ge=0)
    on_time_rate: float = Field(ge=0, le=1)
    delayed_count: int = Field(ge=0)
    delayed_rate: float = Field(ge=0, le=1)
    mean_arrival_delay_minutes: float | None = None
    median_arrival_delay_minutes: float | None = None
    meets_minimum_support: bool


class FlightPredictionResponse(StrictContract):
    """Public prediction response with immutable serving provenance."""

    prediction_id: str = Field(min_length=1)
    delay_probability: float = Field(ge=0, le=1)
    predicted_delayed: bool
    risk_band: RiskBand
    classification_threshold: float = Field(gt=0, lt=1)
    route_reliability: list[RouteReliability] = Field(default_factory=list)
    support_warning: str | None = None
    model_alias: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)
    cache_hit: bool
    latency_ms: float = Field(ge=0)
    created_at: datetime


class FeedbackRequest(StrictContract):
    """Observed post-flight outcome submitted after a prediction."""

    actual_delayed: bool
    arrival_delay_minutes: float | None = Field(default=None, ge=-300, le=2_880)
    notes: str | None = Field(default=None, max_length=1_000)
    source: str = Field(default="user", min_length=1, max_length=100)


class FeedbackRecord(FeedbackRequest):
    """Persisted, revisioned post-flight outcome."""

    feedback_correct: bool
    feedback_at: datetime
    feedback_revision: int = Field(ge=1)


class PredictionRecord(FlightPredictionResponse):
    """Stored event returned by the prediction retrieval endpoint."""

    traffic_source: TrafficSource = TrafficSource.LEGACY_UNATTRIBUTED
    request: FlightPredictionRequest
    event_date: date
    request_status: Literal["success"]
    inference_latency_ms: float = Field(ge=0)
    persistence_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    bundle_digest: str = Field(min_length=1)
    feedback: FeedbackRecord | None = None


class DependencyHealth(StrictContract):
    """Sanitized startup state for one external dependency."""

    status: Literal["ready", "unavailable"]
    detail: str


class HealthResponse(StrictContract):
    """Readiness status for model and persistence dependencies."""

    service: Literal["flight-delay-api"] = "flight-delay-api"
    status: Literal["ready", "degraded"]
    model_loaded: bool
    database_connected: bool
    dependencies: dict[str, DependencyHealth]


class ModelInfoResponse(StrictContract):
    """Exact immutable identity of the active staging or production runtime."""

    registry_path: str
    serving_alias: str
    registry_version: str
    registry_digest: str
    source_artifact_digest: str
    bundle_digest: str
    selection_lock_sha256: str
    route_asset_sha256: str
    classification_threshold: float
    feature_schema: list[str]
    training_partitions: dict[str, str]
    release_decision: dict[str, Any]
    release_git_sha: str
    loaded_at: datetime
    internal_production_gate_passed: bool
    deployment_purpose: Literal["academic_demo", "operational"]
    governance_notice: str
    serving_stage_notice: str


class ErrorResponse(StrictContract):
    """Sanitized API error body."""

    detail: str
