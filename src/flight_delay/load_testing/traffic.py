"""Deterministic, governed synthetic traffic for the real prediction API path."""

from __future__ import annotations

import json
import math
import random
import time as time_module
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flight_delay.contracts import FlightPredictionRequest, FlightPredictionResponse
from flight_delay.ui import ApiClientError

MAX_REQUEST_COUNT = 500
MAX_RATE_PER_SECOND = 10.0


@dataclass(frozen=True)
class RouteTemplate:
    """Plausible scheduled route values; no outcome or final-test fields are present."""

    carrier: str
    origin: str
    destination: str
    departure: time
    arrival: time
    elapsed_minutes: int
    distance_miles: float


ROUTE_TEMPLATES: tuple[RouteTemplate, ...] = (
    RouteTemplate("UA", "DEN", "LAX", time(8, 0), time(9, 30), 150, 862.0),
    RouteTemplate("WN", "DEN", "PHX", time(10, 15), time(11, 20), 125, 602.0),
    RouteTemplate("AA", "DFW", "ORD", time(13, 10), time(15, 35), 145, 802.0),
    RouteTemplate("DL", "ATL", "LGA", time(7, 20), time(9, 35), 135, 762.0),
    RouteTemplate("AS", "SEA", "SFO", time(16, 5), time(18, 15), 130, 679.0),
)


@dataclass(frozen=True)
class TrafficPlan:
    """Bounded deterministic traffic-generation inputs."""

    count: int = 50
    seed: int = 42
    rate_per_second: float = 2.0
    start_date: date = date(2026, 8, 14)

    def __post_init__(self) -> None:
        if type(self.count) is not int or not 1 <= self.count <= MAX_REQUEST_COUNT:
            raise ValueError(f"count must be between 1 and {MAX_REQUEST_COUNT}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if (
            not isinstance(self.rate_per_second, int | float)
            or isinstance(self.rate_per_second, bool)
            or not math.isfinite(self.rate_per_second)
            or not 0 < self.rate_per_second <= MAX_RATE_PER_SECOND
        ):
            raise ValueError(
                f"rate_per_second must be greater than 0 and at most {MAX_RATE_PER_SECOND}"
            )
        if not isinstance(self.start_date, date):
            raise ValueError("start_date must be a date")


@dataclass(frozen=True)
class TrafficAudit:
    """Local, credential-free audit summary for one dry-run or applied generation."""

    generation_timestamp: str
    mode: str
    traffic_kind: str
    seed: int
    planned_count: int
    attempted_count: int
    successful_count: int
    failed_count: int
    returned_prediction_ids: tuple[str, ...]

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        value = asdict(self)
        value["returned_prediction_ids"] = list(self.returned_prediction_ids)
        return value


def validate_api_base_url(value: str) -> str:
    """Accept an HTTP(S) API origin without credentials, query parameters, or fragments."""

    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("API base URL must be a credential-free HTTP(S) origin")
    return value.strip().rstrip("/")


def _shift_clock(value: time, minutes: int) -> time:
    total = (value.hour * 60 + value.minute + minutes) % (24 * 60)
    return time(total // 60, total % 60)


def generate_requests(
    plan: TrafficPlan, *, templates: Sequence[RouteTemplate] = ROUTE_TEMPLATES
) -> tuple[FlightPredictionRequest, ...]:
    """Create valid, leakage-safe scheduled-flight requests from a deterministic seed."""

    if not templates:
        raise ValueError("at least one route template is required")
    generator = random.Random(plan.seed)
    requests: list[FlightPredictionRequest] = []
    for _ in range(plan.count):
        template = templates[generator.randrange(len(templates))]
        clock_shift = generator.choice((-30, -15, 0, 15, 30))
        elapsed_adjustment = generator.choice((-5, 0, 5))
        requests.append(
            FlightPredictionRequest(
                carrier=template.carrier,
                origin=template.origin,
                destination=template.destination,
                flight_date=plan.start_date + timedelta(days=generator.randrange(14)),
                scheduled_departure=_shift_clock(template.departure, clock_shift),
                scheduled_arrival=_shift_clock(template.arrival, clock_shift),
                scheduled_elapsed_minutes=template.elapsed_minutes + elapsed_adjustment,
                distance_miles=template.distance_miles,
            )
        )
    return tuple(requests)


def run_monitoring_traffic(
    plan: TrafficPlan,
    *,
    apply: bool = False,
    sender: Callable[[FlightPredictionRequest], FlightPredictionResponse] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time_module.sleep,
) -> TrafficAudit:
    """Generate a dry-run plan or send bounded requests through ``POST /predict`` only."""

    requests = generate_requests(plan)
    timestamp = now()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generation timestamp must be timezone-aware")
    if not apply:
        return TrafficAudit(
            generation_timestamp=timestamp.astimezone(UTC).isoformat(),
            mode="dry-run",
            traffic_kind="synthetic_load_test_via_prediction_api",
            seed=plan.seed,
            planned_count=plan.count,
            attempted_count=0,
            successful_count=0,
            failed_count=0,
            returned_prediction_ids=(),
        )
    if sender is None:
        raise ValueError("apply mode requires a prediction sender")

    prediction_ids: list[str] = []
    failed = 0
    interval = 1.0 / float(plan.rate_per_second)
    for index, request in enumerate(requests):
        if index:
            sleep(interval)
        try:
            response = sender(request)
        except ApiClientError:
            failed += 1
        else:
            prediction_ids.append(response.prediction_id)
    return TrafficAudit(
        generation_timestamp=timestamp.astimezone(UTC).isoformat(),
        mode="apply",
        traffic_kind="synthetic_load_test_via_prediction_api",
        seed=plan.seed,
        planned_count=plan.count,
        attempted_count=plan.count,
        successful_count=len(prediction_ids),
        failed_count=failed,
        returned_prediction_ids=tuple(prediction_ids),
    )


def write_audit_summary(path: Path, audit: TrafficAudit) -> None:
    """Atomically write a local audit without API URLs or credentials."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(audit.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
