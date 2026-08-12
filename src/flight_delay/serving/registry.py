"""Release-decision-driven, fail-closed W&B Registry runtime loader."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import wandb

from flight_delay.contracts import FlightPredictionRequest, RiskBand, RouteReliability
from flight_delay.data.download import sha256_file
from flight_delay.features.engineering import derive_schedule_features
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.release import ReleaseGuardError, verify_locked_files

ACADEMIC_DEMO_NOTICE = (
    "Academic demonstration — W&B production alias used for course deployment; "
    "the model did not pass the project's stricter internal production-quality gate."
)


class RegistryRuntimeError(RuntimeError):
    """Raised when the selected release cannot be verified and served safely."""


@dataclass(frozen=True)
class ReleaseDecision:
    """Immutable serving control-plane values committed by Brief 05."""

    registry_path: str
    serving_alias: str
    registry_version: str
    registry_digest: str
    source_artifact_digest: str
    bundle_digest: str
    internal_production_gate_passed: bool
    deployment_purpose: str
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ReleaseDecision:
        """Parse and validate the committed release decision."""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegistryRuntimeError("release decision is unavailable or invalid") from error
        required = {
            "registry_path",
            "serving_alias",
            "registry_version",
            "registry_digest",
            "source_artifact_digest",
            "bundle_digest",
            "final_test_passed",
            "internal_production_gate_passed",
            "deployment_purpose",
            "course_production_alias",
            "failed_gates",
            "final_test_failure_summary",
        }
        if not isinstance(raw, dict) or required - raw.keys():
            raise RegistryRuntimeError("release decision schema is incomplete")
        string_fields = {
            "registry_path",
            "serving_alias",
            "registry_version",
            "registry_digest",
            "source_artifact_digest",
            "bundle_digest",
            "deployment_purpose",
            "final_test_failure_summary",
        }
        if any(not isinstance(raw[key], str) or not raw[key] for key in string_fields):
            raise RegistryRuntimeError("release decision identity fields must be non-empty strings")
        boolean_fields = (
            "final_test_passed",
            "internal_production_gate_passed",
            "course_production_alias",
        )
        if any(not isinstance(raw[key], bool) for key in boolean_fields):
            raise RegistryRuntimeError("release decision governance fields must be boolean")
        if raw["internal_production_gate_passed"] != raw["final_test_passed"]:
            raise RegistryRuntimeError("internal production gate conflicts with historical result")
        if raw["deployment_purpose"] not in {"academic_demo", "operational"}:
            raise RegistryRuntimeError("release decision deployment purpose is unsupported")
        failed_gates = raw["failed_gates"]
        if not isinstance(failed_gates, list) or any(
            not isinstance(value, str) or not value for value in failed_gates
        ):
            raise RegistryRuntimeError("release decision failed_gates must be a string list")
        academic_course_release = (
            raw["deployment_purpose"] == "academic_demo"
            and raw["course_production_alias"]
            and not raw["internal_production_gate_passed"]
        )
        expected_alias = (
            "production"
            if raw["internal_production_gate_passed"] or academic_course_release
            else "staging"
        )
        if raw["serving_alias"] != expected_alias:
            raise RegistryRuntimeError("release decision serving alias conflicts with gate result")
        return cls(
            registry_path=raw["registry_path"],
            serving_alias=raw["serving_alias"],
            registry_version=raw["registry_version"],
            registry_digest=raw["registry_digest"],
            source_artifact_digest=raw["source_artifact_digest"],
            bundle_digest=raw["bundle_digest"],
            internal_production_gate_passed=raw["internal_production_gate_passed"],
            deployment_purpose=raw["deployment_purpose"],
            raw=dict(raw),
        )


def risk_band(probability: float, threshold: float) -> RiskBand:
    """Map a finite probability to the release-defined relative risk bands."""

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RegistryRuntimeError("model returned an invalid probability")
    probability_decimal = Decimal(str(probability))
    threshold_decimal = Decimal(str(threshold))
    if probability_decimal < Decimal("0.75") * threshold_decimal:
        return RiskBand.LOW
    if probability_decimal < Decimal("1.25") * threshold_decimal:
        return RiskBand.MEDIUM
    return RiskBand.HIGH


def _feature_frame(request: FlightPredictionRequest, schema: list[str]) -> pd.DataFrame:
    departure = request.scheduled_departure.hour * 100 + request.scheduled_departure.minute
    arrival = request.scheduled_arrival.hour * 100 + request.scheduled_arrival.minute
    raw = pd.DataFrame(
        [
            {
                "Reporting_Airline": request.carrier,
                "Origin": request.origin,
                "Dest": request.destination,
                "Month": request.flight_date.month,
                "DayofMonth": request.flight_date.day,
                "DayOfWeek": request.flight_date.isoweekday(),
                "CRSDepTime": departure,
                "CRSArrTime": arrival,
                "CRSElapsedTime": request.scheduled_elapsed_minutes,
                "Distance": request.distance_miles,
            }
        ]
    )
    derived = derive_schedule_features(raw)
    return derived.loc[:, schema]


def _nullable_number(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


@dataclass
class ServingRuntime:
    """Verified model, release assets, and deterministic inference helpers."""

    model: Any
    routes: pd.DataFrame
    threshold: float
    feature_schema: list[str]
    metadata: dict[str, Any]
    training_baseline: dict[str, Any]
    development_metrics: dict[str, Any]
    model_card: str
    release_policy: str
    identity: dict[str, Any]

    def predict_probability(self, request: FlightPredictionRequest) -> float:
        """Run leakage-safe model inference and reject invalid output."""

        probabilities = self.model.predict_proba(_feature_frame(request, self.feature_schema))
        try:
            probability = float(probabilities[0][1])
        except (IndexError, TypeError, ValueError) as error:
            raise RegistryRuntimeError("model returned an invalid inference shape") from error
        risk_band(probability, self.threshold)
        return probability

    def route_reliability(
        self, origin: str, destination: str, carrier: str | None = None
    ) -> list[RouteReliability]:
        """Return carrier-specific and all-carrier historical evidence when available."""

        matches = self.routes[
            (self.routes["Origin"] == origin) & (self.routes["Dest"] == destination)
        ]
        if carrier:
            matches = matches[
                (matches["Reporting_Airline"].isna()) | (matches["Reporting_Airline"] == carrier)
            ]
        else:
            matches = matches[matches["Reporting_Airline"].isna()]
        records: list[RouteReliability] = []
        for row in matches.to_dict(orient="records"):
            row_carrier = row.get("Reporting_Airline")
            records.append(
                RouteReliability(
                    scope=row["scope"],
                    carrier=None if pd.isna(row_carrier) else str(row_carrier),
                    origin=str(row["Origin"]),
                    destination=str(row["Dest"]),
                    eligible_flights=int(row["eligible_flights"]),
                    on_time_count=int(row["on_time_count"]),
                    on_time_rate=float(row["on_time_rate"]),
                    delayed_count=int(row["delayed_count"]),
                    delayed_rate=float(row["delayed_rate"]),
                    mean_arrival_delay_minutes=_nullable_number(
                        row.get("mean_arrival_delay_minutes")
                    ),
                    median_arrival_delay_minutes=_nullable_number(
                        row.get("median_arrival_delay_minutes")
                    ),
                    meets_minimum_support=bool(row["meets_minimum_support"]),
                )
            )
        return sorted(records, key=lambda item: item.scope)

    def predict(self, request: FlightPredictionRequest) -> dict[str, Any]:
        """Compute the cacheable probability and route context."""

        probability = self.predict_probability(request)
        reliability = self.route_reliability(request.origin, request.destination, request.carrier)
        warning = None
        if not reliability:
            warning = "No historical route evidence is available."
        elif not all(item.meets_minimum_support for item in reliability):
            warning = "One or more route estimates are below minimum historical support."
        return {
            "delay_probability": probability,
            "predicted_delayed": probability >= self.threshold,
            "risk_band": risk_band(probability, self.threshold),
            "route_reliability": reliability,
            "support_warning": warning,
        }


class VerifiedRegistryLoader:
    """Resolve, download, and verify exactly the artifact selected by Brief 05."""

    def __init__(
        self,
        *,
        decision_path: Path = Path("release/release_decision.json"),
        committed_lock_path: Path = Path("release/selection_lock.json"),
        download_root: Path = Path("/tmp/flight-delay-model"),
        api_factory: Callable[[], Any] = wandb.Api,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.decision_path = decision_path
        self.committed_lock_path = committed_lock_path
        self.download_root = download_root
        self.api_factory = api_factory
        self.now = now

    def load(self) -> ServingRuntime:
        """Build a serving runtime, failing closed on any identity or hash drift."""

        decision = ReleaseDecision.load(self.decision_path)
        try:
            artifact = self.api_factory().artifact(
                f"{decision.registry_path}:{decision.serving_alias}"
            )
        except Exception as error:
            raise RegistryRuntimeError("Registry alias could not be resolved") from error
        if (
            artifact.version != decision.registry_version
            or artifact.digest != decision.registry_digest
        ):
            raise RegistryRuntimeError("Registry alias version or digest drift detected")
        source = artifact.source_artifact if getattr(artifact, "is_link", False) else artifact
        if source.digest != decision.source_artifact_digest:
            raise RegistryRuntimeError("Registry source artifact digest drift detected")
        target = self.download_root / f"{decision.registry_version}-{decision.registry_digest}"
        target.mkdir(parents=True, exist_ok=True)
        try:
            root = Path(artifact.download(root=str(target)))
            lock = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
            committed_lock_hash = sha256_file(self.committed_lock_path)
            if sha256_file(root / "selection_lock.json") != committed_lock_hash:
                raise RegistryRuntimeError("downloaded selection lock differs from committed lock")
            if lock.get("aggregate_bundle_digest") != decision.bundle_digest:
                raise RegistryRuntimeError("release bundle digest differs from release decision")
            verify_locked_files(root, lock)
        except RegistryRuntimeError:
            raise
        except (OSError, ValueError, ReleaseGuardError) as error:
            raise RegistryRuntimeError("Registry artifact hash verification failed") from error

        bundle = root / "model_bundle"
        try:
            schema = json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8"))[
                "features"
            ]
            validate_model_features(schema)
            threshold = float(
                json.loads((bundle / "threshold.json").read_text(encoding="utf-8"))["threshold"]
            )
            if threshold != float(lock["threshold"]) or not 0 < threshold < 1:
                raise RegistryRuntimeError("artifact threshold does not match the selection lock")
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            baseline = json.loads((bundle / "training_baseline.json").read_text(encoding="utf-8"))
            metrics = json.loads((bundle / "metrics_development.json").read_text(encoding="utf-8"))
            model = joblib.load(bundle / "model.joblib")
            routes = pd.read_parquet(root / "route_stats.parquet")
            model_card = (bundle / "MODEL_CARD.md").read_text(encoding="utf-8")
            release_policy = (bundle / "release_policy.yaml").read_text(encoding="utf-8")
        except RegistryRuntimeError:
            raise
        except Exception as error:
            raise RegistryRuntimeError("verified Registry bundle could not be loaded") from error

        loaded_at = self.now()
        identity = {
            "registry_path": decision.registry_path,
            "serving_alias": decision.serving_alias,
            "registry_version": decision.registry_version,
            "registry_digest": decision.registry_digest,
            "source_artifact_digest": decision.source_artifact_digest,
            "bundle_digest": decision.bundle_digest,
            "internal_production_gate_passed": decision.internal_production_gate_passed,
            "deployment_purpose": decision.deployment_purpose,
            "selection_lock_sha256": committed_lock_hash,
            "route_asset_sha256": lock["file_hashes"]["route_stats.parquet"],
            "classification_threshold": threshold,
            "feature_schema": list(schema),
            "training_partitions": dict(metadata["partitions"]),
            "release_decision": decision.raw,
            "release_git_sha": _git_sha(),
            "loaded_at": loaded_at,
            "governance_notice": (
                ACADEMIC_DEMO_NOTICE
                if decision.deployment_purpose == "academic_demo"
                else "Operational release certified by the internal production-quality gate."
            ),
            "serving_stage_notice": (
                ACADEMIC_DEMO_NOTICE
                if decision.deployment_purpose == "academic_demo"
                else "Operational release certified by the internal production-quality gate."
            ),
        }
        runtime = ServingRuntime(
            model=model,
            routes=routes,
            threshold=threshold,
            feature_schema=list(schema),
            metadata=metadata,
            training_baseline=baseline,
            development_metrics=metrics,
            model_card=model_card,
            release_policy=release_policy,
            identity=identity,
        )
        runtime.predict_probability(
            FlightPredictionRequest(
                carrier="UA",
                origin="DEN",
                destination="LAX",
                flight_date=date(2026, 1, 15),
                scheduled_departure=time(8, 0),
                scheduled_arrival=time(9, 30),
                scheduled_elapsed_minutes=150,
                distance_miles=862,
            )
        )
        return runtime


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
