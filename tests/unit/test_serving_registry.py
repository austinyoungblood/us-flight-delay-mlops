from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from flight_delay.contracts import FlightPredictionRequest, RiskBand
from flight_delay.data.download import sha256_file
from flight_delay.modeling.release import aggregate_digest
from flight_delay.serving.registry import (
    RegistryRuntimeError,
    ReleaseDecision,
    ServingRuntime,
    VerifiedRegistryLoader,
    risk_band,
)


class ConstantModel:
    def __init__(self, probability: float = 0.2) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.probability, self.probability], (len(frame), 1))


class FakeArtifact:
    version = "v0"
    digest = "registry-digest"
    is_link = True

    def __init__(self, root: Path) -> None:
        self.root = root
        self.source_artifact = type("Source", (), {"digest": "source-digest", "version": "v0"})()

    def download(self, *, root: str) -> str:
        return str(self.root)


class FakeApi:
    def __init__(self, artifact: FakeArtifact) -> None:
        self.expected_path = "wandb-registry-Model/us-flight-arrival-delay-15m:production"
        self.value = artifact

    def artifact(self, path: str) -> FakeArtifact:
        assert path == self.expected_path
        return self.value


def request() -> FlightPredictionRequest:
    return FlightPredictionRequest(
        carrier="UA",
        origin="DEN",
        destination="LAX",
        flight_date=date(2026, 8, 18),
        scheduled_departure=time(7, 30),
        scheduled_arrival=time(9, 0),
        scheduled_elapsed_minutes=150,
        distance_miles=862,
    )


def release_fixture(tmp_path: Path) -> tuple[Path, Path, Path, FakeArtifact]:
    root = tmp_path / "artifact"
    bundle = root / "model_bundle"
    bundle.mkdir(parents=True)
    schema = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_departure_minute_bucket",
        "scheduled_arrival_minute_bucket",
        "is_weekend",
        "scheduled_departure_sin",
        "scheduled_departure_cos",
        "scheduled_arrival_sin",
        "scheduled_arrival_cos",
    ]
    (bundle / "feature_schema.json").write_text(json.dumps({"features": schema}))
    (bundle / "threshold.json").write_text(json.dumps({"threshold": 0.2}))
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "partitions": {
                    "base_fit": "2025-01-01/2025-10-31",
                    "calibration": "2025-11-01/2025-11-15",
                }
            }
        )
    )
    (bundle / "training_baseline.json").write_text(json.dumps({"row_count": 10}))
    (bundle / "metrics_development.json").write_text(json.dumps({"roc_auc": 0.6}))
    (bundle / "MODEL_CARD.md").write_text("staging model")
    (bundle / "release_policy.yaml").write_text("schema_version: 1\n")
    joblib.dump(ConstantModel(), bundle / "model.joblib")
    routes = pd.DataFrame(
        [
            {
                "scope": "carrier_route",
                "Reporting_Airline": "UA",
                "Origin": "DEN",
                "Dest": "LAX",
                "eligible_flights": 100,
                "on_time_count": 70,
                "on_time_rate": 0.7,
                "delayed_count": 30,
                "delayed_rate": 0.3,
                "mean_arrival_delay_minutes": 5.0,
                "median_arrival_delay_minutes": 0.0,
                "meets_minimum_support": True,
            },
            {
                "scope": "all_carriers",
                "Reporting_Airline": None,
                "Origin": "DEN",
                "Dest": "LAX",
                "eligible_flights": 300,
                "on_time_count": 225,
                "on_time_rate": 0.75,
                "delayed_count": 75,
                "delayed_rate": 0.25,
                "mean_arrival_delay_minutes": 3.0,
                "median_arrival_delay_minutes": 0.0,
                "meets_minimum_support": True,
            },
        ]
    )
    routes.to_parquet(root / "route_stats.parquet")
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    lock = {
        "threshold": 0.2,
        "file_hashes": hashes,
        "aggregate_bundle_digest": aggregate_digest(hashes),
    }
    lock_path = root / "selection_lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    committed = tmp_path / "selection_lock.json"
    committed.write_bytes(lock_path.read_bytes())
    decision = {
        "registry_path": "wandb-registry-Model/us-flight-arrival-delay-15m",
        "serving_alias": "production",
        "registry_version": "v0",
        "registry_digest": "registry-digest",
        "source_artifact_digest": "source-digest",
        "bundle_digest": lock["aggregate_bundle_digest"],
        "final_test_passed": False,
        "internal_production_gate_passed": False,
        "deployment_purpose": "academic_demo",
        "course_production_alias": True,
        "failed_gates": ["brier_skill_score"],
        "final_test_failure_summary": "Historical final-test production gates failed.",
    }
    decision_path = tmp_path / "release_decision.json"
    decision_path.write_text(json.dumps(decision))
    return decision_path, committed, root, FakeArtifact(root)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.14, RiskBand.LOW),
        (0.15, RiskBand.MEDIUM),
        (0.249, RiskBand.MEDIUM),
        (0.25, RiskBand.HIGH),
    ],
)
def test_risk_band_boundaries(probability: float, expected: RiskBand) -> None:
    assert risk_band(probability, 0.2) is expected


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_risk_band_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(RegistryRuntimeError, match="invalid probability"):
        risk_band(probability, 0.2)


def test_release_decision_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text('{"serving_alias":"staging"}')
    with pytest.raises(RegistryRuntimeError, match="incomplete"):
        ReleaseDecision.load(path)


def test_verified_loader_uses_alias_and_loads_bundle(tmp_path: Path) -> None:
    decision, committed, _, artifact = release_fixture(tmp_path)
    api = FakeApi(artifact)
    loader = VerifiedRegistryLoader(
        decision_path=decision,
        committed_lock_path=committed,
        download_root=tmp_path / "cache",
        api_factory=lambda: api,
        now=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    runtime = loader.load()
    assert runtime.predict_probability(request()) == pytest.approx(0.2)
    assert runtime.identity["serving_alias"] == "production"
    assert runtime.identity["internal_production_gate_passed"] is False
    assert runtime.identity["deployment_purpose"] == "academic_demo"
    assert "Academic demonstration" in runtime.identity["governance_notice"]
    assert len(runtime.route_reliability("DEN", "LAX", "UA")) == 2


def test_verified_loader_rejects_alias_drift(tmp_path: Path) -> None:
    decision, committed, _, artifact = release_fixture(tmp_path)
    artifact.digest = "different"
    loader = VerifiedRegistryLoader(
        decision_path=decision,
        committed_lock_path=committed,
        api_factory=lambda: FakeApi(artifact),
    )
    with pytest.raises(RegistryRuntimeError, match="drift"):
        loader.load()


def test_verified_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    decision, committed, root, artifact = release_fixture(tmp_path)
    (root / "model_bundle" / "MODEL_CARD.md").write_text("tampered")
    loader = VerifiedRegistryLoader(
        decision_path=decision,
        committed_lock_path=committed,
        api_factory=lambda: FakeApi(artifact),
    )
    with pytest.raises(RegistryRuntimeError, match="hash verification"):
        loader.load()


def test_runtime_rejects_nonfinite_model_output() -> None:
    runtime = ServingRuntime(
        model=ConstantModel(float("nan")),
        routes=pd.DataFrame(),
        threshold=0.2,
        feature_schema=[
            "Reporting_Airline",
            "Origin",
            "Dest",
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "CRSDepTime",
            "CRSArrTime",
            "CRSElapsedTime",
            "Distance",
            "scheduled_departure_hour",
            "scheduled_arrival_hour",
            "scheduled_departure_minute_bucket",
            "scheduled_arrival_minute_bucket",
            "is_weekend",
            "scheduled_departure_sin",
            "scheduled_departure_cos",
            "scheduled_arrival_sin",
            "scheduled_arrival_cos",
        ],
        metadata={},
        training_baseline={},
        development_metrics={},
        model_card="",
        release_policy="",
        identity={},
    )
    with pytest.raises(RegistryRuntimeError, match="invalid probability"):
        runtime.predict_probability(request())
