from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from flight_delay.deployment import (
    DeploymentManifestError,
    EvidenceValidationError,
    SmokeError,
    SmokeRunner,
    load_and_validate_manifest,
    validate_deployment_manifest,
    validate_evidence_manifest,
)
from flight_delay.deployment.evidence import (
    KNOWN_CRITERIA,
    REQUIRED_CRITERIA,
    load_evidence_manifest,
    missing_required_evidence,
)

ROOT = Path(__file__).resolve().parents[2]


def deployment_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = {
        "registry_path": "wandb-registry-Model/us-flight-arrival-delay-15m",
        "serving_alias": "production",
        "registry_version": "v0",
        "registry_digest": "registry-digest",
        "bundle_digest": "bundle-digest",
        "internal_production_gate_passed": False,
        "deployment_purpose": "academic_demo",
    }
    lock = {"threshold": 0.184, "aggregate_bundle_digest": "bundle-digest"}
    sha = "a" * 40
    manifest = {
        "schema_version": 1,
        "project": {
            "name": "us-flight-delay-mlops",
            "public_repository_url": "https://github.com/example/us-flight-delay-mlops",
            "wandb_project_url": (
                "https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops"
            ),
        },
        "deployment_git_sha": sha,
        "model": {
            "registry_collection": release["registry_path"],
            "serving_alias": "production",
            "registry_version": "v0",
            "registry_digest": "registry-digest",
            "release_bundle_digest": "bundle-digest",
            "internal_production_gate_passed": False,
            "deployment_purpose": "academic_demo",
            "classification_threshold": 0.184,
        },
        "images": {
            name: {
                "reference": f"ghcr.io/example/flight-{name}@sha256:{digit * 64}",
                "source_git_sha": sha,
            }
            for name, digit in (("api", "1"), ("traveler", "2"), ("monitor", "3"))
        },
        "runtime": {
            "python": "3.11",
            "ports": {"api": 8000, "traveler": 8501, "monitor": 8501},
            "ec2_logical_names": {
                "api": "flight-api",
                "traveler": "flight-user-ui",
                "monitor": "flight-monitor",
            },
        },
        "dynamodb": {
            "table_name": "flight-delay-events",
            "gsi_name": "event-date-created-at-index",
            "billing_mode": "PAY_PER_REQUEST",
        },
        "environment_variable_names": {
            "api": sorted(
                {
                    "WANDB_API_KEY",
                    "WANDB_ENTITY",
                    "WANDB_PROJECT",
                    "AWS_REGION",
                    "DYNAMODB_TABLE",
                    "MODEL_DOWNLOAD_DIR",
                    "PREDICTION_CACHE_MAXSIZE",
                    "PREDICTION_CACHE_TTL_SECONDS",
                }
            ),
            "traveler": sorted(
                {"API_BASE_URL", "API_CONNECT_TIMEOUT_SECONDS", "API_READ_TIMEOUT_SECONDS"}
            ),
            "monitor": sorted(
                {
                    "AWS_REGION",
                    "DYNAMODB_TABLE",
                    "MONITOR_DEFAULT_DAYS",
                    "MONITOR_MAX_DAYS",
                    "MONITOR_QUERY_CACHE_TTL_SECONDS",
                }
            ),
        },
        "smoke_test": {"schema_version": 1, "reviewed_date": "2026-08-10"},
    }
    return manifest, release, lock


def evidence_inputs() -> dict[str, Any]:
    return json.loads((ROOT / "evidence/evidence_manifest.json").read_text(encoding="utf-8"))


def test_deployment_manifest_accepts_only_frozen_identifiers() -> None:
    manifest, release, lock = deployment_inputs()
    assert (
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)
        == manifest
    )

    for mutation, message in (
        (("images", "api", "reference", "ghcr.io/example/api:latest"), "immutable"),
        (("model", "serving_alias", None, "staging"), "release decision"),
        (("deployment_git_sha", None, None, "short"), "full Git SHA"),
    ):
        invalid = copy.deepcopy(manifest)
        first, second, third, value = mutation
        if second is None:
            invalid[first] = value
        elif third is None:
            invalid[first][second] = value
        else:
            invalid[first][second][third] = value
        with pytest.raises(DeploymentManifestError, match=message):
            validate_deployment_manifest(invalid, release_decision=release, selection_lock=lock)


def test_deployment_manifest_rejects_credentials_and_bad_environment() -> None:
    manifest, release, lock = deployment_inputs()
    manifest["environment_variable_names"]["api"].append("AWS_ACCESS_KEY_ID")
    with pytest.raises(DeploymentManifestError, match="environment-name set"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("project", "name"), "other", "project name"),
        (("project", "public_repository_url"), "https://example.com/repo", "repository URL"),
        (("project", "wandb_project_url"), "https://wandb.ai/other", "W&B project URL"),
        (("deployment_git_sha",), "short", "full Git SHA"),
        (("model", "classification_threshold"), 0.5, "classification threshold"),
        (("model", "release_bundle_digest"), "other", "release decision"),
        (("images", "api", "reference"), "flight-api:latest", "immutable GHCR"),
        (("images", "api", "source_git_sha"), "b" * 40, "source SHA"),
        (("runtime", "python"), "3.12", "Python runtime"),
        (("runtime", "ports", "api"), 8080, "ports"),
        (("runtime", "ec2_logical_names", "api"), "api", "logical names"),
        (("dynamodb", "billing_mode"), "PROVISIONED", "DynamoDB identity"),
        (("smoke_test", "schema_version"), 2, "smoke-test metadata"),
        (("smoke_test", "reviewed_date"), "August 10", "smoke-test metadata"),
    ],
)
def test_deployment_manifest_rejects_structural_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    manifest, release, lock = deployment_inputs()
    target: dict[str, Any] = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DeploymentManifestError, match=message):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)


def test_deployment_manifest_rejects_missing_duplicate_and_secret_values() -> None:
    manifest, release, lock = deployment_inputs()
    manifest.pop("project")
    with pytest.raises(DeploymentManifestError, match="missing project"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["images"].pop("monitor")
    with pytest.raises(DeploymentManifestError, match="exactly api"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["images"]["monitor"]["reference"] = manifest["images"]["api"]["reference"]
    with pytest.raises(DeploymentManifestError, match="must be distinct"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["environment_variable_names"]["unexpected"] = ["AWS_SESSION_TOKEN"]
    with pytest.raises(DeploymentManifestError, match="credential names"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["notes"] = "WANDB_API_KEY=exposed-value"
    with pytest.raises(DeploymentManifestError, match="credential-like"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)


def test_deployment_manifest_preserves_governance_even_if_release_is_changed() -> None:
    manifest, release, lock = deployment_inputs()
    manifest["model"]["serving_alias"] = release["serving_alias"] = "staging"
    with pytest.raises(DeploymentManifestError, match="production alias"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["model"]["internal_production_gate_passed"] = release[
        "internal_production_gate_passed"
    ] = True
    with pytest.raises(DeploymentManifestError, match="failed internal production gate"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)

    manifest, release, lock = deployment_inputs()
    manifest["model"]["deployment_purpose"] = release["deployment_purpose"] = "operational"
    with pytest.raises(DeploymentManifestError, match="academic-demo"):
        validate_deployment_manifest(manifest, release_decision=release, selection_lock=lock)


def test_deployment_git_sha_must_be_reachable_when_repository_is_supplied() -> None:
    manifest, release, lock = deployment_inputs()
    with pytest.raises(DeploymentManifestError, match="reachable local commit"):
        validate_deployment_manifest(
            manifest,
            release_decision=release,
            selection_lock=lock,
            repository_root=ROOT,
        )

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest["deployment_git_sha"] = sha
    for image in manifest["images"].values():
        image["source_git_sha"] = sha
    assert (
        validate_deployment_manifest(
            manifest,
            release_decision=release,
            selection_lock=lock,
            repository_root=ROOT,
        )
        == manifest
    )


def test_evidence_manifest_requires_unique_complete_safe_mapping(tmp_path: Path) -> None:
    captures = [
        {
            "criterion": criterion,
            "filename": f"{index:02d}_{criterion.replace('.', '_')}.png",
            "evidence_mode": "supplemental" if criterion not in REQUIRED_CRITERIA else "screenshot",
            "required": criterion in REQUIRED_CRITERIA,
            "source": "application",
            "status": "missing" if criterion not in REQUIRED_CRITERIA else "pending-live-session",
            "verification": "Capture or inspect the frozen gate.",
        }
        for index, criterion in enumerate(sorted(KNOWN_CRITERIA), start=1)
    ]
    manifest = {"schema_version": 2, "captures": captures}
    assert validate_evidence_manifest(manifest, evidence_root=tmp_path) == manifest
    assert missing_required_evidence(manifest) == sorted(REQUIRED_CRITERIA)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_evidence_manifest(evidence_path) == manifest
    captures[1]["filename"] = captures[0]["filename"]
    with pytest.raises(EvidenceValidationError, match="duplicate filename"):
        validate_evidence_manifest(manifest)


def test_evidence_manifest_supports_curated_multi_file_evidence(tmp_path: Path) -> None:
    captures = [
        {
            "criterion": criterion,
            "filename": f"{index:02d}_{criterion.replace('.', '_')}.png",
            "evidence_mode": "supplemental" if criterion not in REQUIRED_CRITERIA else "screenshot",
            "required": criterion in REQUIRED_CRITERIA,
            "source": "application",
            "status": "missing",
            "verification": "Record the final evidence state.",
        }
        for index, criterion in enumerate(sorted(KNOWN_CRITERIA), start=1)
    ]
    first = captures[0]
    first.pop("filename")
    first.update(
        {
            "filenames": ["08a_aws_security_group_api.png", "08b_aws_security_group_ui.png"],
            "status": "captured",
            "source_url": "live-session://flight-api/security",
            "publication_status": "redaction-required",
            "redaction_notes": "Remove the operator IP before publication.",
        }
    )
    screenshot_root = tmp_path / "aws" / "screenshots"
    screenshot_root.mkdir(parents=True)
    for filename in first["filenames"]:
        (screenshot_root / filename).write_bytes(b"png")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    manifest = {
        "schema_version": 2,
        "artifact_directories": ["aws/screenshots"],
        "captures": captures,
    }
    assert (
        validate_evidence_manifest(manifest, evidence_root=evidence_root, require_files=True)
        == manifest
    )

    invalid = copy.deepcopy(manifest)
    invalid["captures"][0].pop("redaction_notes")
    with pytest.raises(EvidenceValidationError, match="missing redaction notes"):
        validate_evidence_manifest(invalid)


def test_final_evidence_manifest_has_complete_required_evidence_modes() -> None:
    manifest = evidence_inputs()
    assert validate_evidence_manifest(manifest) == manifest
    assert missing_required_evidence(manifest) == []
    assert {item["evidence_mode"] for item in manifest["captures"]} == {
        "public_url",
        "screenshot",
        "supplemental",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_mode", "archive", "unsupported evidence mode"),
        ("required", "yes", "required must be boolean"),
        ("required", False, "disagrees with rubric"),
        ("source", "application", "public URL evidence has unsupported source"),
        ("status", "captured", "unsupported public_url evidence status"),
        ("verification", " ", "missing verification"),
        ("source_url", "http://example.com", "lacks an HTTPS URL"),
    ],
)
def test_public_url_evidence_rejects_unsafe_metadata(
    field: str, value: object, message: str
) -> None:
    manifest = evidence_inputs()
    manifest["captures"][0][field] = value
    with pytest.raises(EvidenceValidationError, match=message):
        validate_evidence_manifest(manifest)


def test_public_url_evidence_cannot_claim_a_screenshot_file() -> None:
    manifest = evidence_inputs()
    manifest["captures"][0]["filename"] = "01_public.png"
    with pytest.raises(EvidenceValidationError, match="must not declare screenshot"):
        validate_evidence_manifest(manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_url": "file:///tmp/capture.png"}, "safe source URL"),
        ({"publication_status": "private"}, "publication status"),
        ({"filename": "../../secret.png"}, "unsafe evidence filename"),
    ],
)
def test_screenshot_evidence_rejects_unsafe_metadata(
    mutation: dict[str, object], message: str
) -> None:
    manifest = evidence_inputs()
    manifest["captures"][6].update(mutation)
    with pytest.raises(EvidenceValidationError, match=message):
        validate_evidence_manifest(manifest)


def test_evidence_filenames_must_be_unambiguous_and_unique() -> None:
    manifest = evidence_inputs()
    manifest["captures"][6]["filenames"] = ["07_duplicate.png"]
    with pytest.raises(EvidenceValidationError, match="exactly one"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][6].pop("filename")
    with pytest.raises(EvidenceValidationError, match="exactly one"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][6].pop("filename")
    manifest["captures"][6]["filenames"] = []
    with pytest.raises(EvidenceValidationError, match="non-empty list"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][7]["filenames"] = ["08a_duplicate.png", "08a_duplicate.png"]
    with pytest.raises(EvidenceValidationError, match="within criterion"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][12]["filename"] = manifest["captures"][6]["filename"]
    with pytest.raises(EvidenceValidationError, match="duplicate filename"):
        validate_evidence_manifest(manifest)


def test_redaction_and_required_file_failures_are_explicit(tmp_path: Path) -> None:
    manifest = evidence_inputs()
    manifest["captures"][6].pop("redaction_notes")
    with pytest.raises(EvidenceValidationError, match="missing redaction notes"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    with pytest.raises(EvidenceValidationError, match="captured evidence file is missing"):
        validate_evidence_manifest(manifest, evidence_root=tmp_path, require_files=True)


@pytest.mark.parametrize("directories", [[], "aws/screenshots", [42], ["../screenshots"]])
def test_artifact_directories_must_be_safe(directories: object) -> None:
    manifest = evidence_inputs()
    manifest["artifact_directories"] = directories
    with pytest.raises(EvidenceValidationError, match=r"artifact.director"):
        validate_evidence_manifest(manifest)


def test_evidence_top_level_and_criterion_integrity() -> None:
    with pytest.raises(EvidenceValidationError, match="schema_version"):
        validate_evidence_manifest({"schema_version": 1, "captures": []})
    with pytest.raises(EvidenceValidationError, match="non-empty list"):
        validate_evidence_manifest({"schema_version": 2, "captures": []})

    manifest = evidence_inputs()
    manifest["captures"][0] = "not-an-object"
    with pytest.raises(EvidenceValidationError, match="must be an object"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][1]["criterion"] = manifest["captures"][0]["criterion"]
    with pytest.raises(EvidenceValidationError, match="duplicate criterion"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"][0]["criterion"] = "unknown.criterion"
    with pytest.raises(EvidenceValidationError, match="unsupported evidence criterion"):
        validate_evidence_manifest(manifest)

    manifest = evidence_inputs()
    manifest["captures"].pop()
    with pytest.raises(EvidenceValidationError, match="criteria are missing"):
        validate_evidence_manifest(manifest)


def test_evidence_loader_rejects_invalid_json_and_non_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="unable to read"):
        load_evidence_manifest(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="must be a JSON object"):
        load_evidence_manifest(path)


def test_deployment_manifest_file_loader(tmp_path: Path) -> None:
    manifest, release, lock = deployment_inputs()
    paths = [tmp_path / name for name in ("manifest.json", "release.json", "lock.json")]
    for path, value in zip(paths, (manifest, release, lock), strict=True):
        path.write_text(json.dumps(value), encoding="utf-8")
    assert (
        load_and_validate_manifest(
            paths[0], release_decision_path=paths[1], selection_lock_path=paths[2]
        )
        == manifest
    )


def test_host_manifest_reader_uses_only_standard_library() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "deploy/read_manifest.py",
            "image",
            "api",
            "--manifest",
            "deploy/deployment_manifest.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
    )
    assert result.stdout.strip().startswith(
        "ghcr.io/austinyoungblood/us-flight-delay-mlops-api@sha256:"
    )


def test_evidence_cli_reports_required_evidence_complete() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_evidence_manifest.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(result.stdout)
    assert payload["required_evidence"] == {
        "complete": True,
        "missing": [],
        "present": 18,
        "total": 18,
    }


def model_info(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    return {
        "registry_path": model["registry_collection"],
        "serving_alias": model["serving_alias"],
        "registry_version": model["registry_version"],
        "registry_digest": model["registry_digest"],
        "source_artifact_digest": "source",
        "bundle_digest": model["release_bundle_digest"],
        "selection_lock_sha256": "lock",
        "route_asset_sha256": "route",
        "classification_threshold": model["classification_threshold"],
        "feature_schema": ["Origin"],
        "training_partitions": {"base_fit": "2025-01-01/2025-10-31"},
        "release_decision": {
            "serving_alias": "production",
            "internal_production_gate_passed": False,
            "deployment_purpose": "academic_demo",
        },
        "release_git_sha": "abc",
        "loaded_at": datetime(2026, 8, 10, tzinfo=UTC).isoformat(),
        "internal_production_gate_passed": False,
        "deployment_purpose": "academic_demo",
        "governance_notice": "Academic demonstration — internal production gate failed.",
        "serving_stage_notice": "Academic demonstration — internal production gate failed.",
    }


def prediction(
    identifier: str, *, cache_hit: bool, feedback: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "prediction_id": identifier,
        "delay_probability": 0.3,
        "predicted_delayed": True,
        "risk_band": "high",
        "classification_threshold": 0.184,
        "route_reliability": [],
        "support_warning": None,
        "model_alias": "production",
        "model_version": "v0",
        "model_digest": "registry-digest",
        "cache_hit": cache_hit,
        "latency_ms": 4.2,
        "created_at": "2026-08-10T12:00:00Z",
        "request": {
            "carrier": "UA",
            "origin": "DEN",
            "destination": "LAX",
            "flight_date": "2026-08-18",
            "scheduled_departure": "07:30:00",
            "scheduled_arrival": "09:00:00",
            "scheduled_elapsed_minutes": 150,
            "distance_miles": 862,
        },
        "event_date": "2026-08-10",
        "request_status": "success",
        "inference_latency_ms": 2.0,
        "persistence_latency_ms": 1.0,
        "total_latency_ms": 4.2,
        "bundle_digest": "bundle-digest",
        "feedback": feedback,
    }


def test_smoke_runner_proves_full_application_path_without_post_retries() -> None:
    manifest, _, _ = deployment_inputs()
    predict_count = 0
    post_paths: list[str] = []
    feedback_written = False
    feedback = {
        "actual_delayed": False,
        "arrival_delay_minutes": 4,
        "notes": "Deployment smoke test",
        "source": "deployment-smoke",
        "feedback_correct": False,
        "feedback_at": "2026-08-10T13:00:00Z",
        "feedback_revision": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal predict_count, feedback_written
        path = request.url.path
        if request.method == "POST":
            post_paths.append(path)
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "service": "flight-delay-api",
                    "status": "ready",
                    "model_loaded": True,
                    "database_connected": True,
                    "dependencies": {},
                },
            )
        if path == "/model-info":
            return httpx.Response(200, json=model_info(manifest))
        if path == "/predict":
            predict_count += 1
            value = prediction(f"id-{predict_count}", cache_hit=predict_count == 2)
            return httpx.Response(
                200,
                json={
                    key: item
                    for key, item in value.items()
                    if key
                    not in {
                        "request",
                        "event_date",
                        "request_status",
                        "inference_latency_ms",
                        "persistence_latency_ms",
                        "total_latency_ms",
                        "bundle_digest",
                        "feedback",
                    }
                },
            )
        if path == "/feedback/id-1":
            feedback_written = True
            return httpx.Response(200, json=feedback)
        if path == "/predictions/id-1":
            return httpx.Response(
                200,
                json=prediction(
                    "id-1", cache_hit=False, feedback=feedback if feedback_written else None
                ),
            )
        if path == "/predictions/id-2":
            return httpx.Response(200, json=prediction("id-2", cache_hit=True))
        if path == "/_stcore/health":
            return httpx.Response(200, text="ok")
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = SmokeRunner(client, manifest).run(
            api_base_url="http://api",
            traveler_base_url="http://traveler",
            monitor_base_url="http://monitor",
        )
    assert result["status"] == "passed"
    assert result["predictions"]["second_cache_hit"] is True
    assert post_paths == ["/predict", "/predict", "/feedback/id-1"]


def test_smoke_runner_fails_closed_on_model_mismatch() -> None:
    manifest, _, _ = deployment_inputs()
    invalid_info = model_info(manifest)
    invalid_info["registry_version"] = "v9"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "service": "flight-delay-api",
                    "status": "ready",
                    "model_loaded": True,
                    "database_connected": True,
                    "dependencies": {},
                },
            )
        return httpx.Response(200, json=invalid_info)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(SmokeError, match="registry_version"),
    ):
        SmokeRunner(client, manifest).run(
            api_base_url="http://api",
            traveler_base_url="http://traveler",
            monitor_base_url="http://monitor",
        )


def test_component_deploy_scripts_pass_validated_dry_run(tmp_path: Path) -> None:
    manifest, _, _ = deployment_inputs()
    release = json.loads((ROOT / "release/release_decision.json").read_text())
    lock = json.loads((ROOT / "release/selection_lock.json").read_text())
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest["deployment_git_sha"] = sha
    manifest["model"].update(
        {
            "registry_collection": release["registry_path"],
            "serving_alias": release["serving_alias"],
            "registry_version": release["registry_version"],
            "registry_digest": release["registry_digest"],
            "release_bundle_digest": release["bundle_digest"],
            "classification_threshold": lock["threshold"],
        }
    )
    for image in manifest["images"].values():
        image["source_git_sha"] = sha
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    values = {
        "api": {
            "WANDB_API_KEY": "test-only-value",
            "WANDB_ENTITY": "test-entity",
            "WANDB_PROJECT": "us-flight-delay-mlops",
            "AWS_REGION": "us-west-2",
            "DYNAMODB_TABLE": "flight-delay-events",
            "MODEL_DOWNLOAD_DIR": "/opt/us-flight-delay-mlops/model",
            "PREDICTION_CACHE_MAXSIZE": "1024",
            "PREDICTION_CACHE_TTL_SECONDS": "300",
        },
        "traveler": {
            "API_BASE_URL": "http://10.0.0.8:8000",
            "API_CONNECT_TIMEOUT_SECONDS": "3",
            "API_READ_TIMEOUT_SECONDS": "15",
        },
        "monitor": {
            "AWS_REGION": "us-west-2",
            "DYNAMODB_TABLE": "flight-delay-events",
            "MONITOR_DEFAULT_DAYS": "7",
            "MONITOR_MAX_DAYS": "31",
            "MONITOR_QUERY_CACHE_TTL_SECONDS": "30",
        },
    }
    test_environment = {
        **os.environ,
        "DEPLOY_DRY_RUN": "1",
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    for component, environment_values in values.items():
        env_path = tmp_path / f"{component}.env"
        env_path.write_text(
            "".join(f"{key}={value}\n" for key, value in environment_values.items()),
            encoding="utf-8",
        )
        env_path.chmod(0o600)
        result = subprocess.run(
            [
                str(ROOT / f"deploy/deploy_{component}.sh"),
                "--manifest",
                str(manifest_path),
                "--env-file",
                str(env_path),
            ],
            cwd=ROOT,
            env=test_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "dry-run validated" in result.stdout


def test_live_package_has_no_mutable_latest_reference_or_private_key() -> None:
    paths = [
        *sorted((ROOT / "deploy").rglob("*")),
        ROOT / "docs/aws-live-command-sheet.md",
        ROOT / "docs/aws-four-hour-runbook.md",
    ]
    content = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    assert ":latest" not in content
    assert "BEGIN PRIVATE KEY" not in content
