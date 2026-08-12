"""Strict validation for the immutable, non-secret deployment manifest."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

IMAGE_PATTERN = re.compile(r"^ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[0-9a-f]{64}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SECRET_VALUE_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(aws_secret_access_key|wandb_api_key)\s*[=:]\s*(?!__)[^\s,}\"]+"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
)

REQUIRED_ENVIRONMENT = {
    "api": {
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "AWS_REGION",
        "DYNAMODB_TABLE",
        "MODEL_DOWNLOAD_DIR",
        "PREDICTION_CACHE_MAXSIZE",
        "PREDICTION_CACHE_TTL_SECONDS",
    },
    "traveler": {
        "API_BASE_URL",
        "API_CONNECT_TIMEOUT_SECONDS",
        "API_READ_TIMEOUT_SECONDS",
    },
    "monitor": {
        "AWS_REGION",
        "DYNAMODB_TABLE",
        "MONITOR_DEFAULT_DAYS",
        "MONITOR_MAX_DAYS",
        "MONITOR_QUERY_CACHE_TTL_SECONDS",
    },
}


class DeploymentManifestError(ValueError):
    """The deployment manifest is incomplete, mutable, or inconsistent."""


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise DeploymentManifestError(f"{context} is missing {key}")
    return mapping[key]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentManifestError(f"unable to read {label}") from error
    if not isinstance(value, dict):
        raise DeploymentManifestError(f"{label} must be a JSON object")
    return value


def _git_commit_is_reachable(repository_root: Path, sha: str) -> bool:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        return False
    reachable = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return reachable.returncode == 0


def validate_deployment_manifest(
    manifest: dict[str, Any],
    *,
    release_decision: dict[str, Any],
    selection_lock: dict[str, Any],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Return the manifest after verifying all frozen identities and safe structure."""

    if manifest.get("schema_version") != 1:
        raise DeploymentManifestError("deployment manifest schema_version must be 1")

    project = _required(manifest, "project", "manifest")
    if project.get("name") != "us-flight-delay-mlops":
        raise DeploymentManifestError("unexpected project name")
    repository_url = project.get("public_repository_url", "")
    repository_pattern = r"https://github\.com/[A-Za-z0-9_.-]+/us-flight-delay-mlops"
    if not re.fullmatch(repository_pattern, repository_url):
        raise DeploymentManifestError("public repository URL is not canonical")
    if project.get("wandb_project_url") != (
        "https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops"
    ):
        raise DeploymentManifestError("W&B project URL is not canonical")

    deployment_sha = _required(manifest, "deployment_git_sha", "manifest")
    if not isinstance(deployment_sha, str) or not SHA_PATTERN.fullmatch(deployment_sha):
        raise DeploymentManifestError("deployment_git_sha must be a full Git SHA")
    if repository_root is not None and not _git_commit_is_reachable(
        repository_root, deployment_sha
    ):
        raise DeploymentManifestError("deployment_git_sha is not a reachable local commit")

    model = _required(manifest, "model", "manifest")
    identity_pairs = {
        "registry_collection": "registry_path",
        "serving_alias": "serving_alias",
        "registry_version": "registry_version",
        "registry_digest": "registry_digest",
        "release_bundle_digest": "bundle_digest",
    }
    for manifest_key, release_key in identity_pairs.items():
        if model.get(manifest_key) != release_decision.get(release_key):
            raise DeploymentManifestError(f"model {manifest_key} disagrees with release decision")
    if model.get("classification_threshold") != selection_lock.get("threshold"):
        raise DeploymentManifestError("classification threshold disagrees with selection lock")
    if model.get("release_bundle_digest") != selection_lock.get("aggregate_bundle_digest"):
        raise DeploymentManifestError("bundle digest disagrees with selection lock")
    if model.get("serving_alias") != "staging":
        raise DeploymentManifestError("Brief 08 must preserve the staging alias")

    images = _required(manifest, "images", "manifest")
    if set(images) != {"api", "traveler", "monitor"}:
        raise DeploymentManifestError("images must contain exactly api, traveler, and monitor")
    references = []
    for component, image in images.items():
        reference = image.get("reference", "")
        if not IMAGE_PATTERN.fullmatch(reference):
            raise DeploymentManifestError(f"{component} image is not an immutable GHCR digest")
        if image.get("source_git_sha") != deployment_sha:
            raise DeploymentManifestError(f"{component} image source SHA is not deployment SHA")
        references.append(reference)
    if len(references) != len(set(references)):
        raise DeploymentManifestError("component image references must be distinct")

    runtime = _required(manifest, "runtime", "manifest")
    if runtime.get("python") != "3.11":
        raise DeploymentManifestError("expected Python runtime must be 3.11")
    ports = runtime.get("ports", {})
    if ports != {"api": 8000, "traveler": 8501, "monitor": 8501}:
        raise DeploymentManifestError("component ports disagree with frozen topology")
    hosts = runtime.get("ec2_logical_names", {})
    if hosts != {
        "api": "flight-api",
        "traveler": "flight-user-ui",
        "monitor": "flight-monitor",
    }:
        raise DeploymentManifestError("EC2 logical names disagree with frozen topology")

    database = _required(manifest, "dynamodb", "manifest")
    if database != {
        "table_name": "flight-delay-events",
        "gsi_name": "event-date-created-at-index",
        "billing_mode": "PAY_PER_REQUEST",
    }:
        raise DeploymentManifestError("DynamoDB identity disagrees with the accepted schema")

    environment = _required(manifest, "environment_variable_names", "manifest")
    for component, required_names in REQUIRED_ENVIRONMENT.items():
        actual = set(environment.get(component, []))
        if actual != required_names:
            raise DeploymentManifestError(f"{component} environment-name set is incomplete")
    forbidden_names = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
    if any(forbidden_names & set(names) for names in environment.values()):
        raise DeploymentManifestError(
            "temporary AWS credential names must not enter host env files"
        )

    smoke = _required(manifest, "smoke_test", "manifest")
    if smoke.get("schema_version") != 1 or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", str(smoke.get("reviewed_date", ""))
    ):
        raise DeploymentManifestError("smoke-test metadata is invalid")

    encoded = json.dumps(manifest, sort_keys=True)
    if any(pattern.search(encoded) for pattern in SECRET_VALUE_PATTERNS):
        raise DeploymentManifestError("deployment manifest contains a credential-like value")
    return manifest


def load_and_validate_manifest(
    manifest_path: Path,
    *,
    release_decision_path: Path,
    selection_lock_path: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Load the manifest and committed release evidence, then validate them."""

    return validate_deployment_manifest(
        _load_json(manifest_path, "deployment manifest"),
        release_decision=_load_json(release_decision_path, "release decision"),
        selection_lock=_load_json(selection_lock_path, "selection lock"),
        repository_root=repository_root,
    )
