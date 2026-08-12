"""Validation for the rubric-mapped final evidence manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ALLOWED_SOURCES = {"github", "wandb", "aws-console", "application"}
ALLOWED_STATUS = {"pending-live-session", "captured", "not-applicable"}
FILENAME_PATTERN = re.compile(r"^[0-9]{2}_[a-z0-9][a-z0-9_-]*\.(png|json|txt|md)$")
REQUIRED_CRITERIA = {
    "github.repository",
    "github.pr_ci",
    "wandb.project",
    "wandb.experiments",
    "wandb.dataset",
    "wandb.registry",
    "aws.ec2_instances",
    "aws.security_groups",
    "aws.instance_profile",
    "aws.dynamodb_schema",
    "aws.dynamodb_prediction",
    "aws.cloudwatch_or_status",
    "app.api_docs",
    "app.health",
    "app.model_info",
    "app.traveler_prediction",
    "app.traveler_feedback",
    "app.monitor_operations",
    "app.monitor_drift",
    "app.monitor_feedback",
}


class EvidenceValidationError(ValueError):
    """The evidence manifest is incomplete or unsafe."""


def validate_evidence_manifest(
    manifest: dict[str, Any], *, evidence_root: Path | None = None, require_files: bool = False
) -> dict[str, Any]:
    """Validate mappings and optionally require every captured file to exist."""

    if manifest.get("schema_version") != 1:
        raise EvidenceValidationError("evidence schema_version must be 1")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise EvidenceValidationError("captures must be a non-empty list")
    seen_criteria: set[str] = set()
    seen_files: set[str] = set()
    for capture in captures:
        if not isinstance(capture, dict):
            raise EvidenceValidationError("each capture must be an object")
        criterion = str(capture.get("criterion", ""))
        filename = str(capture.get("filename", ""))
        source = capture.get("source")
        status = capture.get("status")
        if criterion in seen_criteria:
            raise EvidenceValidationError(f"duplicate criterion: {criterion}")
        if filename in seen_files:
            raise EvidenceValidationError(f"duplicate filename: {filename}")
        if source not in ALLOWED_SOURCES:
            raise EvidenceValidationError(f"unsupported evidence source: {source}")
        if status not in ALLOWED_STATUS:
            raise EvidenceValidationError(f"unsupported evidence status: {status}")
        if not FILENAME_PATTERN.fullmatch(filename):
            raise EvidenceValidationError(f"unsafe evidence filename: {filename}")
        if not str(capture.get("capture_instruction", "")).strip():
            raise EvidenceValidationError(f"missing capture instruction: {criterion}")
        if status == "captured" and not str(capture.get("source_url", "")).startswith(
            ("https://", "http://127.0.0.1:")
        ):
            raise EvidenceValidationError(f"captured evidence lacks a safe source URL: {criterion}")
        if (
            require_files
            and status == "captured"
            and (evidence_root is None or not (evidence_root / filename).is_file())
        ):
            raise EvidenceValidationError(f"captured evidence file is missing: {filename}")
        seen_criteria.add(criterion)
        seen_files.add(filename)
    missing = REQUIRED_CRITERIA - seen_criteria
    if missing:
        raise EvidenceValidationError(f"required evidence criteria are missing: {sorted(missing)}")
    return manifest


def load_evidence_manifest(path: Path, *, require_files: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError("unable to read evidence manifest") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError("evidence manifest must be a JSON object")
    return validate_evidence_manifest(
        value,
        evidence_root=path.parent,
        require_files=require_files,
    )
