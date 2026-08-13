"""Validation for the rubric-mapped final evidence manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ALLOWED_SOURCES = {"github", "wandb", "aws-console", "application"}
ALLOWED_STATUS = {"pending-live-session", "captured", "missing", "not-applicable"}
ALLOWED_PUBLICATION_STATUS = {"ready", "redaction-required"}
FILENAME_PATTERN = re.compile(r"^[0-9]{2}[a-z]?_[a-z0-9][a-z0-9_-]*\.(png|json|txt|md)$")
LIVE_SESSION_LOCATOR_PATTERN = re.compile(r"^live-session://[a-z0-9][a-z0-9/_-]*$")
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


def _capture_filenames(capture: dict[str, Any], criterion: str) -> list[str]:
    has_filename = "filename" in capture
    has_filenames = "filenames" in capture
    if has_filename == has_filenames:
        raise EvidenceValidationError(
            f"evidence must declare exactly one of filename or filenames: {criterion}"
        )
    values = capture.get("filenames") if has_filenames else [capture.get("filename")]
    if not isinstance(values, list) or not values:
        raise EvidenceValidationError(f"evidence filenames must be a non-empty list: {criterion}")
    filenames = [str(value) for value in values]
    if any(not FILENAME_PATTERN.fullmatch(filename) for filename in filenames):
        raise EvidenceValidationError(f"unsafe evidence filename: {filenames}")
    if len(filenames) != len(set(filenames)):
        raise EvidenceValidationError(f"duplicate filename within criterion: {criterion}")
    return filenames


def _artifact_roots(manifest: dict[str, Any], evidence_root: Path | None) -> list[Path]:
    directories = manifest.get("artifact_directories")
    if directories is None:
        return [evidence_root] if evidence_root is not None else []
    if not isinstance(directories, list) or not directories:
        raise EvidenceValidationError("artifact_directories must be a non-empty list")
    roots: list[Path] = []
    for value in directories:
        if not isinstance(value, str):
            raise EvidenceValidationError("artifact directory must be a repository-relative string")
        directory = Path(value)
        if (
            directory.is_absolute()
            or not directory.parts
            or any(part in {".", ".."} for part in directory.parts)
        ):
            raise EvidenceValidationError(f"unsafe artifact directory: {value}")
        if evidence_root is not None:
            roots.append(evidence_root.parent / directory)
    return roots


def validate_evidence_manifest(
    manifest: dict[str, Any], *, evidence_root: Path | None = None, require_files: bool = False
) -> dict[str, Any]:
    """Validate mappings and optionally require every captured file to exist."""

    if manifest.get("schema_version") != 1:
        raise EvidenceValidationError("evidence schema_version must be 1")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise EvidenceValidationError("captures must be a non-empty list")
    artifact_roots = _artifact_roots(manifest, evidence_root)
    seen_criteria: set[str] = set()
    seen_files: set[str] = set()
    for capture in captures:
        if not isinstance(capture, dict):
            raise EvidenceValidationError("each capture must be an object")
        criterion = str(capture.get("criterion", ""))
        filenames = _capture_filenames(capture, criterion)
        source = capture.get("source")
        status = capture.get("status")
        if criterion in seen_criteria:
            raise EvidenceValidationError(f"duplicate criterion: {criterion}")
        duplicate_files = seen_files.intersection(filenames)
        if duplicate_files:
            raise EvidenceValidationError(f"duplicate filename: {sorted(duplicate_files)[0]}")
        if source not in ALLOWED_SOURCES:
            raise EvidenceValidationError(f"unsupported evidence source: {source}")
        if status not in ALLOWED_STATUS:
            raise EvidenceValidationError(f"unsupported evidence status: {status}")
        if not str(capture.get("capture_instruction", "")).strip():
            raise EvidenceValidationError(f"missing capture instruction: {criterion}")
        source_url = str(capture.get("source_url", ""))
        if status == "captured" and not (
            source_url.startswith(("https://", "http://127.0.0.1:"))
            or LIVE_SESSION_LOCATOR_PATTERN.fullmatch(source_url)
        ):
            raise EvidenceValidationError(f"captured evidence lacks a safe source URL: {criterion}")
        publication_status = capture.get("publication_status", "ready")
        if publication_status not in ALLOWED_PUBLICATION_STATUS:
            raise EvidenceValidationError(f"unsupported publication status: {publication_status}")
        if (
            publication_status == "redaction-required"
            and not str(capture.get("redaction_notes", "")).strip()
        ):
            raise EvidenceValidationError(f"missing redaction notes: {criterion}")
        if require_files and status == "captured":
            for filename in filenames:
                if not artifact_roots or not any(
                    (root / filename).is_file() for root in artifact_roots
                ):
                    raise EvidenceValidationError(f"captured evidence file is missing: {filename}")
        seen_criteria.add(criterion)
        seen_files.update(filenames)
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
