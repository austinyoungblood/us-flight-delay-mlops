"""Canonical, self-validating JSON manifests for dataset provenance."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flight_delay.data.preprocessing import DataQualityError

MANIFEST_DIGEST_FIELD = "manifest_digest"


class ManifestError(DataQualityError):
    """Raised when a stable manifest cannot be created or verified."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a JSON mapping deterministically as compact UTF-8 bytes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_digest(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a manifest excluding its digest field."""

    stable_payload = dict(payload)
    stable_payload.pop(MANIFEST_DIGEST_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(stable_payload)).hexdigest()


def with_manifest_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy containing its canonical digest."""

    result = dict(payload)
    result[MANIFEST_DIGEST_FIELD] = manifest_digest(result)
    return result


def validate_manifest(payload: Mapping[str, Any]) -> str:
    """Validate and return a manifest's declared canonical digest."""

    declared = payload.get(MANIFEST_DIGEST_FIELD)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ManifestError("manifest_digest must be a 64-character SHA-256 hex string")
    computed = manifest_digest(payload)
    if declared != computed:
        raise ManifestError(
            f"manifest digest mismatch: declared {declared}, computed {computed}"
        )
    return declared


def write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a canonical manifest with a trailing newline."""

    resolved = with_manifest_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    try:
        temporary.write_bytes(canonical_json_bytes(resolved) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest, require an object payload, and verify its digest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"manifest {path} must contain a JSON object")
    validate_manifest(payload)
    return payload
