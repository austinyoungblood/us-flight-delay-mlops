"""Stable, secret-free candidate records for deterministic model selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class CandidateMetadataError(ValueError):
    """Candidate metadata is incomplete, non-finite, or contains forbidden test fields."""


def _forbidden_key(path: str, fragments: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    normalized = path.lower()
    leaf = normalized.rsplit(".", maxsplit=1)[-1]
    return any(value in normalized for value in fragments) or any(
        leaf.startswith(value) for value in prefixes
    )


def _scan_keys(
    value: object,
    *,
    fragments: tuple[str, ...],
    prefixes: tuple[str, ...],
    path: str = "candidate",
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_path = f"{path}.{key}"
            if _forbidden_key(key_path, fragments, prefixes):
                raise CandidateMetadataError(f"selection input contains forbidden key: {key}")
            _scan_keys(nested, fragments=fragments, prefixes=prefixes, path=key_path)
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _scan_keys(
                nested,
                fragments=fragments,
                prefixes=prefixes,
                path=f"{path}[{index}]",
            )


@dataclass(frozen=True)
class CandidateRecord:
    """Normalized immutable candidate metadata; no final-test fields are representable."""

    candidate_id: str
    registry_path: str
    registry_version: str
    registry_digest: str
    source_artifact_name: str
    source_artifact_version: str
    source_artifact_digest: str
    git_sha: str
    dataset_artifact: str
    dataset_digest: str
    feature_schema_sha256: str
    evaluation_protocol: str
    development_metrics: dict[str, float]
    bundle_size_bytes: int
    release_eligible: bool
    lineage_verified: bool
    serialization_integrity: bool
    inference_compatible: bool
    aliases: tuple[str, ...] = ()

    @property
    def immutable_identity(self) -> str:
        return f"{self.registry_path}:{self.registry_version}@{self.registry_digest}"

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        forbidden_key_fragments: tuple[str, ...] = ("final_test", "sealed_test"),
        forbidden_key_prefixes: tuple[str, ...] = ("test_",),
    ) -> CandidateRecord:
        """Validate a raw record and reject unknown or final-test selection inputs."""

        if not isinstance(value, dict):
            raise CandidateMetadataError("candidate must be a mapping")
        _scan_keys(
            value,
            fragments=forbidden_key_fragments,
            prefixes=forbidden_key_prefixes,
        )
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        missing = allowed - {"aliases"} - set(value)
        if unknown:
            raise CandidateMetadataError(f"candidate contains unknown fields: {sorted(unknown)}")
        if missing:
            raise CandidateMetadataError(f"candidate is missing fields: {sorted(missing)}")

        strings = allowed - {
            "aliases",
            "development_metrics",
            "bundle_size_bytes",
            "release_eligible",
            "lineage_verified",
            "serialization_integrity",
            "inference_compatible",
        }
        if any(not isinstance(value[name], str) or not value[name] for name in strings):
            raise CandidateMetadataError("candidate identity fields must be non-empty strings")
        if not GIT_SHA_PATTERN.fullmatch(value["git_sha"]):
            raise CandidateMetadataError("candidate git_sha must be a full commit SHA")
        if not isinstance(value["bundle_size_bytes"], int) or value["bundle_size_bytes"] <= 0:
            raise CandidateMetadataError("candidate bundle_size_bytes must be positive")
        booleans = (
            "release_eligible",
            "lineage_verified",
            "serialization_integrity",
            "inference_compatible",
        )
        if any(type(value[name]) is not bool for name in booleans):
            raise CandidateMetadataError("candidate gate fields must be boolean")
        metrics = value["development_metrics"]
        if not isinstance(metrics, dict) or not metrics:
            raise CandidateMetadataError(
                "candidate development_metrics must be a non-empty mapping"
            )
        normalized_metrics: dict[str, float] = {}
        for name, metric in metrics.items():
            if not isinstance(name, str) or not name:
                raise CandidateMetadataError("candidate metric names must be non-empty strings")
            if isinstance(metric, bool) or not isinstance(metric, int | float):
                raise CandidateMetadataError(f"candidate metric {name} must be numeric")
            numeric = float(metric)
            if not math.isfinite(numeric):
                raise CandidateMetadataError(f"candidate metric {name} must be finite")
            normalized_metrics[name] = numeric
        aliases = value.get("aliases", ())
        if not isinstance(aliases, list | tuple) or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise CandidateMetadataError("candidate aliases must be strings")
        return cls(
            **{
                **{name: value[name] for name in strings},
                **{name: value[name] for name in booleans},
                "development_metrics": normalized_metrics,
                "bundle_size_bytes": value["bundle_size_bytes"],
                "aliases": tuple(sorted(set(aliases))),
            }
        )

    def audit_view(self, metric_names: tuple[str, ...]) -> dict[str, Any]:
        """Return only selection-relevant, non-secret evidence."""

        return {
            "candidate_id": self.candidate_id,
            "immutable_identity": self.immutable_identity,
            "registry_version": self.registry_version,
            "registry_digest": self.registry_digest,
            "source_artifact_name": self.source_artifact_name,
            "source_artifact_version": self.source_artifact_version,
            "source_artifact_digest": self.source_artifact_digest,
            "git_sha": self.git_sha,
            "dataset_artifact": self.dataset_artifact,
            "dataset_digest": self.dataset_digest,
            "feature_schema_sha256": self.feature_schema_sha256,
            "evaluation_protocol": self.evaluation_protocol,
            "development_metrics": {
                name: self.development_metrics[name]
                for name in metric_names
                if name in self.development_metrics
            },
            "bundle_size_bytes": self.bundle_size_bytes,
            "release_eligible": self.release_eligible,
            "lineage_verified": self.lineage_verified,
            "serialization_integrity": self.serialization_integrity,
            "inference_compatible": self.inference_compatible,
            "aliases": list(self.aliases),
        }
