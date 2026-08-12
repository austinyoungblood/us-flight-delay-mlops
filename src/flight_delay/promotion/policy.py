"""Versioned promotion policy loading and semantic validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class PolicyError(ValueError):
    """The committed promotion policy is incomplete or unsafe."""


@dataclass(frozen=True)
class RankingRule:
    field: str
    direction: str


@dataclass(frozen=True)
class PromotionPolicy:
    schema_version: str
    policy_version: str
    policy_sha256: str
    purpose: str
    registry_collection: str
    source_project: str
    artifact_type: str
    target_alias: str
    allowed_target_aliases: tuple[str, ...]
    dry_run_default: bool
    required_metadata: tuple[str, ...]
    dataset_artifact: str
    allowed_dataset_digests: tuple[str, ...]
    feature_schema_sha256: str
    allowed_evaluation_protocols: tuple[str, ...]
    required_boolean_gates: dict[str, bool]
    max_bundle_size_bytes: int
    ranking: tuple[RankingRule, ...]
    prefer_incumbent_on_metric_tie: bool
    tie_break_field: str
    forbidden_key_fragments: tuple[str, ...]
    forbidden_key_prefixes: tuple[str, ...]

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(rule.field for rule in self.ranking if rule.field != "bundle_size_bytes")

    def validate_target_alias(self, alias: str) -> None:
        if alias not in self.allowed_target_aliases:
            raise PolicyError(f"target alias is not allowed by policy: {alias}")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be a mapping")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise PolicyError(f"{label} must be a non-empty string list")
    return tuple(value)


def load_policy(path: Path) -> PromotionPolicy:
    """Load and fail-closed validate the committed promotion policy."""

    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    root = _mapping(payload, "promotion policy")
    if root.get("schema_version") != "brief09b-v1":
        raise PolicyError("unexpected promotion policy schema version")
    for name in ("policy_version", "purpose", "target_alias"):
        if not isinstance(root.get(name), str) or not root[name]:
            raise PolicyError(f"promotion policy {name} must be a non-empty string")
    if type(root.get("dry_run_default")) is not bool:
        raise PolicyError("promotion policy dry_run_default must be boolean")

    scope = _mapping(root.get("scope"), "scope")
    lineage = _mapping(root.get("lineage"), "lineage")
    gates = _mapping(root.get("mandatory_gates"), "mandatory_gates")
    tie = _mapping(root.get("tie_breaking"), "tie_breaking")
    inputs = _mapping(root.get("selection_inputs"), "selection_inputs")
    if inputs.get("development_validation_only") is not True:
        raise PolicyError("promotion selection must be development/validation only")

    ranking_payload = root.get("ranking")
    if not isinstance(ranking_payload, list) or not ranking_payload:
        raise PolicyError("promotion policy ranking must be non-empty")
    ranking: list[RankingRule] = []
    for item in ranking_payload:
        rule = _mapping(item, "ranking rule")
        field = rule.get("field")
        direction = rule.get("direction")
        if not isinstance(field, str) or not field:
            raise PolicyError("ranking field must be a non-empty string")
        if direction not in {"maximize", "minimize"}:
            raise PolicyError(f"ranking direction is invalid for {field}")
        lowered = field.lower()
        if "final_test" in lowered or lowered.startswith("test_"):
            raise PolicyError("final-test metrics are prohibited from ranking")
        ranking.append(RankingRule(field=field, direction=direction))

    boolean_gate_names = {
        "release_eligible",
        "lineage_verified",
        "serialization_integrity",
        "inference_compatible",
    }
    if set(gates) != {*boolean_gate_names, "max_bundle_size_bytes"}:
        raise PolicyError("mandatory gate set is unexpected")
    if any(type(gates[name]) is not bool for name in boolean_gate_names):
        raise PolicyError("mandatory candidate gates must be boolean")
    max_size = gates["max_bundle_size_bytes"]
    if not isinstance(max_size, int) or max_size <= 0:
        raise PolicyError("max_bundle_size_bytes must be positive")

    policy = PromotionPolicy(
        schema_version=root["schema_version"],
        policy_version=root["policy_version"],
        policy_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        purpose=root["purpose"],
        registry_collection=str(scope.get("registry_collection", "")),
        source_project=str(scope.get("source_project", "")),
        artifact_type=str(scope.get("artifact_type", "")),
        target_alias=root["target_alias"],
        allowed_target_aliases=_strings(root.get("allowed_target_aliases"), "allowed aliases"),
        dry_run_default=root["dry_run_default"],
        required_metadata=_strings(root.get("required_metadata"), "required metadata"),
        dataset_artifact=str(lineage.get("dataset_artifact", "")),
        allowed_dataset_digests=_strings(
            lineage.get("allowed_dataset_digests"), "allowed dataset digests"
        ),
        feature_schema_sha256=str(lineage.get("feature_schema_sha256", "")),
        allowed_evaluation_protocols=_strings(
            lineage.get("allowed_evaluation_protocols"), "evaluation protocols"
        ),
        required_boolean_gates={name: gates[name] for name in sorted(boolean_gate_names)},
        max_bundle_size_bytes=max_size,
        ranking=tuple(ranking),
        prefer_incumbent_on_metric_tie=bool(tie.get("prefer_incumbent_on_metric_tie")),
        tie_break_field=str(tie.get("final_field", "")),
        forbidden_key_fragments=_strings(
            inputs.get("forbidden_key_fragments"), "forbidden key fragments"
        ),
        forbidden_key_prefixes=_strings(
            inputs.get("forbidden_key_prefixes"), "forbidden key prefixes"
        ),
    )
    if not all(
        (
            policy.registry_collection,
            policy.source_project,
            policy.artifact_type,
            policy.dataset_artifact,
            policy.feature_schema_sha256,
        )
    ):
        raise PolicyError("promotion policy scope or lineage is incomplete")
    policy.validate_target_alias(policy.target_alias)
    if policy.tie_break_field != "candidate_id":
        raise PolicyError("candidate_id must be the final deterministic tie-break field")
    return policy
