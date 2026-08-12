"""Sanitized, machine-readable promotion decision records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flight_delay.data.manifest import canonical_json_bytes
from flight_delay.promotion.policy import PromotionPolicy
from flight_delay.promotion.selector import SelectionResult
from flight_delay.promotion.wandb_registry import AliasState


def _alias_view(state: AliasState | None) -> dict[str, str] | None:
    if state is None:
        return None
    return {
        "registry_path": state.registry_path,
        "alias": state.alias,
        "version": state.version,
        "digest": state.digest,
        "source_digest": state.source_digest,
        "immutable_identity": state.immutable_identity,
    }


def build_audit_record(
    *,
    mode: str,
    policy: PromotionPolicy,
    git_sha: str,
    target_alias: str,
    selection: SelectionResult,
    before: AliasState | None,
    after: AliasState | None,
    actual_action: str,
    wandb_verified: bool,
    workflow_identity: dict[str, str] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a complete decision without credentials or final-test metrics."""

    timestamp = timestamp or datetime.now(UTC)
    return {
        "schema_version": 1,
        "timestamp": timestamp.isoformat(),
        "mode": mode,
        "policy": {
            "schema_version": policy.schema_version,
            "version": policy.policy_version,
            "sha256": policy.policy_sha256,
            "purpose": policy.purpose,
        },
        "git_sha": git_sha,
        "target_collection": policy.registry_collection,
        "target_alias": target_alias,
        "incumbent_identity": selection.incumbent_identity,
        "candidates": [
            candidate.audit_view(policy.metric_names) for candidate in selection.eligible
        ],
        "rejected_candidates": [
            {
                "candidate_id": item.candidate_id,
                "category": item.category,
                "reasons": list(item.reasons),
            }
            for item in selection.rejected
        ],
        "selected_winner": (
            selection.winner.audit_view(policy.metric_names)
            if selection.winner is not None
            else None
        ),
        "ranking_explanation": list(selection.ranking_explanation),
        "outcome": selection.outcome,
        "requested_action": selection.requested_action,
        "actual_action": actual_action,
        "before_alias": _alias_view(before),
        "after_alias": _alias_view(after),
        "wandb_verification": {"verified": wandb_verified},
        "workflow": workflow_identity or {},
        "selection_data_boundary": "development_validation_only",
    }


def write_audit_record(path: Path, record: dict[str, Any]) -> None:
    """Write canonical JSON and reject obvious credential fields before persistence."""

    encoded = json.dumps(record, sort_keys=True).lower()
    forbidden = ("wandb_api_key", "aws_secret_access_key", "aws_session_token", "private_key")
    if any(name in encoded for name in forbidden):
        raise ValueError("promotion audit contains a credential-like field")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(record) + b"\n")
