"""Pure deterministic policy evaluation and candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

from flight_delay.promotion.candidates import CandidateMetadataError, CandidateRecord
from flight_delay.promotion.policy import PromotionPolicy


@dataclass(frozen=True)
class Rejection:
    candidate_id: str
    category: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    outcome: str
    winner: CandidateRecord | None
    incumbent_identity: str | None
    eligible: tuple[CandidateRecord, ...]
    rejected: tuple[Rejection, ...]
    ranking_explanation: tuple[dict[str, Any], ...]
    requested_action: str


def _gate_reasons(candidate: CandidateRecord, policy: PromotionPolicy) -> list[str]:
    reasons: list[str] = []
    if candidate.registry_path != policy.registry_collection:
        reasons.append("registry collection is outside policy scope")
    if candidate.dataset_artifact != policy.dataset_artifact:
        reasons.append("dataset artifact is outside the allowed lineage")
    if candidate.dataset_digest not in policy.allowed_dataset_digests:
        reasons.append("dataset digest is not allowed")
    if candidate.feature_schema_sha256 != policy.feature_schema_sha256:
        reasons.append("feature schema is incompatible")
    if candidate.evaluation_protocol not in policy.allowed_evaluation_protocols:
        reasons.append("evaluation protocol is incompatible")
    for name, expected in policy.required_boolean_gates.items():
        if getattr(candidate, name) is not expected:
            reasons.append(f"mandatory gate failed: {name}")
    if candidate.bundle_size_bytes > policy.max_bundle_size_bytes:
        reasons.append("bundle size exceeds operational gate")
    missing_metrics = sorted(set(policy.metric_names) - candidate.development_metrics.keys())
    if missing_metrics:
        reasons.append(f"required ranking metrics are missing: {missing_metrics}")
    return reasons


def _rank_value(candidate: CandidateRecord, field: str) -> float:
    if field == "bundle_size_bytes":
        return float(candidate.bundle_size_bytes)
    return candidate.development_metrics[field]


def _same_rank_values(
    left: CandidateRecord, right: CandidateRecord, policy: PromotionPolicy
) -> bool:
    return all(
        _rank_value(left, rule.field) == _rank_value(right, rule.field) for rule in policy.ranking
    )


def _compare(
    left: CandidateRecord,
    right: CandidateRecord,
    *,
    policy: PromotionPolicy,
    incumbent_identity: str | None,
) -> int:
    for rule in policy.ranking:
        left_value = _rank_value(left, rule.field)
        right_value = _rank_value(right, rule.field)
        if left_value == right_value:
            continue
        if rule.direction == "maximize":
            return -1 if left_value > right_value else 1
        return -1 if left_value < right_value else 1
    if policy.prefer_incumbent_on_metric_tie and _same_rank_values(left, right, policy):
        if left.immutable_identity == incumbent_identity:
            return -1
        if right.immutable_identity == incumbent_identity:
            return 1
    if left.candidate_id == right.candidate_id:
        return 0
    return -1 if left.candidate_id < right.candidate_id else 1


def select_candidates(
    raw_candidates: list[CandidateRecord | dict[str, Any]],
    policy: PromotionPolicy,
    *,
    incumbent_identity: str | None,
) -> SelectionResult:
    """Validate, gate, rank, and compare candidates without external side effects."""

    parsed: list[CandidateRecord] = []
    rejected: list[Rejection] = []
    for index, raw in enumerate(raw_candidates):
        try:
            candidate = (
                raw
                if isinstance(raw, CandidateRecord)
                else CandidateRecord.from_mapping(
                    raw,
                    forbidden_key_fragments=policy.forbidden_key_fragments,
                    forbidden_key_prefixes=policy.forbidden_key_prefixes,
                )
            )
        except CandidateMetadataError as error:
            candidate_id = (
                raw.get("candidate_id", f"candidate-{index}")
                if isinstance(raw, dict)
                else f"candidate-{index}"
            )
            rejected.append(Rejection(str(candidate_id), "invalid_metadata", (str(error),)))
            continue
        reasons = _gate_reasons(candidate, policy)
        if reasons:
            rejected.append(Rejection(candidate.candidate_id, "policy_gate", tuple(reasons)))
        else:
            parsed.append(candidate)

    ranked = sorted(
        parsed,
        key=cmp_to_key(
            lambda left, right: _compare(
                left,
                right,
                policy=policy,
                incumbent_identity=incumbent_identity,
            )
        ),
    )
    metric_names = tuple(rule.field for rule in policy.ranking)
    explanation = tuple(
        {
            "rank": index,
            "candidate_id": candidate.candidate_id,
            "immutable_identity": candidate.immutable_identity,
            "values": {name: _rank_value(candidate, name) for name in metric_names},
            "incumbent": candidate.immutable_identity == incumbent_identity,
        }
        for index, candidate in enumerate(ranked, start=1)
    )
    if not ranked:
        outcome = (
            "blocked_invalid_metadata"
            if any(item.category == "invalid_metadata" for item in rejected)
            else "no_eligible_candidate"
        )
        return SelectionResult(
            outcome=outcome,
            winner=None,
            incumbent_identity=incumbent_identity,
            eligible=(),
            rejected=tuple(rejected),
            ranking_explanation=(),
            requested_action="none",
        )
    winner = ranked[0]
    retained = winner.immutable_identity == incumbent_identity
    return SelectionResult(
        outcome="retain_current" if retained else "promote",
        winner=winner,
        incumbent_identity=incumbent_identity,
        eligible=tuple(ranked),
        rejected=tuple(rejected),
        ranking_explanation=explanation,
        requested_action="none" if retained else "move_alias",
    )
