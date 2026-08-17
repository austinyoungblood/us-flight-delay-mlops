"""V3 operating-region metrics, temporal-robustness ranking, and November gates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from flight_delay.modeling.v1_selection import (
    GateEvidence,
    probability_metrics,
    select_v1_threshold,
)
from flight_delay.modeling.v2.selection import fold_metrics, search_operating_point
from flight_delay.modeling.v3.protocol import CANDIDATE_RANKING_ORDER, FOLD_IDS

__all__ = [
    "V3SelectionError",
    "advance_family",
    "choose_november_winner",
    "evaluate_november_gates",
    "finalist_evidence",
    "fold_metrics",
    "rank_candidates",
    "search_operating_point",
    "summarize_candidate",
]

OPERATING_PRECISION = "max_precision_at_operating_recall"


class V3SelectionError(ValueError):
    """Raised when v3 evidence is incomplete or cannot be ranked deterministically."""


def summarize_candidate(
    *,
    candidate_id: str,
    family: str,
    base_configuration: str,
    weight_policy: str,
    backend: str,
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one candidate's four ordered folds into every ranking statistic."""

    if len(folds) != len(FOLD_IDS) or [row.get("fold_id") for row in folds] != list(FOLD_IDS):
        raise V3SelectionError("candidate evidence requires the four ordered temporal folds")
    primary = np.asarray([row[OPERATING_PRECISION] for row in folds], dtype=float)
    return {
        "candidate_id": candidate_id,
        "family": family,
        "base_configuration": base_configuration,
        "weight_policy": weight_policy,
        "backend": backend,
        "status": "completed",
        "folds": folds,
        "worst_fold_operating_precision": float(primary.min()),
        "fold_4_november_operating_precision": float(primary[3]),
        "mean_fold_2_through_fold_4_operating_precision": float(primary[1:].mean()),
        "mean_all_fold_operating_precision": float(primary.mean()),
        "mean_average_precision": float(np.mean([row["average_precision"] for row in folds])),
        "mean_roc_auc": float(np.mean([row["roc_auc"] for row in folds])),
        "mean_log_loss": float(np.mean([row["log_loss"] for row in folds])),
        "mean_brier_score": float(np.mean([row["brier_score"] for row in folds])),
        "std_operating_precision": float(primary.std()),
    }


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the exact nine-level v3 ranking, which prioritizes temporal robustness.

    The worst fold and the November fold outrank every aggregate mean, so a candidate that is
    merely strong in summer cannot displace one that holds up across the whole horizon.
    """

    completed = [row for row in rows if row.get("status") == "completed"]
    return sorted(
        completed,
        key=lambda row: (
            -float(row["worst_fold_operating_precision"]),
            -float(row["fold_4_november_operating_precision"]),
            -float(row["mean_fold_2_through_fold_4_operating_precision"]),
            -float(row["mean_all_fold_operating_precision"]),
            -float(row["mean_average_precision"]),
            -float(row["mean_roc_auc"]),
            float(row["mean_log_loss"]),
            float(row["mean_brier_score"]),
            str(row["candidate_id"]),
        ),
    )


def ranking_order() -> tuple[str, ...]:
    return CANDIDATE_RANKING_ORDER


def advance_family(
    rows: list[dict[str, Any]], *, family: str, expected: int, advance: int
) -> tuple[dict[str, Any], ...]:
    """Rank one family and advance its top identities, requiring every candidate to complete."""

    family_rows = [row for row in rows if row.get("family") == family]
    if (
        len(family_rows) != expected
        or len({row.get("candidate_id") for row in family_rows}) != expected
    ):
        raise V3SelectionError(f"{family} advancement requires {expected} unique candidates")
    ranked = rank_candidates(family_rows)
    if len(ranked) != expected:
        raise V3SelectionError(f"{family} advancement requires every candidate to complete")
    return tuple(ranked[:advance])


def screening_confirmation_differences(
    screening: list[dict[str, Any]], confirmation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Record GPU-screening versus authoritative-CPU deltas for every confirmed identity."""

    screen_by_id = {row["candidate_id"]: row for row in screening}
    metrics = (
        "worst_fold_operating_precision",
        "fold_4_november_operating_precision",
        "mean_fold_2_through_fold_4_operating_precision",
        "mean_all_fold_operating_precision",
        "mean_average_precision",
        "mean_roc_auc",
        "mean_log_loss",
        "mean_brier_score",
    )
    differences: list[dict[str, Any]] = []
    for row in confirmation:
        candidate_id = row["candidate_id"]
        if candidate_id not in screen_by_id:
            raise V3SelectionError("CPU confirmation candidate was not screened")
        differences.append(
            {
                "candidate_id": candidate_id,
                **{
                    name: float(row[name]) - float(screen_by_id[candidate_id][name])
                    for name in metrics
                },
            }
        )
    return differences


def _gate(name: str, requirement: str, observed: Any, passed: bool) -> GateEvidence:
    return GateEvidence(name, requirement, observed, bool(passed))


def evaluate_november_gates(
    *, metrics: dict[str, Any], governance: dict[str, bool], protocol: dict[str, Any]
) -> tuple[GateEvidence, ...]:
    """Evaluate every mandatory November gate; none may be relaxed relative to v2."""

    gates = protocol["november_selection"]["acceptance_gates"]
    operating = gates["operating_point"]
    probability = gates["probability"]
    discrimination = gates["discrimination"]
    operational = gates["operational"]
    evidence = [
        _gate(
            "recall",
            f">= {operating['recall_min']}",
            metrics["recall"],
            metrics["recall"] >= operating["recall_min"],
        ),
        _gate(
            "precision",
            f">= {operating['precision_min']}",
            metrics["precision"],
            metrics["precision"] >= operating["precision_min"],
        ),
        _gate(
            "f1", f">= {operating['f1_min']}", metrics["f1"], metrics["f1"] >= operating["f1_min"]
        ),
        _gate(
            "predicted_positive_rate",
            f"<= {operating['predicted_positive_rate_max']}",
            metrics["predicted_positive_rate"],
            metrics["predicted_positive_rate"] <= operating["predicted_positive_rate_max"],
        ),
        _gate(
            "brier_skill_score",
            "> 0",
            metrics["brier_skill_score"],
            metrics["brier_skill_score"] > 0,
        ),
        _gate(
            "brier_score_below_prior",
            "< contemporaneous prior",
            metrics["brier_score"] - metrics["prior_brier_score"],
            metrics["brier_score"] < metrics["prior_brier_score"],
        ),
        _gate(
            "log_loss_below_prior",
            "< contemporaneous prior",
            metrics["log_loss"] - metrics["prior_log_loss"],
            metrics["log_loss"] < metrics["prior_log_loss"],
        ),
        _gate(
            "probability_prevalence_gap",
            f"<= {probability['absolute_probability_prevalence_gap_max']}",
            metrics["absolute_probability_prevalence_gap"],
            metrics["absolute_probability_prevalence_gap"]
            <= probability["absolute_probability_prevalence_gap_max"],
        ),
        _gate(
            "equal_frequency_ece_15",
            f"<= {probability['equal_frequency_ece_15_max']}",
            metrics["equal_frequency_ece_15"],
            metrics["equal_frequency_ece_15"] <= probability["equal_frequency_ece_15_max"],
        ),
        _gate(
            "average_precision",
            f">= {discrimination['average_precision_absolute_min']}",
            metrics["average_precision"],
            metrics["average_precision"] >= discrimination["average_precision_absolute_min"],
        ),
        _gate(
            "roc_auc",
            f">= {discrimination['roc_auc_absolute_min']}",
            metrics["roc_auc"],
            metrics["roc_auc"] >= discrimination["roc_auc_absolute_min"],
        ),
        _gate(
            "single_row_inference_p95_ms",
            f"< {operational['single_row_inference_p95_ms_strict_max']}",
            metrics["single_row_inference_p95_ms"],
            metrics["single_row_inference_p95_ms"]
            < operational["single_row_inference_p95_ms_strict_max"],
        ),
        _gate(
            "serialized_bundle_bytes",
            f"< {operational['serialized_bundle_bytes_strict_max']}",
            metrics["serialized_bundle_bytes"],
            metrics["serialized_bundle_bytes"] < operational["serialized_bundle_bytes_strict_max"],
        ),
    ]
    for name, required in gates["governance"].items():
        observed = governance.get(name, False)
        evidence.append(_gate(name, f"is {required}", observed, observed is required))
    return tuple(evidence)


def finalist_evidence(
    *,
    finalist_id: str,
    labels: Any,
    probabilities: Any,
    audit_metrics: dict[str, Any],
    governance: dict[str, bool],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Select the November threshold before evaluating every downstream gate."""

    rules = protocol["november_selection"]["threshold_objective"]["eligibility"]
    threshold = select_v1_threshold(
        labels,
        probabilities,
        recall_min=rules["recall_min"],
        precision_min=rules["precision_min"],
        predicted_positive_rate_max=rules["predicted_positive_rate_max"],
    )
    if threshold.selected_metrics is None:
        return {
            "finalist_id": finalist_id,
            "status": "no_eligible_threshold",
            "threshold_selection": threshold.as_dict(),
            "metrics": None,
            "gate_evidence": (),
            "passed": False,
        }
    metrics = probability_metrics(labels, probabilities)
    metrics.update(asdict(threshold.selected_metrics))
    metrics.update(audit_metrics)
    metrics[OPERATING_PRECISION] = metrics["precision"]
    gates = evaluate_november_gates(metrics=metrics, governance=governance, protocol=protocol)
    return {
        "finalist_id": finalist_id,
        "status": "completed",
        "threshold_selection": threshold.as_dict(),
        "metrics": metrics,
        "gate_evidence": gates,
        "passed": all(item.passed for item in gates),
    }


def choose_november_winner(finalists: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the single passing finalist under the locked order, or ``None`` for governed stop."""

    passing = [row for row in finalists if row.get("passed") is True]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            -float(row["metrics"][OPERATING_PRECISION]),
            -float(row["metrics"]["average_precision"]),
            -float(row["metrics"]["roc_auc"]),
            float(row["metrics"]["log_loss"]),
            float(row["metrics"]["brier_score"]),
            -float(row["metrics"]["f1"]),
            float(row["metrics"]["predicted_positive_rate"]),
            str(row["finalist_id"]),
        ),
    )
