"""Operating-region metrics, CPU ranking, gates, and deterministic v2 selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from flight_delay.modeling.v1_selection import (
    GateEvidence,
    probability_metrics,
    select_v1_threshold,
)


class V2SelectionError(ValueError):
    """Raised when v2 evidence is incomplete or cannot be ranked deterministically."""


@dataclass(frozen=True)
class SearchOperatingPoint:
    threshold: float | None
    precision: float
    recall: float
    predicted_positive_rate: float
    best_f1_under_ppr_max: float

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _binary(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or not len(target) or len(target) != len(scores):
        raise V2SelectionError("labels and probabilities must be aligned non-empty vectors")
    if set(np.unique(target)) != {0, 1}:
        raise V2SelectionError("v2 metrics require both target classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise V2SelectionError("probabilities must be finite values in [0, 1]")
    return target, scores


def search_operating_point(
    labels: Any,
    probabilities: Any,
    *,
    recall_min: float = 0.60,
    predicted_positive_rate_max: float = 0.50,
) -> SearchOperatingPoint:
    """Maximize precision in the frozen high-recall, bounded-PPR operating region."""

    target, scores = _binary(labels, probabilities)
    eligible: list[tuple[float, float, float, float]] = []
    f1_values: list[float] = []
    for threshold in sorted(set(map(float, scores)), reverse=True):
        predicted = (scores >= threshold).astype(int)
        positive_rate = float(predicted.mean())
        recall = float(recall_score(target, predicted, zero_division=0))
        precision = float(precision_score(target, predicted, zero_division=0))
        f1 = float(f1_score(target, predicted, zero_division=0))
        if positive_rate <= predicted_positive_rate_max:
            f1_values.append(f1)
            if recall >= recall_min:
                eligible.append((threshold, precision, recall, positive_rate))
    best_f1 = max(f1_values, default=0.0)
    if not eligible:
        return SearchOperatingPoint(None, 0.0, 0.0, 0.0, best_f1)
    threshold, precision, recall, positive_rate = min(
        eligible,
        key=lambda row: (-row[1], -row[2], row[3], abs(row[0] - 0.50), -row[0]),
    )
    return SearchOperatingPoint(threshold, precision, recall, positive_rate, best_f1)


def fold_metrics(labels: Any, probabilities: Any) -> dict[str, float | None]:
    """Return every locked search metric for one outer fold."""

    metrics: dict[str, float | None] = dict(probability_metrics(labels, probabilities))
    point = search_operating_point(labels, probabilities)
    metrics.update(
        {
            "max_precision_at_operating_recall": point.precision,
            "operating_threshold": point.threshold,
            "operating_recall": point.recall,
            "operating_ppr": point.predicted_positive_rate,
            "best_f1_under_ppr_max": point.best_f1_under_ppr_max,
        }
    )
    return metrics


def summarize_candidate(
    *,
    candidate_id: str,
    family: str,
    backend: str,
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(folds) != 4 or [row.get("fold_id") for row in folds] != [
        "FOLD_1",
        "FOLD_2",
        "FOLD_3",
        "FOLD_4",
    ]:
        raise V2SelectionError("candidate evidence requires the four ordered temporal folds")
    primary = np.asarray([row["max_precision_at_operating_recall"] for row in folds], dtype=float)
    return {
        "candidate_id": candidate_id,
        "family": family,
        "backend": backend,
        "status": "completed",
        "folds": folds,
        "mean_max_precision_at_operating_recall": float(primary.mean()),
        "worst_fold_max_precision_at_operating_recall": float(primary.min()),
        "mean_average_precision": float(np.mean([row["average_precision"] for row in folds])),
        "mean_roc_auc": float(np.mean([row["roc_auc"] for row in folds])),
        "mean_log_loss": float(np.mean([row["log_loss"] for row in folds])),
        "mean_brier_score": float(np.mean([row["brier_score"] for row in folds])),
        "std_max_precision_at_operating_recall": float(primary.std()),
    }


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the exact eight-level CPU-confirmed ranking order."""

    completed = [row for row in rows if row.get("status") == "completed"]
    return sorted(
        completed,
        key=lambda row: (
            -float(row["mean_max_precision_at_operating_recall"]),
            -float(row["worst_fold_max_precision_at_operating_recall"]),
            -float(row["mean_average_precision"]),
            -float(row["mean_roc_auc"]),
            float(row["mean_log_loss"]),
            float(row["mean_brier_score"]),
            float(row["std_max_precision_at_operating_recall"]),
            str(row["candidate_id"]),
        ),
    )


def advance_family(
    rows: list[dict[str, Any]], *, family: str, expected: int, advance: int
) -> tuple[dict[str, Any], ...]:
    family_rows = [row for row in rows if row.get("family") == family]
    if (
        len(family_rows) != expected
        or len({row.get("candidate_id") for row in family_rows}) != expected
    ):
        raise V2SelectionError(f"{family} advancement requires {expected} unique candidates")
    ranked = rank_candidates(family_rows)
    if len(ranked) != expected:
        raise V2SelectionError(f"{family} advancement requires every candidate to complete")
    return tuple(ranked[:advance])


def screening_confirmation_differences(
    screening: list[dict[str, Any]], confirmation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    screen_by_id = {row["candidate_id"]: row for row in screening}
    metrics = (
        "mean_max_precision_at_operating_recall",
        "worst_fold_max_precision_at_operating_recall",
        "mean_average_precision",
        "mean_roc_auc",
        "mean_log_loss",
        "mean_brier_score",
    )
    differences: list[dict[str, Any]] = []
    for row in confirmation:
        candidate_id = row["candidate_id"]
        if candidate_id not in screen_by_id:
            raise V2SelectionError("CPU confirmation candidate was not screened")
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
            "status": "no_eligible_threshold",
            "threshold_selection": threshold.as_dict(),
            "metrics": None,
            "gate_evidence": (),
            "passed": False,
        }
    metrics = probability_metrics(labels, probabilities)
    metrics.update(asdict(threshold.selected_metrics))
    metrics.update(audit_metrics)
    metrics["max_precision_at_operating_recall"] = metrics["precision"]
    gates = evaluate_november_gates(metrics=metrics, governance=governance, protocol=protocol)
    return {
        "status": "completed",
        "threshold_selection": threshold.as_dict(),
        "metrics": metrics,
        "gate_evidence": gates,
        "passed": all(item.passed for item in gates),
    }


def choose_november_winner(finalists: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in finalists if row.get("passed") is True]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            -float(row["metrics"]["max_precision_at_operating_recall"]),
            -float(row["metrics"]["average_precision"]),
            -float(row["metrics"]["roc_auc"]),
            float(row["metrics"]["log_loss"]),
            float(row["metrics"]["brier_score"]),
            -float(row["metrics"]["f1"]),
            float(row["metrics"]["predicted_positive_rate"]),
            str(row["finalist_id"]),
        ),
    )
