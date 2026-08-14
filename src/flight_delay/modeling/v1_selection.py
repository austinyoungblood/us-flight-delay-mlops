"""Pure probability, threshold, gate, and winner selection for governed v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class V1SelectionError(ValueError):
    """Raised when deterministic v1 selection cannot produce valid evidence."""


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive_rate: float
    eligible: bool


@dataclass(frozen=True)
class V1ThresholdSelection:
    selected_threshold: float | None
    selected_metrics: ThresholdRow | None
    threshold_table: tuple[ThresholdRow, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_threshold": self.selected_threshold,
            "selected_metrics": (
                asdict(self.selected_metrics) if self.selected_metrics is not None else None
            ),
            "threshold_table": [asdict(row) for row in self.threshold_table],
        }


@dataclass(frozen=True)
class GateEvidence:
    gate_name: str
    requirement: str
    observed: bool | float | int | str
    passed: bool


def _binary_inputs(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or not len(target) or len(target) != len(scores):
        raise V1SelectionError("labels and probabilities must be non-empty aligned vectors")
    if set(np.unique(target)) != {0, 1}:
        raise V1SelectionError("v1 evaluation requires both target classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise V1SelectionError("probabilities must be finite values in [0, 1]")
    return target, scores


def probability_metrics(labels: Any, probabilities: Any) -> dict[str, float]:
    """Calculate complete governed probability and contemporaneous-prior evidence."""

    target, scores = _binary_inputs(labels, probabilities)
    prevalence = float(target.mean())
    prior = np.full(len(target), prevalence, dtype=float)
    brier = float(brier_score_loss(target, scores))
    prior_brier = float(brier_score_loss(target, prior))
    model_log_loss = float(log_loss(target, scores, labels=[0, 1]))
    prior_log_loss = float(log_loss(target, prior, labels=[0, 1]))
    average_precision = float(average_precision_score(target, scores))
    return {
        "prevalence": prevalence,
        "average_precision": average_precision,
        "roc_auc": float(roc_auc_score(target, scores)),
        "brier_score": brier,
        "log_loss": model_log_loss,
        "probability_mean": float(scores.mean()),
        "probability_min": float(scores.min()),
        "probability_max": float(scores.max()),
        "probability_median": float(np.median(scores)),
        "probability_std": float(scores.std()),
        "prior_brier_score": prior_brier,
        "prior_log_loss": prior_log_loss,
        "brier_skill_score": float(1.0 - brier / prior_brier),
        "average_precision_lift_over_prevalence": float(average_precision / prevalence),
        "absolute_probability_prevalence_gap": float(abs(scores.mean() - prevalence)),
    }


def select_v1_threshold(
    labels: Any,
    probabilities: Any,
    *,
    recall_min: float,
    precision_min: float,
    predicted_positive_rate_max: float,
) -> V1ThresholdSelection:
    """Select a threshold using every unique score and the exact protocol ordering."""

    target, scores = _binary_inputs(labels, probabilities)
    rows: list[ThresholdRow] = []
    for threshold in sorted(set(map(float, scores)), reverse=True):
        predicted = (scores >= threshold).astype(int)
        precision = float(precision_score(target, predicted, zero_division=0))
        recall = float(recall_score(target, predicted, zero_division=0))
        f1 = float(f1_score(target, predicted, zero_division=0))
        positive_rate = float(predicted.mean())
        rows.append(
            ThresholdRow(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                predicted_positive_rate=positive_rate,
                eligible=(
                    recall >= recall_min
                    and precision >= precision_min
                    and positive_rate <= predicted_positive_rate_max
                ),
            )
        )
    selected = choose_threshold_row(rows)
    if selected is None:
        return V1ThresholdSelection(None, None, tuple(rows))
    return V1ThresholdSelection(selected.threshold, selected, tuple(rows))


def choose_threshold_row(rows: list[ThresholdRow]) -> ThresholdRow | None:
    """Apply all six locked threshold tie breaks to precomputed rows."""

    eligible = [row for row in rows if row.eligible]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -row.f1,
            -row.precision,
            -row.recall,
            row.predicted_positive_rate,
            abs(row.threshold - 0.50),
            -row.threshold,
        ),
    )


def rank_rolling_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank the four completed bases using the exact six-level protocol ordering."""

    completed = [row for row in rows if row.get("status") == "completed"]
    return sorted(
        completed,
        key=lambda row: (
            -float(row["mean_average_precision"]),
            -float(row["mean_roc_auc"]),
            float(row["mean_log_loss"]),
            float(row["mean_brier_score"]),
            float(row["std_average_precision"]),
            str(row["candidate_id"]),
        ),
    )


def top_two_catboost(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ranked = rank_rolling_candidates(rows)
    if len(ranked) != 4 or {row["candidate_id"] for row in ranked} != {
        "CB1",
        "CB2",
        "CB3",
        "CB4",
    }:
        raise V1SelectionError("rolling selection requires exactly completed CB1-CB4")
    return ranked[0], ranked[1]


def _gate(name: str, requirement: str, observed: Any, passed: bool) -> GateEvidence:
    return GateEvidence(name, requirement, observed, bool(passed))


def evaluate_november_gates(
    *,
    metrics: dict[str, Any],
    protocol: dict[str, Any],
    governance: dict[str, bool],
) -> tuple[GateEvidence, ...]:
    """Evaluate every mandatory November gate directly from validated protocol values."""

    gates = protocol["november_selection"]["acceptance_gates"]
    incumbent = protocol["control"]["historical_november_metrics"]
    discrimination = gates["discrimination"]
    scoring = gates["proper_scoring"]
    calibration = gates["calibration"]
    operating = gates["operating_point"]
    operational = gates["operational"]
    required_governance = gates["governance"]
    ap_margin = metrics["average_precision"] - incumbent["average_precision"]
    auc_margin = metrics["roc_auc"] - incumbent["roc_auc"]
    evidence = [
        _gate(
            "average_precision_incumbent_margin",
            f">= {discrimination['average_precision_incumbent_margin_min']}",
            ap_margin,
            ap_margin >= discrimination["average_precision_incumbent_margin_min"],
        ),
        _gate(
            "roc_auc_incumbent_margin",
            f">= {discrimination['roc_auc_incumbent_margin_min']}",
            auc_margin,
            auc_margin >= discrimination["roc_auc_incumbent_margin_min"],
        ),
        _gate(
            "average_precision_lift_over_prevalence",
            f">= {discrimination['average_precision_lift_over_prevalence_min']}",
            metrics["average_precision_lift_over_prevalence"],
            metrics["average_precision_lift_over_prevalence"]
            >= discrimination["average_precision_lift_over_prevalence_min"],
        ),
        _gate(
            "brier_skill_score_vs_prior",
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
            "brier_score_incumbent_max",
            f"<= {scoring['brier_score_incumbent_max']}",
            metrics["brier_score"],
            metrics["brier_score"] <= scoring["brier_score_incumbent_max"],
        ),
        _gate(
            "log_loss_incumbent_max",
            f"<= {scoring['log_loss_incumbent_max']}",
            metrics["log_loss"],
            metrics["log_loss"] <= scoring["log_loss_incumbent_max"],
        ),
        _gate(
            "absolute_probability_prevalence_gap",
            f"<= {calibration['absolute_probability_prevalence_gap_max']}",
            metrics["absolute_probability_prevalence_gap"],
            metrics["absolute_probability_prevalence_gap"]
            <= calibration["absolute_probability_prevalence_gap_max"],
        ),
        _gate(
            "equal_frequency_ece_15",
            f"<= {calibration['equal_frequency_ece_15_max']}",
            metrics["equal_frequency_ece_15"],
            metrics["equal_frequency_ece_15"] <= calibration["equal_frequency_ece_15_max"],
        ),
    ]
    for metric, suffix in (
        ("recall", "min"),
        ("precision", "min"),
        ("f1", "min"),
        ("predicted_positive_rate", "max"),
    ):
        limit = operating[f"{metric}_{suffix}"]
        passed = metrics[metric] >= limit if suffix == "min" else metrics[metric] <= limit
        evidence.append(
            _gate(metric, f"{'>=' if suffix == 'min' else '<='} {limit}", metrics[metric], passed)
        )
    latency_limit = operational["single_row_inference_p95_ms_strict_max"]
    size_limit = operational["serialized_bundle_bytes_strict_max"]
    evidence.extend(
        (
            _gate(
                "single_row_inference_p95_ms",
                f"< {latency_limit}",
                metrics["single_row_inference_p95_ms"],
                metrics["single_row_inference_p95_ms"] < latency_limit,
            ),
            _gate(
                "serialized_bundle_bytes",
                f"< {size_limit}",
                metrics["serialized_bundle_bytes"],
                metrics["serialized_bundle_bytes"] < size_limit,
            ),
        )
    )
    for name, required in required_governance.items():
        observed = governance.get(name, False)
        evidence.append(_gate(name, f"is {required}", observed, observed is required))
    return tuple(evidence)


def all_gates_pass(evidence: tuple[GateEvidence, ...]) -> bool:
    return bool(evidence) and all(item.passed for item in evidence)


def choose_november_winner(finalists: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the exact deterministic winner, or ``None`` for governed stop."""

    eligible = [row for row in finalists if all_gates_pass(tuple(row["gate_evidence"]))]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["metrics"]["average_precision"]),
            float(row["metrics"]["log_loss"]),
            float(row["metrics"]["brier_score"]),
            -float(row["metrics"]["roc_auc"]),
            -float(row["metrics"]["f1"]),
            float(row["metrics"]["predicted_positive_rate"]),
            float(row["metrics"]["single_row_inference_p95_ms"]),
            str(row["finalist_id"]),
        ),
    )


def evaluate_qualification_gates(
    *, metrics: dict[str, Any], protocol: dict[str, Any], governance_passed: bool
) -> tuple[GateEvidence, ...]:
    """Evaluate the locked December/future absolute gate family without mutation."""

    rules = protocol["december_qualification"]["gates"]
    checks = (
        (
            "brier_skill_score_vs_prior",
            "> 0",
            metrics["brier_skill_score"],
            metrics["brier_skill_score"] > 0,
        ),
        (
            "log_loss_below_prior",
            "< contemporaneous prior",
            metrics["log_loss"] - metrics["prior_log_loss"],
            metrics["log_loss"] < metrics["prior_log_loss"],
        ),
        (
            "absolute_probability_prevalence_gap",
            f"<= {rules['absolute_probability_prevalence_gap_max']}",
            metrics["absolute_probability_prevalence_gap"],
            metrics["absolute_probability_prevalence_gap"]
            <= rules["absolute_probability_prevalence_gap_max"],
        ),
        (
            "equal_frequency_ece_15",
            f"<= {rules['equal_frequency_ece_15_max']}",
            metrics["equal_frequency_ece_15"],
            metrics["equal_frequency_ece_15"] <= rules["equal_frequency_ece_15_max"],
        ),
        (
            "average_precision_lift_over_prevalence",
            f">= {rules['average_precision_lift_over_prevalence_min']}",
            metrics["average_precision_lift_over_prevalence"],
            metrics["average_precision_lift_over_prevalence"]
            >= rules["average_precision_lift_over_prevalence_min"],
        ),
        (
            "roc_auc",
            f">= {rules['roc_auc_min']}",
            metrics["roc_auc"],
            metrics["roc_auc"] >= rules["roc_auc_min"],
        ),
        (
            "recall",
            f">= {rules['recall_min']}",
            metrics["recall"],
            metrics["recall"] >= rules["recall_min"],
        ),
        (
            "precision",
            f">= {rules['precision_min']}",
            metrics["precision"],
            metrics["precision"] >= rules["precision_min"],
        ),
        ("f1", f">= {rules['f1_min']}", metrics["f1"], metrics["f1"] >= rules["f1_min"]),
        (
            "predicted_positive_rate",
            f"<= {rules['predicted_positive_rate_max']}",
            metrics["predicted_positive_rate"],
            metrics["predicted_positive_rate"] <= rules["predicted_positive_rate_max"],
        ),
        (
            "single_row_inference_p95_ms",
            f"< {rules['single_row_inference_p95_ms_strict_max']}",
            metrics["single_row_inference_p95_ms"],
            metrics["single_row_inference_p95_ms"]
            < rules["single_row_inference_p95_ms_strict_max"],
        ),
        (
            "serialized_bundle_bytes",
            f"< {rules['serialized_bundle_bytes_strict_max']}",
            metrics["serialized_bundle_bytes"],
            metrics["serialized_bundle_bytes"] < rules["serialized_bundle_bytes_strict_max"],
        ),
        (
            "lineage_schema_leakage_serialization_checks_pass",
            "is True",
            governance_passed,
            governance_passed is True,
        ),
    )
    return tuple(_gate(*check) for check in checks)
