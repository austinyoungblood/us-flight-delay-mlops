"""Deterministic threshold search and predeclared validation selection gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    f1: float
    table: tuple[dict[str, float], ...]


def select_threshold(
    target: Any, probabilities: Any, *, minimum_recall: float = 0.60
) -> ThresholdSelection:
    """Maximize F1 subject to recall, with the exact deterministic tie-break contract."""

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    rows: list[dict[str, float]] = []
    for index, threshold in enumerate(thresholds):
        denominator = precision[index] + recall[index]
        f1 = 0.0 if denominator == 0 else 2 * precision[index] * recall[index] / denominator
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1),
            }
        )
    eligible = [row for row in rows if row["recall"] >= minimum_recall]
    if not eligible:
        raise ValueError(f"no threshold satisfies recall >= {minimum_recall}")
    selected = max(
        eligible,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return ThresholdSelection(
        threshold=selected["threshold"],
        precision=selected["precision"],
        recall=selected["recall"],
        f1=selected["f1"],
        table=tuple(rows),
    )


def select_threshold_remediation(
    target: Any, probabilities: Any, *, minimum_recall: float = 0.60
) -> ThresholdSelection:
    """Apply Brief 04's vectorized threshold objective and revised tie-break ordering."""

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    rows = tuple(
        {
            "threshold": float(threshold),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, threshold in enumerate(thresholds)
    )
    eligible = [row for row in rows if row["recall"] >= minimum_recall]
    if not eligible:
        raise ValueError(f"no threshold satisfies recall >= {minimum_recall}")
    selected = max(
        eligible,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )
    return ThresholdSelection(
        threshold=selected["threshold"],
        precision=selected["precision"],
        recall=selected["recall"],
        f1=selected["f1"],
        table=rows,
    )


def november_gates(
    metrics: dict[str, float | int | bool],
    *,
    prior: dict[str, float],
    ece: float,
    latency_p95_ms: float,
    bundle_bytes: int,
) -> dict[str, bool]:
    """Evaluate every mandatory Brief 04 November selection gate."""

    return {
        "average_precision": float(metrics["average_precision"]) >= 0.320719,
        "roc_auc": float(metrics["roc_auc"]) >= 0.617293,
        "brier_score": float(metrics["brier_score"]) < prior["brier_score"],
        "log_loss": float(metrics["log_loss"]) < prior["log_loss"],
        "mean_probability_gap": float(metrics["mean_probability_gap"]) <= 0.03,
        "equal_frequency_ece_15": ece <= 0.03,
        "recall": float(metrics["recall"]) >= 0.60,
        "f1": float(metrics["f1"]) >= 0.41,
        "latency_p95": latency_p95_ms < 25.0,
        "bundle_size": bundle_bytes < 10 * 1024 * 1024,
        "lineage": bool(metrics.get("lineage_verified")),
        "schema": bool(metrics.get("schema_check_passed")),
        "leakage": bool(metrics.get("leakage_check_passed")),
        "convergence": bool(metrics.get("convergence_check_passed")),
        "serialization": bool(metrics.get("serialization_check_passed")),
    }


def choose_remediation_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best all-gates-passing November finalist."""

    eligible = [candidate for candidate in candidates if all(candidate["gates"].values())]
    if not eligible:
        raise ValueError("no calibrated finalist passes every mandatory November gate")
    return min(
        eligible,
        key=lambda item: (
            -float(item["metrics"]["average_precision"]),
            -float(item["metrics"]["roc_auc"]),
            float(item["metrics"]["brier_score"]),
            float(item["calibration"]["equal_frequency_ece_15"]),
            float(item["latency"]["p95_ms"]),
            str(item["finalist_id"]),
        ),
    )


def validation_gates(
    metrics: dict[str, float | int], *, ece: float, latency_p95_ms: float, bundle_bytes: int
) -> dict[str, bool]:
    """Evaluate every mandatory validation gate without discretionary judgment."""

    prevalence_gap = abs(float(metrics["probability_mean"]) - float(metrics["prevalence"]))
    return {
        "lineage": bool(metrics.get("lineage_verified")),
        "leakage": bool(metrics.get("leakage_check_passed")),
        "average_precision": float(metrics["average_precision"]) >= 0.320719,
        "roc_auc": float(metrics["roc_auc"]) >= 0.617293,
        "brier_score": float(metrics["brier_score"]) < 0.180962,
        "log_loss": float(metrics["log_loss"]) < 0.548087,
        "mean_probability_gap": prevalence_gap <= 0.03,
        "ece": ece <= 0.03,
        "recall": float(metrics["recall"]) >= 0.60,
        "f1": float(metrics["f1"]) >= 0.41,
        "latency_p95": latency_p95_ms < 25.0,
        "bundle_size": bundle_bytes < 10 * 1024 * 1024,
    }


def choose_winner(candidates: dict[str, dict[str, Any]]) -> str:
    """Choose among all-gates-passing candidates using the declared ordering."""

    eligible = [
        (name, evidence) for name, evidence in candidates.items() if all(evidence["gates"].values())
    ]
    if not eligible:
        raise ValueError("no candidate passes every mandatory validation gate")
    return max(
        eligible,
        key=lambda item: (
            item[1]["metrics"]["average_precision"],
            -item[1]["metrics"]["log_loss"],
            -item[1]["metrics"]["brier_score"],
            item[1]["metrics"]["f1"],
            -item[1]["latency"]["p95_ms"],
            -item[1]["bundle_bytes"],
        ),
    )[0]
