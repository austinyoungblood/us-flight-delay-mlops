from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from flight_delay.modeling.v2.selection import (
    V2SelectionError,
    advance_family,
    choose_november_winner,
    finalist_evidence,
    fold_metrics,
    rank_candidates,
    screening_confirmation_differences,
    search_operating_point,
    summarize_candidate,
)


def _candidate(candidate_id: str, family: str = "catboost") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "backend": "CPU",
        "status": "completed",
        "mean_max_precision_at_operating_recall": 0.4,
        "worst_fold_max_precision_at_operating_recall": 0.3,
        "mean_average_precision": 0.5,
        "mean_roc_auc": 0.7,
        "mean_log_loss": 0.4,
        "mean_brier_score": 0.14,
        "std_max_precision_at_operating_recall": 0.02,
    }


def test_primary_metric_finds_high_recall_bounded_ppr_point() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 0])
    scores = np.asarray([0.05, 0.10, 0.95, 0.85, 0.75, 0.70])
    point = search_operating_point(labels, scores)
    assert point.threshold == pytest.approx(0.75)
    assert point.precision == 1.0
    assert point.recall == 1.0
    assert point.predicted_positive_rate == 0.5
    assert point.best_f1_under_ppr_max == 1.0
    metrics = fold_metrics(labels, scores)
    assert metrics["max_precision_at_operating_recall"] == 1.0
    assert metrics["operating_threshold"] == pytest.approx(0.75)


def test_primary_metric_scores_zero_when_operating_region_is_empty() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.9, 0.8]
    point = search_operating_point(labels, scores)
    assert point.threshold is None
    assert point.precision == point.recall == point.predicted_positive_rate == 0.0
    with pytest.raises(V2SelectionError, match="both target classes"):
        search_operating_point([1, 1], [0.8, 0.9])


def test_candidate_summary_and_exact_ranking_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    folds = [{"fold_id": f"FOLD_{index}", **fold_metrics(labels, scores)} for index in range(1, 5)]
    summary = summarize_candidate(
        candidate_id="CB01", family="catboost", backend="CPU", folds=folds
    )
    assert summary["mean_max_precision_at_operating_recall"] == 1.0
    with pytest.raises(V2SelectionError, match="four ordered"):
        summarize_candidate(candidate_id="CB01", family="catboost", backend="CPU", folds=folds[:3])

    first = _candidate("CB01")
    second = _candidate("CB02")
    assert [row["candidate_id"] for row in rank_candidates([second, first])] == ["CB01", "CB02"]
    for name, better in (
        ("mean_max_precision_at_operating_recall", 0.5),
        ("worst_fold_max_precision_at_operating_recall", 0.4),
        ("mean_average_precision", 0.6),
        ("mean_roc_auc", 0.8),
        ("mean_log_loss", 0.3),
        ("mean_brier_score", 0.1),
        ("std_max_precision_at_operating_recall", 0.01),
    ):
        challenger = deepcopy(second)
        challenger[name] = better
        assert rank_candidates([first, challenger])[0]["candidate_id"] == "CB02"


def test_family_advancement_and_cpu_differences() -> None:
    rows = [_candidate(f"CB{index:02d}") for index in range(1, 5)]
    rows[2]["mean_max_precision_at_operating_recall"] = 0.9
    advanced = advance_family(rows, family="catboost", expected=4, advance=2)
    assert [row["candidate_id"] for row in advanced] == ["CB03", "CB01"]
    with pytest.raises(V2SelectionError, match="unique"):
        advance_family(rows[:3], family="catboost", expected=4, advance=2)

    confirmation = deepcopy(rows[:2])
    confirmation[0]["mean_average_precision"] += 0.02
    differences = screening_confirmation_differences(rows, confirmation)
    assert differences[0]["mean_average_precision"] == pytest.approx(0.02)
    with pytest.raises(V2SelectionError, match="not screened"):
        screening_confirmation_differences(rows[:1], confirmation)


def _governance(protocol: dict[str, Any]) -> dict[str, bool]:
    return {name: True for name in protocol["november_selection"]["acceptance_gates"]["governance"]}


def test_finalist_threshold_short_circuit_and_all_gates(v2_protocol: dict[str, Any]) -> None:
    no_threshold = finalist_evidence(
        labels=[1, 1, 0, 0],
        probabilities=[0.1, 0.2, 0.9, 0.8],
        audit_metrics={"equal_frequency_ece_15": 0.0},
        governance=_governance(v2_protocol),
        protocol=v2_protocol,
    )
    assert no_threshold["status"] == "no_eligible_threshold"
    assert no_threshold["metrics"] is None
    assert no_threshold["gate_evidence"] == ()

    labels = np.asarray([0] * 5 + [1] * 5)
    scores = np.asarray([0.02] * 5 + [0.98] * 5)
    passing = finalist_evidence(
        labels=labels,
        probabilities=scores,
        audit_metrics={
            "equal_frequency_ece_15": 0.02,
            "single_row_inference_p95_ms": 1.0,
            "serialized_bundle_bytes": 1000,
        },
        governance=_governance(v2_protocol),
        protocol=v2_protocol,
    )
    assert passing["status"] == "completed"
    assert passing["passed"] is True
    assert all(item.passed for item in passing["gate_evidence"])

    failed_governance = _governance(v2_protocol)
    failed_governance["historical_state_integrity_passed"] = False
    failing = finalist_evidence(
        labels=labels,
        probabilities=scores,
        audit_metrics={
            "equal_frequency_ece_15": 0.02,
            "single_row_inference_p95_ms": 1.0,
            "serialized_bundle_bytes": 1000,
        },
        governance=failed_governance,
        protocol=v2_protocol,
    )
    assert failing["passed"] is False


def test_winner_selection_is_deterministic() -> None:
    assert choose_november_winner([]) is None
    metrics = {
        "max_precision_at_operating_recall": 0.31,
        "average_precision": 0.4,
        "roc_auc": 0.7,
        "log_loss": 0.4,
        "brier_score": 0.14,
        "f1": 0.4,
        "predicted_positive_rate": 0.45,
    }
    finalists = [
        {"finalist_id": "CB02-none", "passed": True, "metrics": dict(metrics)},
        {"finalist_id": "CB01-none", "passed": True, "metrics": dict(metrics)},
        {"finalist_id": "CB03-none", "passed": False, "metrics": dict(metrics)},
    ]
    assert choose_november_winner(finalists)["finalist_id"] == "CB01-none"
