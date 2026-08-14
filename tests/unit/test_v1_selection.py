from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from flight_delay.modeling.v1_selection import (
    GateEvidence,
    ThresholdRow,
    V1SelectionError,
    all_gates_pass,
    choose_november_winner,
    choose_threshold_row,
    evaluate_november_gates,
    evaluate_qualification_gates,
    probability_metrics,
    rank_rolling_candidates,
    select_v1_threshold,
    top_two_catboost,
)

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())


def _row(
    threshold: float,
    *,
    f1: float = 0.5,
    precision: float = 0.4,
    recall: float = 0.7,
    positive_rate: float = 0.4,
    eligible: bool = True,
) -> ThresholdRow:
    return ThresholdRow(threshold, precision, recall, f1, positive_rate, eligible)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (_row(0.4, f1=0.6), _row(0.6, f1=0.5), 0.4),
        (_row(0.4, precision=0.5), _row(0.6, precision=0.4), 0.4),
        (_row(0.4, recall=0.8), _row(0.6, recall=0.7), 0.4),
        (_row(0.4, positive_rate=0.3), _row(0.6, positive_rate=0.4), 0.4),
        (_row(0.45), _row(0.8), 0.45),
        (_row(0.4), _row(0.6), 0.6),
    ],
)
def test_threshold_selector_every_tie_break(
    left: ThresholdRow, right: ThresholdRow, expected: float
) -> None:
    assert choose_threshold_row([right, left]).threshold == expected


def test_threshold_constraints_reject_low_ppr_and_no_eligible() -> None:
    result = select_v1_threshold(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.1],
        recall_min=0.6,
        precision_min=0.3,
        predicted_positive_rate_max=0.5,
    )
    low = next(row for row in result.threshold_table if row.threshold == 0.1)
    assert low.predicted_positive_rate == 1.0 and low.eligible is False
    assert result.selected_threshold == 0.8
    none = select_v1_threshold(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.1],
        recall_min=1.1,
        precision_min=0.3,
        predicted_positive_rate_max=0.5,
    )
    assert none.selected_threshold is None and none.selected_metrics is None


def test_threshold_precision_recall_repeated_probability_and_determinism() -> None:
    arguments = ([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2])
    first = select_v1_threshold(
        *arguments, recall_min=0.5, precision_min=0.6, predicted_positive_rate_max=0.5
    )
    second = select_v1_threshold(
        *arguments, recall_min=0.5, precision_min=0.6, predicted_positive_rate_max=0.5
    )
    assert first.as_dict() == second.as_dict()
    assert len(first.threshold_table) == 2
    assert first.selected_threshold is None


def _rolling(candidate_id: str, **overrides: float) -> dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "status": "completed",
        "mean_average_precision": 0.5,
        "mean_roc_auc": 0.7,
        "mean_log_loss": 0.4,
        "mean_brier_score": 0.15,
        "std_average_precision": 0.02,
    }
    row.update(overrides)
    return row


def test_rolling_ranking_exact_order_and_top_two_cardinality() -> None:
    rows = [
        _rolling("CB4"),
        _rolling("CB3", std_average_precision=0.01),
        _rolling("CB2", mean_brier_score=0.14),
        _rolling("CB1", mean_log_loss=0.39),
    ]
    assert [row["candidate_id"] for row in rank_rolling_candidates(rows)] == [
        "CB1",
        "CB2",
        "CB3",
        "CB4",
    ]
    assert tuple(row["candidate_id"] for row in top_two_catboost(rows)) == ("CB1", "CB2")
    with pytest.raises(V1SelectionError, match="exactly completed"):
        top_two_catboost(rows[:3])


def _good_metrics() -> dict[str, float]:
    return {
        "average_precision": 0.50,
        "roc_auc": 0.80,
        "prevalence": 0.20,
        "average_precision_lift_over_prevalence": 2.5,
        "brier_score": 0.10,
        "prior_brier_score": 0.16,
        "brier_skill_score": 0.375,
        "log_loss": 0.30,
        "prior_log_loss": 0.50,
        "absolute_probability_prevalence_gap": 0.01,
        "equal_frequency_ece_15": 0.01,
        "recall": 0.70,
        "precision": 0.40,
        "f1": 0.50,
        "predicted_positive_rate": 0.40,
        "single_row_inference_p95_ms": 2.0,
        "serialized_bundle_bytes": 1000,
    }


def _governance() -> dict[str, bool]:
    names = _protocol()["november_selection"]["acceptance_gates"]["governance"]
    return {name: True for name in names}


def test_all_november_gates_retain_auditable_protocol_evidence() -> None:
    evidence = evaluate_november_gates(
        metrics=_good_metrics(), protocol=_protocol(), governance=_governance()
    )
    names = {item.gate_name for item in evidence}
    assert len(evidence) == 23
    assert all_gates_pass(evidence)
    assert names >= {
        "average_precision_incumbent_margin",
        "brier_score_below_prior",
        "predicted_positive_rate",
        "serialization_load_inference_check_passed",
    }
    assert all(item.requirement and item.observed is not None for item in evidence)


@pytest.mark.parametrize(
    ("metric", "value", "failed_gate"),
    [
        ("average_precision", 0.28, "average_precision_incumbent_margin"),
        ("roc_auc", 0.62, "roc_auc_incumbent_margin"),
        ("average_precision_lift_over_prevalence", 1.0, "average_precision_lift_over_prevalence"),
        ("brier_skill_score", 0.0, "brier_skill_score_vs_prior"),
        ("brier_score", 0.17, "brier_score_below_prior"),
        ("log_loss", 0.51, "log_loss_below_prior"),
        ("absolute_probability_prevalence_gap", 0.04, "absolute_probability_prevalence_gap"),
        ("equal_frequency_ece_15", 0.04, "equal_frequency_ece_15"),
        ("recall", 0.59, "recall"),
        ("precision", 0.29, "precision"),
        ("f1", 0.37, "f1"),
        ("predicted_positive_rate", 0.51, "predicted_positive_rate"),
        ("single_row_inference_p95_ms", 25.0, "single_row_inference_p95_ms"),
        ("serialized_bundle_bytes", 10_485_760, "serialized_bundle_bytes"),
    ],
)
def test_each_numeric_november_gate_can_fail(metric: str, value: float, failed_gate: str) -> None:
    metrics = _good_metrics()
    metrics[metric] = value
    evidence = evaluate_november_gates(
        metrics=metrics, protocol=_protocol(), governance=_governance()
    )
    assert next(item for item in evidence if item.gate_name == failed_gate).passed is False


def test_each_governance_gate_can_fail() -> None:
    for name in _governance():
        governance = _governance()
        governance[name] = False
        evidence = evaluate_november_gates(
            metrics=_good_metrics(), protocol=_protocol(), governance=governance
        )
        assert next(item for item in evidence if item.gate_name == name).passed is False


def _finalist(finalist_id: str, **metric_overrides: float) -> dict[str, Any]:
    metrics = _good_metrics()
    metrics.update(metric_overrides)
    return {
        "finalist_id": finalist_id,
        "metrics": metrics,
        "gate_evidence": (GateEvidence("all", "is true", True, True),),
    }


def test_zero_pass_governed_stop_and_deterministic_winner_order() -> None:
    failed = _finalist("CB1-none")
    failed["gate_evidence"] = (GateEvidence("all", "is true", False, False),)
    assert choose_november_winner([failed]) is None
    candidates = [
        _finalist("CB2-none"),
        _finalist("CB1-none", log_loss=0.29),
    ]
    assert choose_november_winner(candidates)["finalist_id"] == "CB1-none"
    lexical = [_finalist("CB2-none"), _finalist("CB1-none")]
    assert choose_november_winner(lexical)["finalist_id"] == "CB1-none"


def test_probability_metrics_validate_classes_and_probability_domain() -> None:
    metrics = probability_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["brier_skill_score"] > 0
    assert metrics["average_precision_lift_over_prevalence"] == pytest.approx(2.0)
    with pytest.raises(V1SelectionError, match="both target classes"):
        probability_metrics([0, 0], [0.1, 0.2])
    with pytest.raises(V1SelectionError, match="finite"):
        probability_metrics([0, 1], [0.1, float("nan")])


def test_gate_values_come_from_protocol_not_independent_constants() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["november_selection"]["acceptance_gates"]["operating_point"]["f1_min"] = 0.51
    evidence = evaluate_november_gates(
        metrics=_good_metrics(), protocol=protocol, governance=_governance()
    )
    assert next(item for item in evidence if item.gate_name == "f1").passed is False


def test_qualification_gates_are_complete_protocol_driven_and_mandatory() -> None:
    evidence = evaluate_qualification_gates(
        metrics=_good_metrics(), protocol=_protocol(), governance_passed=True
    )
    assert len(evidence) == 13
    assert all_gates_pass(evidence)
    assert {item.gate_name for item in evidence} == {
        "brier_skill_score_vs_prior",
        "log_loss_below_prior",
        "absolute_probability_prevalence_gap",
        "equal_frequency_ece_15",
        "average_precision_lift_over_prevalence",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "predicted_positive_rate",
        "single_row_inference_p95_ms",
        "serialized_bundle_bytes",
        "lineage_schema_leakage_serialization_checks_pass",
    }
    failed = evaluate_qualification_gates(
        metrics=_good_metrics(), protocol=_protocol(), governance_passed=False
    )
    assert failed[-1].passed is False


def test_qualification_strict_and_inclusive_boundaries() -> None:
    metrics = _good_metrics()
    rules = _protocol()["december_qualification"]["gates"]
    metrics.update(
        {
            "single_row_inference_p95_ms": rules["single_row_inference_p95_ms_strict_max"],
            "serialized_bundle_bytes": rules["serialized_bundle_bytes_strict_max"],
            "predicted_positive_rate": rules["predicted_positive_rate_max"],
            "recall": rules["recall_min"],
            "precision": rules["precision_min"],
            "f1": rules["f1_min"],
            "roc_auc": rules["roc_auc_min"],
        }
    )
    by_name = {
        item.gate_name: item
        for item in evaluate_qualification_gates(
            metrics=metrics, protocol=_protocol(), governance_passed=True
        )
    }
    assert by_name["single_row_inference_p95_ms"].passed is False
    assert by_name["serialized_bundle_bytes"].passed is False
    assert by_name["predicted_positive_rate"].passed is True
    assert by_name["recall"].passed is True
