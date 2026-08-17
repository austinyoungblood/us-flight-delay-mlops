"""Temporal-robustness ranking, advancement, and the unrelaxed November gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flight_delay.modeling.v3.protocol import FOLD_IDS
from flight_delay.modeling.v3.selection import (
    V3SelectionError,
    advance_family,
    choose_november_winner,
    evaluate_november_gates,
    finalist_evidence,
    fold_metrics,
    rank_candidates,
    ranking_order,
    screening_confirmation_differences,
    search_operating_point,
    summarize_candidate,
)


def make_folds(precisions: list[float], **overrides: Any) -> list[dict[str, Any]]:
    return [
        {
            "fold_id": fold_id,
            "max_precision_at_operating_recall": precision,
            "average_precision": overrides.get("average_precision", 0.30),
            "roc_auc": overrides.get("roc_auc", 0.64),
            "log_loss": overrides.get("log_loss", 0.50),
            "brier_score": overrides.get("brier_score", 0.16),
        }
        for fold_id, precision in zip(FOLD_IDS, precisions, strict=True)
    ]


def candidate(candidate_id: str, precisions: list[float], **overrides: Any) -> dict[str, Any]:
    return summarize_candidate(
        candidate_id=candidate_id,
        family=overrides.pop("family", "lightgbm"),
        base_configuration=overrides.pop("base_configuration", "LGBM12"),
        weight_policy=overrides.pop("weight_policy", "UNIFORM"),
        backend="CPU",
        folds=make_folds(precisions, **overrides),
    )


def test_summary_computes_every_ranking_statistic() -> None:
    summary = candidate("LGBM12-UNIFORM", [0.30, 0.34, 0.36, 0.28])
    assert summary["worst_fold_operating_precision"] == pytest.approx(0.28)
    assert summary["fold_4_november_operating_precision"] == pytest.approx(0.28)
    assert summary["mean_fold_2_through_fold_4_operating_precision"] == pytest.approx(
        np.mean([0.34, 0.36, 0.28])
    )
    assert summary["mean_all_fold_operating_precision"] == pytest.approx(0.32)


def test_summary_requires_the_four_ordered_folds() -> None:
    with pytest.raises(V3SelectionError, match="four ordered"):
        summarize_candidate(
            candidate_id="X",
            family="lightgbm",
            base_configuration="LGBM12",
            weight_policy="UNIFORM",
            backend="CPU",
            folds=make_folds([0.3, 0.3, 0.3, 0.3])[:3],
        )


def test_worst_fold_outranks_a_stronger_summer_average() -> None:
    """The v3 objective: temporal robustness beats strong summer performance."""

    summer = candidate("A-UNIFORM", [0.44, 0.44, 0.44, 0.20])
    steady = candidate("B-UNIFORM", [0.32, 0.32, 0.32, 0.31])
    ranked = rank_candidates([summer, steady])
    assert [row["candidate_id"] for row in ranked] == ["B-UNIFORM", "A-UNIFORM"]
    assert summer["mean_all_fold_operating_precision"] > steady["mean_all_fold_operating_precision"]


def test_november_fold_breaks_a_worst_fold_tie() -> None:
    weaker_november = candidate("A-UNIFORM", [0.30, 0.40, 0.40, 0.32])
    stronger_november = candidate("B-UNIFORM", [0.30, 0.35, 0.35, 0.38])
    ranked = rank_candidates([weaker_november, stronger_november])
    assert [row["candidate_id"] for row in ranked] == ["B-UNIFORM", "A-UNIFORM"]


def test_ranking_falls_through_to_lexical_identity() -> None:
    first = candidate("CB04-EXP120", [0.31, 0.32, 0.33, 0.34])
    second = candidate("CB04-UNIFORM", [0.31, 0.32, 0.33, 0.34])
    ranked = rank_candidates([second, first])
    assert [row["candidate_id"] for row in ranked] == ["CB04-EXP120", "CB04-UNIFORM"]


def test_ranking_order_matches_the_frozen_protocol(v3_protocol: dict) -> None:
    assert list(ranking_order()) == v3_protocol["search_metric"]["candidate_ranking"]


def test_advancement_takes_the_top_two_then_top_one() -> None:
    rows = [
        candidate("LGBM12-UNIFORM", [0.30, 0.30, 0.30, 0.30]),
        candidate("LGBM12-EXP120", [0.33, 0.33, 0.33, 0.33]),
        candidate("LGBM10-UNIFORM", [0.31, 0.31, 0.31, 0.31]),
        candidate("LGBM10-EXP120", [0.29, 0.29, 0.29, 0.29]),
    ]
    top_two = advance_family(rows, family="lightgbm", expected=4, advance=2)
    assert [row["candidate_id"] for row in top_two] == ["LGBM12-EXP120", "LGBM10-UNIFORM"]
    top_one = advance_family(list(top_two), family="lightgbm", expected=2, advance=1)
    assert [row["candidate_id"] for row in top_one] == ["LGBM12-EXP120"]


def test_advancement_requires_every_candidate_to_complete() -> None:
    rows = [candidate("LGBM12-UNIFORM", [0.3, 0.3, 0.3, 0.3])]
    with pytest.raises(V3SelectionError, match="4 unique"):
        advance_family(rows, family="lightgbm", expected=4, advance=2)
    incomplete = [candidate(f"C{index}", [0.3, 0.3, 0.3, 0.3]) for index in range(4)]
    incomplete[0]["status"] = "failed"
    with pytest.raises(V3SelectionError, match="every candidate"):
        advance_family(incomplete, family="lightgbm", expected=4, advance=2)


def test_screening_versus_cpu_differences_are_recorded() -> None:
    screening = [candidate("LGBM12-UNIFORM", [0.30, 0.30, 0.30, 0.30])]
    confirmation = [candidate("LGBM12-UNIFORM", [0.32, 0.32, 0.32, 0.32])]
    differences = screening_confirmation_differences(screening, confirmation)
    assert differences[0]["worst_fold_operating_precision"] == pytest.approx(0.02)
    with pytest.raises(V3SelectionError, match="was not screened"):
        screening_confirmation_differences([], confirmation)


def test_operating_point_respects_recall_and_ppr_constraints() -> None:
    labels = [0, 1] * 50
    scores = np.linspace(0.01, 0.99, 100)
    point = search_operating_point(labels, scores)
    if point.threshold is not None:
        assert point.recall >= 0.60
        assert point.predicted_positive_rate <= 0.50


def test_unreachable_operating_region_scores_zero() -> None:
    labels = [0] * 90 + [1] * 10
    # Every positive sits at the very bottom, so recall 0.60 is unreachable under PPR 0.50.
    scores = np.concatenate([np.linspace(0.6, 0.99, 90), np.linspace(0.01, 0.05, 10)])
    point = search_operating_point(labels, scores)
    assert point.threshold is None
    assert point.precision == 0.0
    metrics = fold_metrics(labels, scores)
    assert metrics["max_precision_at_operating_recall"] == 0.0


def _passing_metrics() -> dict[str, Any]:
    return {
        "recall": 0.62,
        "precision": 0.33,
        "f1": 0.43,
        "predicted_positive_rate": 0.40,
        "brier_skill_score": 0.02,
        "brier_score": 0.15,
        "prior_brier_score": 0.16,
        "log_loss": 0.48,
        "prior_log_loss": 0.50,
        "absolute_probability_prevalence_gap": 0.01,
        "equal_frequency_ece_15": 0.02,
        "average_precision": 0.31,
        "roc_auc": 0.65,
        "single_row_inference_p95_ms": 5.0,
        "serialized_bundle_bytes": 1024,
    }


def _passing_governance(v3_protocol: dict) -> dict[str, bool]:
    return dict.fromkeys(v3_protocol["november_selection"]["acceptance_gates"]["governance"], True)


def test_all_gates_pass_for_a_qualifying_finalist(v3_protocol: dict) -> None:
    gates = evaluate_november_gates(
        metrics=_passing_metrics(),
        governance=_passing_governance(v3_protocol),
        protocol=v3_protocol,
    )
    assert all(gate.passed for gate in gates)
    assert {gate.gate_name for gate in gates} >= {
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
        "seasonal_prior_year_check_passed",
        "weight_policy_check_passed",
    }


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("precision", 0.29, "precision"),
        ("recall", 0.59, "recall"),
        ("f1", 0.37, "f1"),
        ("predicted_positive_rate", 0.51, "predicted_positive_rate"),
        ("average_precision", 0.20, "average_precision"),
        ("roc_auc", 0.55, "roc_auc"),
        ("equal_frequency_ece_15", 0.20, "equal_frequency_ece_15"),
        ("single_row_inference_p95_ms", 40.0, "single_row_inference_p95_ms"),
    ],
)
def test_each_locked_threshold_is_enforced(
    v3_protocol: dict, field: str, value: float, gate: str
) -> None:
    metrics = _passing_metrics()
    metrics[field] = value
    gates = evaluate_november_gates(
        metrics=metrics, governance=_passing_governance(v3_protocol), protocol=v3_protocol
    )
    failed = {item.gate_name for item in gates if not item.passed}
    assert gate in failed


def test_a_failed_governance_flag_fails_the_finalist(v3_protocol: dict) -> None:
    governance = _passing_governance(v3_protocol)
    governance["seasonal_prior_year_check_passed"] = False
    gates = evaluate_november_gates(
        metrics=_passing_metrics(), governance=governance, protocol=v3_protocol
    )
    assert not all(gate.passed for gate in gates)


def test_finalist_without_an_eligible_threshold_fails(v3_protocol: dict) -> None:
    labels = [0] * 90 + [1] * 10
    scores = np.concatenate([np.linspace(0.6, 0.99, 90), np.linspace(0.01, 0.05, 10)])
    evidence = finalist_evidence(
        finalist_id="ENS50-none",
        labels=labels,
        probabilities=scores,
        audit_metrics={},
        governance=_passing_governance(v3_protocol),
        protocol=v3_protocol,
    )
    assert evidence["status"] == "no_eligible_threshold"
    assert evidence["passed"] is False


def test_no_passing_finalist_yields_a_governed_stop() -> None:
    assert choose_november_winner([{"finalist_id": "A", "passed": False}]) is None
    assert choose_november_winner([]) is None


def test_winner_selection_follows_the_locked_order() -> None:
    def finalist(name: str, precision: float) -> dict[str, Any]:
        return {
            "finalist_id": name,
            "passed": True,
            "metrics": {
                "max_precision_at_operating_recall": precision,
                "average_precision": 0.31,
                "roc_auc": 0.65,
                "log_loss": 0.48,
                "brier_score": 0.15,
                "f1": 0.43,
                "predicted_positive_rate": 0.40,
            },
        }

    winner = choose_november_winner([finalist("ENS25-none", 0.31), finalist("CB04-sigmoid", 0.34)])
    assert winner is not None
    assert winner["finalist_id"] == "CB04-sigmoid"
