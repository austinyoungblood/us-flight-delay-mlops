from __future__ import annotations

import pytest

from flight_delay.modeling.selection import choose_winner, select_threshold, validation_gates


def _passing_metrics() -> dict[str, float | bool]:
    return {
        "lineage_verified": True,
        "leakage_check_passed": True,
        "average_precision": 0.35,
        "roc_auc": 0.65,
        "brier_score": 0.17,
        "log_loss": 0.51,
        "probability_mean": 0.24,
        "prevalence": 0.23,
        "recall": 0.65,
        "f1": 0.44,
    }


def test_threshold_selection_is_deterministic_and_recall_constrained() -> None:
    first = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], minimum_recall=0.60)
    second = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], minimum_recall=0.60)
    assert first == second
    assert first.recall >= 0.60
    assert first.f1 == 1.0


def test_validation_gates_and_winner_ordering() -> None:
    metrics = _passing_metrics()
    gates = validation_gates(metrics, ece=0.02, latency_p95_ms=4.0, bundle_bytes=1000)
    assert all(gates.values())
    worse = dict(metrics, average_precision=0.34)
    candidates = {
        "a": {"metrics": worse, "gates": gates, "latency": {"p95_ms": 4}, "bundle_bytes": 900},
        "b": {"metrics": metrics, "gates": gates, "latency": {"p95_ms": 5}, "bundle_bytes": 1000},
    }
    assert choose_winner(candidates) == "b"


def test_no_gate_eligible_candidate_stops_selection() -> None:
    metrics = _passing_metrics()
    gates = validation_gates(metrics, ece=0.20, latency_p95_ms=4, bundle_bytes=1000)
    with pytest.raises(ValueError, match="no candidate"):
        choose_winner(
            {
                "candidate": {
                    "metrics": metrics,
                    "gates": gates,
                    "latency": {"p95_ms": 4},
                    "bundle_bytes": 1000,
                }
            }
        )
