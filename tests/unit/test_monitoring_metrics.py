from __future__ import annotations

import pandas as pd
import pytest

from flight_delay.contracts import TrafficSource
from flight_delay.monitoring.metrics import (
    feedback_metrics,
    jensen_shannon_divergence,
    operational_metrics,
    population_stability_index,
    prediction_frame,
    target_drift,
)


def items() -> list[dict[str, object]]:
    return [
        {
            "prediction_id": "one",
            "created_at": "2026-08-01T08:00:00Z",
            "request_status": "success",
            "traffic_source": "traveler_ui",
            "cache_hit": False,
            "latency_ms": 10.0,
            "inference_latency_ms": 2.0,
            "persistence_latency_ms": 3.0,
            "predicted_delayed": True,
            "delay_probability": 0.8,
            "risk_band": "high",
            "model_version": "v0",
            "request": {
                "carrier": "UA",
                "origin": "DEN",
                "destination": "LAX",
                "flight_date": "2026-08-15",
                "scheduled_departure": "08:00:00",
                "scheduled_elapsed_minutes": 120,
                "distance_miles": 850,
            },
            "feedback": {"actual_delayed": True, "feedback_correct": True},
        },
        {
            "prediction_id": "two",
            "created_at": "2026-08-01T09:00:00Z",
            "request_status": "success",
            "cache_hit": True,
            "latency_ms": 30.0,
            "inference_latency_ms": 1.0,
            "persistence_latency_ms": 10.0,
            "predicted_delayed": False,
            "delay_probability": 0.2,
            "risk_band": "medium",
            "model_version": "v0",
            "demo_data": True,
            "request": {
                "carrier": "XX",
                "origin": "DEN",
                "destination": "PHX",
                "flight_date": "2026-08-16",
                "scheduled_departure": "09:00:00",
                "scheduled_elapsed_minutes": 90,
                "distance_miles": 600,
            },
            "feedback": None,
        },
    ]


def test_normalization_operational_and_target_drift() -> None:
    frame = prediction_frame(items())
    assert frame.loc[0, "route"] == "DEN-LAX"
    assert frame.loc[0, "scheduled_departure_hour"] == 8
    assert frame.loc[0, "traffic_source"] == TrafficSource.TRAVELER_UI.value
    assert frame.loc[1, "traffic_source"] == TrafficSource.LEGACY_UNATTRIBUTED.value
    assert frame["demo_data"].tolist() == [False, True]
    operational = operational_metrics(frame)
    assert operational["request_count"] == 2
    assert operational["cache_hit_rate"] == 0.5
    assert operational["latency_ms"]["p50"] == 20
    drift = target_drift(frame, 0.25)
    assert drift["live_predicted_positive_prevalence"] == 0.5
    assert drift["absolute_prevalence_delta"] == 0.25


def test_psi_and_js_handle_zero_unseen_and_missing_baseline() -> None:
    psi = population_stability_index(
        pd.Series([0.2, 0.8, 1.4]),
        {"bin_edges": [0, 0.5, 1, 2], "bin_proportions": [0.0, 0.5, 0.5]},
    )
    assert psi["status"] == "ok"
    assert psi["value"] >= 0
    js = jensen_shannon_divergence(pd.Series(["UA", "XX", "XX"]), {"UA": 0.8, "__OTHER__": 0.2})
    assert js["status"] == "ok"
    assert js["value"] >= 0
    assert population_stability_index(pd.Series([1]), None)["status"] == "insufficient_baseline"
    assert jensen_shannon_divergence(pd.Series(["UA"]), None)["status"] == "insufficient_baseline"


def test_feedback_metrics_show_coverage_and_undefined_one_class_metrics() -> None:
    frame = prediction_frame(items())
    result = feedback_metrics(frame)
    assert result["coverage"] == 0.5
    assert result["n_feedback"] == 1
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["brier_score"] == pytest.approx(0.04)

    frame.loc[0, "predicted_delayed"] = False
    frame.loc[0, "actual_delayed"] = False
    edge = feedback_metrics(frame.iloc[:1])
    assert edge["precision"] is None
    assert edge["recall"] is None
    assert edge["f1"] is None
