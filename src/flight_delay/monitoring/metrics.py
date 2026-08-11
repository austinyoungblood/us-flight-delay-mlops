"""Pure monitoring normalization, operational metrics, drift, and feedback metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-8


def prediction_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten deserialized event items while retaining nullable optional fields."""

    rows: list[dict[str, Any]] = []
    for item in items:
        request = item.get("request") or {}
        feedback = item.get("feedback") or {}
        origin = request.get("origin")
        destination = request.get("destination")
        departure = request.get("scheduled_departure")
        try:
            departure_hour = int(str(departure).split(":", maxsplit=1)[0])
        except (TypeError, ValueError):
            departure_hour = None
        rows.append(
            {
                **item,
                "carrier": request.get("carrier"),
                "origin": origin,
                "destination": destination,
                "route": f"{origin}-{destination}" if origin and destination else None,
                "distance": request.get("distance_miles"),
                "scheduled_elapsed_time": request.get("scheduled_elapsed_minutes"),
                "scheduled_departure_hour": departure_hour,
                "month": _month(request.get("flight_date")),
                "actual_delayed": feedback.get("actual_delayed"),
                "feedback_correct": feedback.get("feedback_correct"),
                "feedback_at": feedback.get("feedback_at"),
                "demo_data": bool(item.get("demo_data", False)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("created_at", "feedback_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def _month(value: object) -> int | None:
    try:
        return int(str(value).split("-", maxsplit=2)[1])
    except (IndexError, TypeError, ValueError):
        return None


def latency_percentiles(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(clean.quantile(0.5)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def operational_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    success = int((frame.get("request_status") == "success").sum()) if total else 0
    cache_hits = int(frame.get("cache_hit", pd.Series(dtype=bool)).fillna(False).sum())
    result: dict[str, Any] = {
        "request_count": total,
        "success_count": success,
        "error_count": total - success,
        "success_rate": success / total if total else None,
        "cache_hit_count": cache_hits,
        "cache_hit_rate": cache_hits / total if total else None,
    }
    for column in ("latency_ms", "inference_latency_ms", "persistence_latency_ms"):
        result[column] = latency_percentiles(frame.get(column, pd.Series(dtype=float)))
    return result


def target_drift(frame: pd.DataFrame, baseline_prevalence: float | None) -> dict[str, Any]:
    successful = frame[frame.get("request_status") == "success"] if not frame.empty else frame
    values = successful.get("predicted_delayed", pd.Series(dtype=bool)).dropna()
    live = float(values.astype(bool).mean()) if len(values) else None
    delta = (
        abs(live - baseline_prevalence)
        if live is not None and baseline_prevalence is not None
        else None
    )
    return {
        "n_success": len(successful),
        "live_predicted_positive_prevalence": live,
        "training_delayed_prevalence": baseline_prevalence,
        "absolute_prevalence_delta": delta,
    }


def population_stability_index(
    values: pd.Series, baseline: dict[str, Any] | None, *, epsilon: float = EPSILON
) -> dict[str, Any]:
    """Compute PSI with fixed baseline edges and epsilon smoothing."""

    if not baseline or not baseline.get("bin_edges") or not baseline.get("bin_proportions"):
        return {"status": "insufficient_baseline", "value": None}
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if not len(clean):
        return {"status": "insufficient_live_data", "value": None}
    edges = np.asarray(baseline["bin_edges"], dtype=float)
    expected = np.asarray(baseline["bin_proportions"], dtype=float)
    if len(edges) != len(expected) + 1:
        return {"status": "insufficient_baseline", "value": None}
    actual, _ = np.histogram(clean, bins=edges)
    actual = actual / actual.sum()
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    value = float(np.sum((actual - expected) * np.log(actual / expected)))
    return {"status": "ok", "value": value, "n": len(clean)}


def jensen_shannon_divergence(
    values: pd.Series, baseline: dict[str, float] | None, *, epsilon: float = EPSILON
) -> dict[str, Any]:
    """Compute categorical JS divergence with unseen values folded into __OTHER__."""

    if not baseline:
        return {"status": "insufficient_baseline", "value": None}
    clean = values.dropna().astype(str)
    if clean.empty:
        return {"status": "insufficient_live_data", "value": None}
    categories = sorted(set(baseline) | set(clean.unique()) | {"__OTHER__"})
    expected_map = dict(baseline)
    actual_counts = clean.value_counts(normalize=True).to_dict()
    known = set(baseline) - {"__OTHER__"}
    unseen_actual = sum(value for key, value in actual_counts.items() if key not in known)
    expected = np.array([expected_map.get(key, 0.0) for key in categories], dtype=float)
    actual = np.array(
        [
            unseen_actual if key == "__OTHER__" else actual_counts.get(key, 0.0)
            for key in categories
        ],
        dtype=float,
    )
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected /= expected.sum()
    actual /= actual.sum()
    midpoint = (expected + actual) / 2
    value = 0.5 * np.sum(expected * np.log(expected / midpoint)) + 0.5 * np.sum(
        actual * np.log(actual / midpoint)
    )
    return {"status": "ok", "value": float(value), "n": len(clean)}


def feedback_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame.get("request_status") == "success"] if not frame.empty else frame
    labeled = (
        successful[successful.get("actual_delayed").notna()] if not successful.empty else successful
    )
    n_success = len(successful)
    n_feedback = len(labeled)
    result: dict[str, Any] = {
        "n_success": n_success,
        "n_feedback": n_feedback,
        "coverage": n_feedback / n_success if n_success else None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "brier_score": None,
        "correct_count": 0,
        "incorrect_count": 0,
    }
    if not n_feedback:
        return result
    actual = labeled["actual_delayed"].astype(bool).to_numpy()
    predicted = labeled["predicted_delayed"].astype(bool).to_numpy()
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    correct = int(np.sum(actual == predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    probability = pd.to_numeric(labeled.get("delay_probability"), errors="coerce")
    valid_probability = probability.notna()
    brier = (
        float(np.mean((probability[valid_probability] - actual[valid_probability]) ** 2))
        if valid_probability.any()
        else None
    )
    result.update(
        {
            "accuracy": correct / n_feedback,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier_score": brier,
            "correct_count": correct,
            "incorrect_count": n_feedback - correct,
        }
    )
    return result


def finite_metric(value: float | None) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.3f}"
