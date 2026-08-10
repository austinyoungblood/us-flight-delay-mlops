"""Pure binary validation metrics, plots, and bounded latency measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised when binary evaluation would produce misleading evidence."""


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float | int]
    figures: dict[str, Any]


def evaluate_binary(y_true: Any, probabilities: Any, *, threshold: float = 0.5) -> EvaluationResult:
    """Calculate the complete Brief 02 contract for a binary validation split."""

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise EvaluationError("labels and probabilities must be non-empty aligned vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise EvaluationError("binary evaluation requires both target classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError("probabilities must be finite values in [0, 1]")
    predictions = (scores >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "predicted_positive_rate": float(predictions.mean()),
        "probability_min": float(scores.min()),
        "probability_mean": float(scores.mean()),
        "probability_median": float(np.median(scores)),
        "probability_max": float(scores.max()),
        "probability_std": float(scores.std()),
    }
    figures: dict[str, Any] = {}
    displays = {
        "confusion_matrix": lambda axis: ConfusionMatrixDisplay.from_predictions(
            labels, predictions, labels=[0, 1], ax=axis
        ),
        "precision_recall_curve": lambda axis: PrecisionRecallDisplay.from_predictions(
            labels, scores, ax=axis
        ),
        "roc_curve": lambda axis: RocCurveDisplay.from_predictions(labels, scores, ax=axis),
        "calibration": lambda axis: CalibrationDisplay.from_predictions(
            labels, scores, n_bins=10, strategy="quantile", ax=axis
        ),
    }
    for name, render in displays.items():
        figure, axis = plt.subplots(figsize=(6, 5))
        render(axis)
        figure.tight_layout()
        figures[name] = figure
    return EvaluationResult(metrics=metrics, figures=figures)


def measure_single_row_latency(
    model: Any, features: Any, *, sample_count: int = 200
) -> dict[str, float | int]:
    """Warm and time deterministic single-row probability predictions."""

    count = min(sample_count, len(features))
    if count < 1:
        raise EvaluationError("latency measurement requires at least one row")
    sample = features.iloc[:count]
    model.predict_proba(sample.iloc[[0]])
    durations: list[int] = []
    for index in range(count):
        start = time.perf_counter_ns()
        model.predict_proba(sample.iloc[[index]])
        durations.append(time.perf_counter_ns() - start)
    milliseconds = np.asarray(durations, dtype=float) / 1_000_000
    return {
        "sample_count": count,
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "max_ms": float(milliseconds.max()),
    }
