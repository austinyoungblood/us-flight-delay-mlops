"""Time-aware development partitions and probability-calibration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from flight_delay.data.preprocessing import DataQualityError


@dataclass(frozen=True)
class DevelopmentPartitions:
    base_fit: pd.DataFrame
    tuning: pd.DataFrame
    refit: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class CalibrationAudit:
    """Independent calibration metrics required by the Brief 04 release gate."""

    mean_probability_gap: float
    equal_width_ece_10: float
    equal_frequency_ece_15: float
    equal_frequency_mce_15: float
    equal_width_table_10: tuple[dict[str, float | int | None], ...]
    equal_frequency_table_15: tuple[dict[str, float | int | None], ...]


def partition_development_data(
    train: pd.DataFrame, validation: pd.DataFrame, *, date_column: str = "flight_date"
) -> DevelopmentPartitions:
    """Create the five disjoint Brief 03 development partitions without a test input."""

    for name, frame in (("train", train), ("validation", validation)):
        if date_column not in frame:
            raise DataQualityError(f"{name} is missing {date_column}")
        if pd.to_datetime(frame[date_column], errors="coerce").isna().any():
            raise DataQualityError(f"{name} contains invalid dates")
    train_dates = pd.to_datetime(train[date_column]).dt.normalize()
    validation_dates = pd.to_datetime(validation[date_column]).dt.normalize()

    def select_period(frame: pd.DataFrame, dates: pd.Series, start: str, end: str) -> pd.DataFrame:
        selected = frame.loc[dates.between(start, end, inclusive="left")].copy()
        return selected.sort_values(date_column, kind="stable")

    base_fit = select_period(train, train_dates, "2025-01-01", "2025-09-01")
    tuning = select_period(train, train_dates, "2025-09-01", "2025-10-01")
    refit = select_period(train, train_dates, "2025-01-01", "2025-10-01")
    calibration = select_period(train, train_dates, "2025-10-01", "2025-11-01")
    validation = validation.sort_values(date_column, kind="stable").copy()
    expected = {
        "base_fit": (base_fit, "2025-01-01", "2025-09-01"),
        "tuning": (tuning, "2025-09-01", "2025-10-01"),
        "refit": (refit, "2025-01-01", "2025-10-01"),
        "calibration": (calibration, "2025-10-01", "2025-11-01"),
        "validation": (validation, "2025-11-01", "2026-01-01"),
    }
    for name, (frame, start, end) in expected.items():
        dates = pd.to_datetime(frame[date_column]).dt.normalize()
        if frame.empty or not (dates.ge(start).all() and dates.lt(end).all()):
            raise DataQualityError(f"{name} does not match [{start}, {end})")
        if not dates.is_monotonic_increasing:
            raise DataQualityError(f"{name} dates are not monotonic")
    if set(tuning.index) & set(calibration.index):
        raise DataQualityError("tuning and calibration overlap")
    if set(refit.index) & set(calibration.index):
        raise DataQualityError("refit and calibration overlap")
    if validation_dates.min() <= train_dates.max():
        raise DataQualityError("validation must follow all training partitions")
    return DevelopmentPartitions(base_fit, tuning, refit, calibration, validation)


def fit_sigmoid_calibrator(
    fitted_estimator: Any, features: pd.DataFrame, target: pd.Series
) -> CalibratedClassifierCV:
    """Fit Platt scaling on a disjoint calibration split using a frozen estimator."""

    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method="sigmoid")
    calibrator.fit(features, target)
    return calibrator


def _calibration_inputs(
    target: Any, probabilities: Any, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if bins < 2 or labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("aligned one-dimensional inputs and at least two bins are required")
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration evidence requires both target classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return labels, scores


def mean_probability_gap(target: Any, probabilities: Any) -> float:
    """Return the absolute global probability/prevalence difference."""

    labels, scores = _calibration_inputs(target, probabilities, bins=2)
    return float(abs(scores.mean() - labels.mean()))


def calibration_table(
    target: Any,
    probabilities: Any,
    *,
    bins: int,
    strategy: Literal["equal_width", "equal_frequency"],
) -> tuple[list[dict[str, float | int | None]], float, float]:
    """Return deterministic calibration bins, ECE, and maximum calibration error.

    Equal-frequency boundaries are empirical quantiles. Duplicate boundaries are collapsed so
    repeated probabilities are never divided between bins.
    """

    labels, scores = _calibration_inputs(target, probabilities, bins)
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, bins + 1)
        assignments = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    elif strategy == "equal_frequency":
        edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) == 1:
            edges = np.array([edges[0], edges[0]])
        assignments = np.searchsorted(edges[1:-1], scores, side="right")
    else:
        raise ValueError(f"unsupported calibration-bin strategy: {strategy}")
    table: list[dict[str, float | int | None]] = []
    ece = 0.0
    mce = 0.0
    for number in range(len(edges) - 1):
        indices = np.flatnonzero(assignments == number)
        count = len(indices)
        if strategy == "equal_frequency" and not count:
            continue
        mean_probability = float(scores[indices].mean()) if count else None
        observed_rate = float(labels[indices].mean()) if count else None
        absolute_error = (
            abs(mean_probability - observed_rate)
            if mean_probability is not None and observed_rate is not None
            else None
        )
        if absolute_error is not None:
            ece += (count / len(labels)) * absolute_error
            mce = max(mce, absolute_error)
        table.append(
            {
                "bin": len(table) + 1,
                "count": count,
                "lower_bound": float(edges[number]),
                "upper_bound": float(edges[number + 1]),
                "mean_probability": mean_probability,
                "observed_frequency": observed_rate,
                "absolute_error": absolute_error,
            }
        )
    return table, float(ece), float(mce)


def calibration_audit(target: Any, probabilities: Any) -> CalibrationAudit:
    """Calculate all independently specified Brief 04 calibration metrics."""

    width_table, width_ece, _ = calibration_table(
        target, probabilities, bins=10, strategy="equal_width"
    )
    frequency_table, frequency_ece, frequency_mce = calibration_table(
        target, probabilities, bins=15, strategy="equal_frequency"
    )
    return CalibrationAudit(
        mean_probability_gap=mean_probability_gap(target, probabilities),
        equal_width_ece_10=width_ece,
        equal_frequency_ece_15=frequency_ece,
        equal_frequency_mce_15=frequency_mce,
        equal_width_table_10=tuple(width_table),
        equal_frequency_table_15=tuple(frequency_table),
    )


def reliability_table(
    target: Any, probabilities: Any, *, bins: int = 10
) -> tuple[list[dict[str, float | int | None]], float]:
    """Compatibility wrapper for Brief 03's equal-frequency reliability evidence."""

    table, ece, _ = calibration_table(target, probabilities, bins=bins, strategy="equal_frequency")
    return table, ece
