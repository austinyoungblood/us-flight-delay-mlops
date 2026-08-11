"""Time-aware development partitions and probability-calibration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    base_fit = train.loc[train_dates.between("2025-01-01", "2025-08-31")].copy()
    tuning = train.loc[train_dates.between("2025-09-01", "2025-09-30")].copy()
    refit = train.loc[train_dates.between("2025-01-01", "2025-09-30")].copy()
    calibration = train.loc[train_dates.between("2025-10-01", "2025-10-31")].copy()
    expected = {
        "base_fit": (base_fit, "2025-01-01", "2025-09-01"),
        "tuning": (tuning, "2025-09-01", "2025-10-01"),
        "refit": (refit, "2025-01-01", "2025-10-01"),
        "calibration": (calibration, "2025-10-01", "2025-11-01"),
        "validation": (validation.copy(), "2025-11-01", "2026-01-01"),
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
    return DevelopmentPartitions(base_fit, tuning, refit, calibration, validation.copy())


def fit_sigmoid_calibrator(
    fitted_estimator: Any, features: pd.DataFrame, target: pd.Series
) -> CalibratedClassifierCV:
    """Fit Platt scaling on a disjoint calibration split using a frozen estimator."""

    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method="sigmoid")
    calibrator.fit(features, target)
    return calibrator


def reliability_table(
    target: Any, probabilities: Any, *, bins: int = 10
) -> tuple[list[dict[str, float | int]], float]:
    """Return deterministic equal-frequency reliability bins and weighted absolute-gap ECE."""

    labels = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if bins < 2 or labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("aligned one-dimensional inputs and at least two bins are required")
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration evidence requires both target classes")
    order = np.argsort(scores, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    table: list[dict[str, float | int]] = []
    ece = 0.0
    for number, indices in enumerate(groups, start=1):
        mean_probability = float(scores[indices].mean())
        observed_rate = float(labels[indices].mean())
        weight = len(indices) / len(labels)
        ece += weight * abs(mean_probability - observed_rate)
        table.append(
            {
                "bin": number,
                "count": len(indices),
                "probability_min": float(scores[indices].min()),
                "probability_max": float(scores[indices].max()),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return table, float(ece)
