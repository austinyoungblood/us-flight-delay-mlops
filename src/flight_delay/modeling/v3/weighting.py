"""Precommitted training-weight policies.

Exactly two policies exist. Weights apply only to fit partitions: evaluation, calibration, and
selection rows are never weighted, so no weight can influence a reported metric except through the
model it trained.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from flight_delay.modeling.v3.protocol import EXPONENTIAL_HALF_LIFE_DAYS, WEIGHT_POLICY_IDS


class V3WeightError(ValueError):
    """Raised when a weight policy is unknown or cannot be applied deterministically."""


def fit_weights(
    flight_date: pd.Series, *, policy: str, fit_cutoff: str | date
) -> np.ndarray | None:
    """Return fit-row weights for a policy, or ``None`` when the policy is uniform.

    ``None`` means "pass no ``sample_weight`` to the estimator", which is exactly equivalent to a
    vector of ones and keeps the UNIFORM path byte-identical to an unweighted fit.
    """

    if policy not in WEIGHT_POLICY_IDS:
        raise V3WeightError(f"unknown v3 weight policy: {policy}")
    dates = pd.to_datetime(flight_date, errors="coerce").dt.normalize()
    if dates.empty or dates.isna().any():
        raise V3WeightError("weight policies require valid non-empty flight dates")
    if policy == "UNIFORM":
        return None

    cutoff = pd.Timestamp(
        date.fromisoformat(fit_cutoff) if isinstance(fit_cutoff, str) else fit_cutoff
    )
    age_days = (cutoff - dates).dt.days.to_numpy(dtype=float)
    if (age_days < 0).any():
        raise V3WeightError("fit rows cannot postdate the fit cutoff")
    raw = np.power(0.5, age_days / EXPONENTIAL_HALF_LIFE_DAYS)
    mean = float(raw.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise V3WeightError("exponential weights degenerated to a non-positive mean")
    weights = raw / mean
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise V3WeightError("exponential weights must be finite and strictly positive")
    return weights


def weight_summary(weights: np.ndarray | None, *, policy: str) -> dict[str, float | str | int]:
    """Return auditable statistics proving the normalization contract held."""

    if weights is None:
        return {
            "weight_policy": policy,
            "rows": 0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "normalized_to_mean_one": True,
        }
    return {
        "weight_policy": policy,
        "rows": int(weights.size),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "normalized_to_mean_one": bool(np.isclose(float(weights.mean()), 1.0, atol=1e-9)),
    }
