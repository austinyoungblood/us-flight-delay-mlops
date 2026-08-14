"""Lazy, exact-constructor CatBoost support for the optional governed-v1 environment."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from flight_delay.modeling.calibration import fit_calibrator
from flight_delay.modeling.v1_data import V1_CATEGORICAL_FEATURES

CATBOOST_VERSION = "1.2.10"
CATBOOST_CANDIDATE_IDS = ("CB1", "CB2", "CB3", "CB4")


class V1CatBoostError(RuntimeError):
    """Raised when optional CatBoost execution differs from the locked protocol."""


@dataclass(frozen=True)
class CatBoostCandidate:
    candidate_id: str
    parameters: dict[str, Any]


def installed_catboost_version() -> str | None:
    """Inspect distribution metadata without importing the CatBoost runtime module."""

    try:
        return importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_catboost_version() -> str:
    version = installed_catboost_version()
    if version != CATBOOST_VERSION:
        raise V1CatBoostError(f"governed v1 requires catboost=={CATBOOST_VERSION}; found {version}")
    return version


def candidate_specs(protocol: dict[str, Any]) -> tuple[CatBoostCandidate, ...]:
    """Translate exactly CB1-CB4, omitting non-constructor governance markers."""

    search = protocol["catboost_search"]
    common = dict(search["common_parameters"])
    if common.pop("early_stopping") != "disabled":
        raise V1CatBoostError("CatBoost early stopping must remain disabled")
    if common.pop("class_weights") is not None or common.pop("auto_class_weights") is not None:
        raise V1CatBoostError("CatBoost weighting is prohibited")
    candidates = tuple(
        CatBoostCandidate(
            candidate_id=str(row["id"]),
            parameters={
                **common,
                "depth": int(row["depth"]),
                "iterations": int(row["iterations"]),
                "learning_rate": float(row["learning_rate"]),
                "l2_leaf_reg": row["l2_leaf_reg"],
            },
        )
        for row in search["candidates"]
    )
    if tuple(item.candidate_id for item in candidates) != CATBOOST_CANDIDATE_IDS:
        raise V1CatBoostError("CatBoost grid must contain exactly ordered CB1-CB4")
    return candidates


def build_catboost_candidate(
    protocol: dict[str, Any],
    candidate_id: str,
    *,
    classifier_type: Callable[..., Any] | None = None,
) -> Any:
    """Build exactly one locked candidate, lazily importing CatBoost when necessary."""

    selected = {item.candidate_id: item for item in candidate_specs(protocol)}
    if candidate_id not in selected:
        raise V1CatBoostError(f"unauthorized CatBoost candidate: {candidate_id}")
    if classifier_type is None:
        require_catboost_version()
        classifier_type = importlib.import_module("catboost").CatBoostClassifier
    return classifier_type(**selected[candidate_id].parameters)


def validate_catboost_runtime_contract(
    protocol: dict[str, Any], *, classifier_type: Callable[..., Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Instantiate CB1-CB4 and verify constructor parameters before any real fit."""

    validated: dict[str, dict[str, Any]] = {}
    for spec in candidate_specs(protocol):
        try:
            model = build_catboost_candidate(
                protocol, spec.candidate_id, classifier_type=classifier_type
            )
        except Exception as error:
            raise V1CatBoostError(
                f"CatBoost {CATBOOST_VERSION} rejected {spec.candidate_id} parameters"
            ) from error
        observed = model.get_params() if hasattr(model, "get_params") else spec.parameters
        if any(observed.get(name) != value for name, value in spec.parameters.items()):
            raise V1CatBoostError(f"CatBoost constructor drifted for {spec.candidate_id}")
        validated[spec.candidate_id] = dict(spec.parameters)
    return validated


def fit_catboost_base(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    flight_dates: pd.Series,
) -> Any:
    """Fit chronologically with native category names and no evaluation/early-stop channel."""

    if not pd.to_datetime(flight_dates).is_monotonic_increasing:
        raise V1CatBoostError("CatBoost fit rows must be chronologically sorted")
    model.fit(features, target, cat_features=list(V1_CATEGORICAL_FEATURES))
    return model


def build_calibration_variant(
    fitted_base: Any,
    *,
    method: Literal["none", "sigmoid", "isotonic"],
    calibration_features: pd.DataFrame | None = None,
    calibration_target: pd.Series | None = None,
    calibrator_factory: Callable[..., Any] = fit_calibrator,
) -> Any:
    """Return raw or frozen-base calibrated probability model without mutating the base."""

    if method == "none":
        return fitted_base
    if method not in {"sigmoid", "isotonic"}:
        raise V1CatBoostError(f"unsupported v1 calibration method: {method}")
    if calibration_features is None or calibration_target is None:
        raise V1CatBoostError("calibrated variants require the locked calibration window")
    before = np.asarray(fitted_base.predict_proba(calibration_features), dtype=float)
    calibrated = calibrator_factory(
        fitted_base, calibration_features, calibration_target, method=method
    )
    after = np.asarray(fitted_base.predict_proba(calibration_features), dtype=float)
    if not np.array_equal(before, after):
        raise V1CatBoostError("calibrator construction mutated the frozen CatBoost base")
    return calibrated
