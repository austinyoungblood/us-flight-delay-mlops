"""Lazy native-categorical model construction with governed backend separation."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from flight_delay.modeling.v2.protocol import CATEGORICAL_FEATURES

Family = Literal["lightgbm", "catboost"]
Backend = Literal["CPU", "GPU"]
LIGHTGBM_VERSION = "4.7.0"
CATBOOST_VERSION = "1.2.10"


class V2ModelError(RuntimeError):
    """Raised when a model constructor or backend differs from the frozen policy."""


@dataclass(frozen=True)
class CandidateSpec:
    family: Family
    candidate_id: str
    identity_parameters: dict[str, Any]
    constructor_parameters: dict[str, Any]
    backend: Backend


@dataclass
class CalibratedV2Model:
    """Frozen native model plus a one-dimensional November score calibrator."""

    base_model: Any
    family: Family
    method: str
    calibrator: Any | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = predict_positive(self.base_model, features, family=self.family)
        if self.method == "sigmoid":
            positive = self.calibrator.predict_proba(positive.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            positive = self.calibrator.predict(positive)
        negative = 1.0 - positive
        return np.column_stack((negative, positive))


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_versions() -> dict[str, str]:
    versions = {
        "lightgbm": installed_version("lightgbm"),
        "catboost": installed_version("catboost"),
    }
    expected = {"lightgbm": LIGHTGBM_VERSION, "catboost": CATBOOST_VERSION}
    if versions != expected:
        raise V2ModelError(
            f"governed v2 requires exact modeling versions {expected}; found {versions}"
        )
    return {name: str(value) for name, value in versions.items()}


def candidate_specs(
    protocol: dict[str, Any], *, family: Family, backend: Backend
) -> tuple[CandidateSpec, ...]:
    """Translate the exact matrix while keeping backend out of candidate identity."""

    if family == "lightgbm" and backend != "CPU":
        raise V2ModelError("LightGBM screening and confirmation are CPU-only")
    search = protocol[f"{family}_search"]
    common = dict(search["common_parameters"])
    result: list[CandidateSpec] = []
    for row in search["candidates"]:
        identity = {name: value for name, value in row.items() if name != "id"}
        constructor = {**common, **identity}
        if family == "lightgbm":
            constructor["n_jobs"] = int(
                protocol["execution_policy"]["lightgbm_screening"]["n_jobs"]
            )
        else:
            weight = float(constructor.pop("positive_class_weight"))
            constructor["class_weights"] = [1.0, weight]
            constructor["task_type"] = backend
            if backend == "GPU":
                constructor["devices"] = str(
                    protocol["execution_policy"]["catboost_screening"]["devices"]
                )
        result.append(
            CandidateSpec(
                family=family,
                candidate_id=str(row["id"]),
                identity_parameters=identity,
                constructor_parameters=constructor,
                backend=backend,
            )
        )
    expected = 16 if family == "lightgbm" else 12
    if len(result) != expected:
        raise V2ModelError(f"{family} matrix must contain exactly {expected} candidates")
    return tuple(result)


def build_candidate(
    spec: CandidateSpec, *, classifier_type: Callable[..., Any] | None = None
) -> Any:
    """Instantiate a candidate, importing its optional runtime only when needed."""

    if classifier_type is None:
        require_versions()
        module_name, type_name = (
            ("lightgbm", "LGBMClassifier")
            if spec.family == "lightgbm"
            else ("catboost", "CatBoostClassifier")
        )
        classifier_type = getattr(importlib.import_module(module_name), type_name)
    try:
        return classifier_type(**spec.constructor_parameters)
    except Exception as error:
        raise V2ModelError(f"{spec.candidate_id} constructor rejected frozen parameters") from error


def native_categorical_frame(features: pd.DataFrame, *, family: Family) -> pd.DataFrame:
    """Return a copied frame encoded for the family's native categorical interface."""

    missing = set(CATEGORICAL_FEATURES) - set(features)
    if missing:
        raise V2ModelError(f"native categorical columns are missing: {sorted(missing)}")
    result = features.copy(deep=True)
    for column in CATEGORICAL_FEATURES:
        values = result[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise V2ModelError(f"native categorical column {column} contains invalid values")
        result[column] = values.astype("category" if family == "lightgbm" else str)
    return result


def fit_candidate(
    model: Any,
    spec: CandidateSpec,
    features: pd.DataFrame,
    target: pd.Series,
    flight_date: pd.Series,
) -> Any:
    """Fit chronologically with native categories and no outer evaluation channel."""

    if not pd.to_datetime(flight_date).is_monotonic_increasing:
        raise V2ModelError("v2 fit rows must be chronologically ordered")
    native = native_categorical_frame(features, family=spec.family)
    if spec.family == "lightgbm":
        model.fit(native, target, categorical_feature=list(CATEGORICAL_FEATURES))
    else:
        model.fit(native, target, cat_features=list(CATEGORICAL_FEATURES))
    return model


def predict_positive(model: Any, features: pd.DataFrame, *, family: Family) -> np.ndarray:
    native = native_categorical_frame(features, family=family)
    probabilities = np.asarray(model.predict_proba(native), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(features), 2):
        raise V2ModelError("model predict_proba must return one binary row per input")
    positive = probabilities[:, 1]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise V2ModelError("model probabilities must be finite values in [0, 1]")
    return positive


def build_calibration_variant(
    fitted_base: Any,
    *,
    family: Family,
    method: Literal["none", "sigmoid", "isotonic"],
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
) -> CalibratedV2Model:
    """Fit only a score calibrator while keeping the authoritative CPU base frozen."""

    before = predict_positive(fitted_base, calibration_features, family=family)
    calibrator: Any | None = None
    if method == "sigmoid":
        calibrator = LogisticRegression(random_state=42, solver="lbfgs")
        calibrator.fit(before.reshape(-1, 1), calibration_target)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(before, calibration_target)
    elif method != "none":
        raise V2ModelError(f"unsupported v2 calibration method: {method}")
    after = predict_positive(fitted_base, calibration_features, family=family)
    if not np.array_equal(before, after):
        raise V2ModelError("calibration mutated the frozen CPU base")
    return CalibratedV2Model(fitted_base, family, method, calibrator)


def validate_constructor_contract(
    protocol: dict[str, Any], *, classifier_type: Callable[..., Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Instantiate all frozen identities with a fake or exact runtime, without fitting."""

    observed: dict[str, dict[str, Any]] = {}
    for family, backend in (("lightgbm", "CPU"), ("catboost", "GPU"), ("catboost", "CPU")):
        for spec in candidate_specs(protocol, family=family, backend=backend):
            model = build_candidate(spec, classifier_type=classifier_type)
            parameters = (
                model.get_params() if hasattr(model, "get_params") else spec.constructor_parameters
            )
            if any(
                parameters.get(name) != value for name, value in spec.constructor_parameters.items()
            ):
                raise V2ModelError(f"constructor drifted for {spec.candidate_id} on {backend}")
            observed[f"{spec.candidate_id}:{backend}"] = dict(spec.constructor_parameters)
    return observed
