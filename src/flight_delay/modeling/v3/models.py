"""V3 candidate construction, expanded native categoricals, weighted fits, and ensembles."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from flight_delay.modeling.v2.models import require_versions
from flight_delay.modeling.v3.protocol import (
    CALIBRATION_VARIANTS,
    CANDIDATE_IDENTITY_IDS,
    CATEGORICAL_FEATURES,
    INTEGER_CATEGORICAL_FEATURES,
    STRING_CATEGORICAL_FEATURES,
    WEIGHT_POLICY_SUFFIX,
)
from flight_delay.modeling.v3.weighting import fit_weights

Family = Literal["lightgbm", "catboost"]
Backend = Literal["CPU", "GPU"]
CalibrationMethod = Literal["none", "sigmoid", "isotonic"]


class V3ModelError(RuntimeError):
    """Raised when a model constructor, backend, or weight policy differs from frozen policy."""


@dataclass(frozen=True)
class V3CandidateSpec:
    """One of the eight frozen identities: base hyperparameters crossed with a weight policy."""

    family: Family
    candidate_id: str
    base_configuration: str
    weight_policy: str
    identity_parameters: dict[str, Any]
    constructor_parameters: dict[str, Any]
    backend: Backend


@dataclass
class CalibratedV3Model:
    """Frozen base score source plus a one-dimensional November score calibrator."""

    base_model: Any
    family: Family
    method: CalibrationMethod
    calibrator: Any | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([0, 1])

    def positive_scores(self, features: pd.DataFrame) -> np.ndarray:
        return _apply_calibrator(
            predict_positive(self.base_model, features, family=self.family),
            method=self.method,
            calibrator=self.calibrator,
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = self.positive_scores(features)
        return np.column_stack((1.0 - positive, positive))


@dataclass
class CalibratedV3Ensemble:
    """Probability blend of two frozen uncalibrated bases plus its own calibrator."""

    lightgbm_model: Any
    catboost_model: Any
    lightgbm_weight: float
    method: CalibrationMethod
    calibrator: Any | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([0, 1])

    def blended_scores(self, features: pd.DataFrame) -> np.ndarray:
        return blend_scores(
            predict_positive(self.lightgbm_model, features, family="lightgbm"),
            predict_positive(self.catboost_model, features, family="catboost"),
            lightgbm_weight=self.lightgbm_weight,
        )

    def positive_scores(self, features: pd.DataFrame) -> np.ndarray:
        return _apply_calibrator(
            self.blended_scores(features), method=self.method, calibrator=self.calibrator
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = self.positive_scores(features)
        return np.column_stack((1.0 - positive, positive))


def candidate_specs(
    protocol: dict[str, Any], *, family: Family, backend: Backend
) -> tuple[V3CandidateSpec, ...]:
    """Translate the frozen identity matrix, keeping backend out of candidate identity."""

    if family == "lightgbm" and backend != "CPU":
        raise V3ModelError("LightGBM screening and confirmation are CPU-only")
    carried = protocol["carried_forward_configurations"]
    bases = {str(row["id"]): row for row in carried["base_configurations"]}
    common = {
        "lightgbm": dict(carried["lightgbm_common_parameters"]),
        "catboost": dict(carried["catboost_common_parameters"]),
    }
    if common["lightgbm"].get("subsample_freq") != 1:
        raise V3ModelError("LightGBM row subsampling must remain activated with subsample_freq=1")

    result: list[V3CandidateSpec] = []
    for row in protocol["candidate_identities"]["identities"]:
        base = bases[str(row["base_configuration"])]
        if str(base["family"]) != family:
            continue
        identity = {name: value for name, value in base.items() if name not in {"id", "family"}}
        constructor = {**common[family], **identity}
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
        policy = str(row["weight_policy"])
        expected_id = f"{base['id']}-{WEIGHT_POLICY_SUFFIX[policy]}"
        if str(row["id"]) != expected_id:
            raise V3ModelError(f"candidate identity {row['id']} does not match {expected_id}")
        result.append(
            V3CandidateSpec(
                family=family,
                candidate_id=str(row["id"]),
                base_configuration=str(base["id"]),
                weight_policy=policy,
                identity_parameters={**identity, "weight_policy": policy},
                constructor_parameters=constructor,
                backend=backend,
            )
        )
    if len(result) != 4:
        raise V3ModelError(f"{family} must contribute exactly four v3 identities")
    return tuple(result)


def all_candidate_ids(protocol: dict[str, Any]) -> tuple[str, ...]:
    ids = tuple(
        spec.candidate_id
        for family, backend in (("lightgbm", "CPU"), ("catboost", "CPU"))
        for spec in candidate_specs(protocol, family=family, backend=backend)
    )
    if sorted(ids) != sorted(CANDIDATE_IDENTITY_IDS):
        raise V3ModelError("v3 candidate identity set drifted from the frozen protocol")
    return ids


def build_candidate(
    spec: V3CandidateSpec, *, classifier_type: Callable[..., Any] | None = None
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
        raise V3ModelError(f"{spec.candidate_id} constructor rejected frozen parameters") from error


def native_categorical_frame(features: pd.DataFrame, *, family: Family) -> pd.DataFrame:
    """Encode all eight native categorical columns for the family's own interface."""

    missing = set(CATEGORICAL_FEATURES) - set(features)
    if missing:
        raise V3ModelError(f"native categorical columns are missing: {sorted(missing)}")
    result = features.copy(deep=True)
    for column in STRING_CATEGORICAL_FEATURES:
        values = result[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise V3ModelError(f"native categorical column {column} contains invalid values")
        result[column] = values.astype("category" if family == "lightgbm" else str)
    for column in INTEGER_CATEGORICAL_FEATURES:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not values.mod(1).eq(0).all():
            raise V3ModelError(f"native categorical column {column} must contain integers")
        integers = values.astype(int)
        result[column] = (
            integers.astype("category") if family == "lightgbm" else integers.astype(str)
        )
    return result


def fit_candidate(
    model: Any,
    spec: V3CandidateSpec,
    features: pd.DataFrame,
    target: pd.Series,
    flight_date: pd.Series,
    *,
    fit_cutoff: str,
) -> tuple[Any, np.ndarray | None]:
    """Fit chronologically with native categories, frozen weights, and no outer eval channel."""

    dates = pd.to_datetime(flight_date)
    if not dates.is_monotonic_increasing:
        raise V3ModelError("v3 fit rows must be chronologically ordered")
    weights = fit_weights(dates, policy=spec.weight_policy, fit_cutoff=fit_cutoff)
    native = native_categorical_frame(features, family=spec.family)
    if spec.family == "lightgbm":
        model.fit(
            native,
            target,
            sample_weight=weights,
            categorical_feature=list(CATEGORICAL_FEATURES),
        )
    else:
        model.fit(native, target, sample_weight=weights, cat_features=list(CATEGORICAL_FEATURES))
    return model, weights


def predict_positive(model: Any, features: pd.DataFrame, *, family: Family) -> np.ndarray:
    native = native_categorical_frame(features, family=family)
    probabilities = np.asarray(model.predict_proba(native), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(features), 2):
        raise V3ModelError("model predict_proba must return one binary row per input")
    positive = probabilities[:, 1]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise V3ModelError("model probabilities must be finite values in [0, 1]")
    return positive


def blend_scores(
    lightgbm_scores: np.ndarray, catboost_scores: np.ndarray, *, lightgbm_weight: float
) -> np.ndarray:
    """Blend two uncalibrated base score vectors at a precommitted weight."""

    if not 0.0 <= lightgbm_weight <= 1.0:
        raise V3ModelError("ensemble weight must lie in [0, 1]")
    left = np.asarray(lightgbm_scores, dtype=float)
    right = np.asarray(catboost_scores, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise V3ModelError("ensemble inputs must be aligned non-empty score vectors")
    blended = lightgbm_weight * left + (1.0 - lightgbm_weight) * right
    if not np.isfinite(blended).all() or ((blended < 0) | (blended > 1)).any():
        raise V3ModelError("ensemble scores must be finite values in [0, 1]")
    return blended


def _fit_calibrator(
    scores: np.ndarray, target: pd.Series, *, method: CalibrationMethod
) -> Any | None:
    if method == "none":
        return None
    if method == "sigmoid":
        calibrator = LogisticRegression(random_state=42, solver="lbfgs")
        calibrator.fit(scores.reshape(-1, 1), target)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(scores, target)
        return calibrator
    raise V3ModelError(f"unsupported v3 calibration method: {method}")


def _apply_calibrator(
    scores: np.ndarray, *, method: CalibrationMethod, calibrator: Any | None
) -> np.ndarray:
    if method == "none":
        return scores
    if method == "sigmoid":
        return np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    if method == "isotonic":
        return np.asarray(calibrator.predict(scores), dtype=float)
    raise V3ModelError(f"unsupported v3 calibration method: {method}")


def build_calibration_variant(
    fitted_base: Any,
    *,
    family: Family,
    method: CalibrationMethod,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
) -> CalibratedV3Model:
    """Fit only a score calibrator while keeping the authoritative CPU base frozen."""

    before = predict_positive(fitted_base, calibration_features, family=family)
    calibrator = _fit_calibrator(before, calibration_target, method=method)
    after = predict_positive(fitted_base, calibration_features, family=family)
    if not np.array_equal(before, after):
        raise V3ModelError("calibration mutated the frozen CPU base")
    return CalibratedV3Model(fitted_base, family, method, calibrator)


def build_ensemble_variant(
    lightgbm_model: Any,
    catboost_model: Any,
    *,
    lightgbm_weight: float,
    method: CalibrationMethod,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
) -> CalibratedV3Ensemble:
    """Blend two frozen uncalibrated bases and calibrate the blend, fitting no new base model."""

    lightgbm_before = predict_positive(lightgbm_model, calibration_features, family="lightgbm")
    catboost_before = predict_positive(catboost_model, calibration_features, family="catboost")
    blended = blend_scores(lightgbm_before, catboost_before, lightgbm_weight=lightgbm_weight)
    calibrator = _fit_calibrator(blended, calibration_target, method=method)
    if not np.array_equal(
        lightgbm_before, predict_positive(lightgbm_model, calibration_features, family="lightgbm")
    ) or not np.array_equal(
        catboost_before, predict_positive(catboost_model, calibration_features, family="catboost")
    ):
        raise V3ModelError("ensemble calibration mutated a frozen CPU base")
    return CalibratedV3Ensemble(lightgbm_model, catboost_model, lightgbm_weight, method, calibrator)


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
                raise V3ModelError(f"constructor drifted for {spec.candidate_id} on {backend}")
            observed[f"{spec.candidate_id}:{backend}"] = dict(spec.constructor_parameters)
    if len(observed) != 12:
        raise V3ModelError("constructor contract must cover eight identities across backends")
    return observed


def calibration_variant_ids() -> tuple[str, ...]:
    return CALIBRATION_VARIANTS
