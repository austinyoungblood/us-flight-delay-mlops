"""Candidate identity construction, native categoricals, weighted fits, and ensembles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v3.models import (
    V3ModelError,
    all_candidate_ids,
    blend_scores,
    build_calibration_variant,
    build_candidate,
    build_ensemble_variant,
    candidate_specs,
    fit_candidate,
    native_categorical_frame,
    predict_positive,
    validate_constructor_contract,
)
from flight_delay.modeling.v3.protocol import (
    CANDIDATE_IDENTITY_IDS,
    CATEGORICAL_FEATURES,
    INTEGER_CATEGORICAL_FEATURES,
    STRING_CATEGORICAL_FEATURES,
    V3_FEATURES,
)


class FakeClassifier:
    """Deterministic stand-in so identity and weighting are tested without a real runtime."""

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.fitted_rows = 0
        self.sample_weight: np.ndarray | None = None
        self.categorical: list[str] | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self.parameters)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        sample_weight: Any = None,
        categorical_feature: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> FakeClassifier:
        self.fitted_rows = len(features)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        self.categorical = categorical_feature or cat_features
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.linspace(0.05, 0.95, len(features))
        return np.column_stack((1.0 - positive, positive))


def frame(rows: int = 40) -> pd.DataFrame:
    values: dict[str, Any] = {}
    for name in V3_FEATURES:
        if name in STRING_CATEGORICAL_FEATURES:
            values[name] = ["AA" if index % 2 else "DL" for index in range(rows)]
        elif name in INTEGER_CATEGORICAL_FEATURES:
            values[name] = [index % 12 + 1 for index in range(rows)]
        else:
            values[name] = np.linspace(0.1, 0.9, rows)
    result = pd.DataFrame(values, columns=list(V3_FEATURES))
    result["route"] = ["DEN-SFO" if index % 2 else "SFO-DEN" for index in range(rows)]
    return result


@pytest.fixture
def fit_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = frame()
    target = pd.Series([index % 2 for index in range(len(features))])
    flight_date = pd.Series(pd.date_range("2025-06-01", periods=len(features), freq="D"))
    return features, target, flight_date


def test_eight_identities_split_evenly_between_the_two_families(v3_protocol: dict) -> None:
    assert sorted(all_candidate_ids(v3_protocol)) == sorted(CANDIDATE_IDENTITY_IDS)
    lightgbm = candidate_specs(v3_protocol, family="lightgbm", backend="CPU")
    catboost = candidate_specs(v3_protocol, family="catboost", backend="CPU")
    assert len(lightgbm) == 4
    assert len(catboost) == 4
    assert {spec.weight_policy for spec in lightgbm} == {"UNIFORM", "EXPONENTIAL_120D"}


def test_weight_policy_is_in_identity_but_not_in_constructor(v3_protocol: dict) -> None:
    specs = {
        spec.candidate_id: spec
        for spec in candidate_specs(v3_protocol, family="lightgbm", backend="CPU")
    }
    uniform = specs["LGBM12-UNIFORM"]
    exponential = specs["LGBM12-EXP120"]
    assert uniform.base_configuration == exponential.base_configuration == "LGBM12"
    assert uniform.identity_parameters["weight_policy"] == "UNIFORM"
    assert exponential.identity_parameters["weight_policy"] == "EXPONENTIAL_120D"
    # Same hyperparameters; only the weight policy separates them.
    assert uniform.constructor_parameters == exponential.constructor_parameters


def test_backend_never_enters_candidate_identity(v3_protocol: dict) -> None:
    gpu = {
        spec.candidate_id: spec
        for spec in candidate_specs(v3_protocol, family="catboost", backend="GPU")
    }
    cpu = {
        spec.candidate_id: spec
        for spec in candidate_specs(v3_protocol, family="catboost", backend="CPU")
    }
    for candidate_id, spec in gpu.items():
        assert spec.identity_parameters == cpu[candidate_id].identity_parameters
        assert spec.constructor_parameters["task_type"] == "GPU"
        assert cpu[candidate_id].constructor_parameters["task_type"] == "CPU"


def test_lightgbm_refuses_a_gpu_backend(v3_protocol: dict) -> None:
    with pytest.raises(V3ModelError, match="CPU-only"):
        candidate_specs(v3_protocol, family="lightgbm", backend="GPU")


def test_constructor_contract_covers_every_identity_and_backend(v3_protocol: dict) -> None:
    observed = validate_constructor_contract(v3_protocol, classifier_type=FakeClassifier)
    assert len(observed) == 12
    assert all(key.split(":")[0] in CANDIDATE_IDENTITY_IDS for key in observed)


def test_all_eight_native_categoricals_are_encoded() -> None:
    features = frame()
    lightgbm = native_categorical_frame(features, family="lightgbm")
    catboost = native_categorical_frame(features, family="catboost")
    for column in CATEGORICAL_FEATURES:
        assert str(lightgbm[column].dtype) == "category"
        assert catboost[column].dtype == object
    # The source frame is never mutated in place.
    assert str(features["Month"].dtype) != "category"


def test_missing_or_invalid_categoricals_are_refused() -> None:
    features = frame()
    with pytest.raises(V3ModelError, match="missing"):
        native_categorical_frame(features.drop(columns=["route"]), family="lightgbm")
    broken = frame()
    broken["Month"] = broken["Month"].astype(float)
    broken.loc[0, "Month"] = 1.5
    with pytest.raises(V3ModelError, match="integers"):
        native_categorical_frame(broken, family="lightgbm")
    blank = frame()
    blank.loc[0, "Origin"] = " "
    with pytest.raises(V3ModelError, match="invalid values"):
        native_categorical_frame(blank, family="lightgbm")


def test_uniform_fit_passes_no_sample_weight(
    v3_protocol: dict, fit_inputs: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    features, target, flight_date = fit_inputs
    spec = next(
        s
        for s in candidate_specs(v3_protocol, family="lightgbm", backend="CPU")
        if s.candidate_id == "LGBM12-UNIFORM"
    )
    model = build_candidate(spec, classifier_type=FakeClassifier)
    fitted, weights = fit_candidate(
        model, spec, features, target, flight_date, fit_cutoff="2025-10-31"
    )
    assert weights is None
    assert fitted.sample_weight is None
    assert fitted.categorical == list(CATEGORICAL_FEATURES)


def test_exponential_fit_passes_normalized_weights(
    v3_protocol: dict, fit_inputs: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    features, target, flight_date = fit_inputs
    spec = next(
        s
        for s in candidate_specs(v3_protocol, family="catboost", backend="CPU")
        if s.candidate_id == "CB04-EXP120"
    )
    model = build_candidate(spec, classifier_type=FakeClassifier)
    fitted, weights = fit_candidate(
        model, spec, features, target, flight_date, fit_cutoff="2025-10-31"
    )
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-12)
    assert fitted.sample_weight is not None
    assert len(fitted.sample_weight) == len(features)


def test_fit_requires_chronological_rows(
    v3_protocol: dict, fit_inputs: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    features, target, flight_date = fit_inputs
    spec = candidate_specs(v3_protocol, family="lightgbm", backend="CPU")[0]
    model = build_candidate(spec, classifier_type=FakeClassifier)
    with pytest.raises(V3ModelError, match="chronologically"):
        fit_candidate(model, spec, features, target, flight_date[::-1], fit_cutoff="2025-10-31")


def test_predict_positive_validates_its_output() -> None:
    class Broken:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.full((len(features), 2), 2.0)

    with pytest.raises(V3ModelError, match=r"\[0, 1\]"):
        predict_positive(Broken(), frame(4), family="lightgbm")


def test_blend_is_a_convex_combination() -> None:
    left = np.array([0.2, 0.8])
    right = np.array([0.6, 0.4])
    assert blend_scores(left, right, lightgbm_weight=0.25) == pytest.approx([0.5, 0.5])
    assert blend_scores(left, right, lightgbm_weight=1.0) == pytest.approx(left)
    assert blend_scores(left, right, lightgbm_weight=0.0) == pytest.approx(right)


def test_blend_rejects_bad_weights_and_shapes() -> None:
    left = np.array([0.2, 0.8])
    with pytest.raises(V3ModelError, match=r"\[0, 1\]"):
        blend_scores(left, left, lightgbm_weight=1.5)
    with pytest.raises(V3ModelError, match="aligned"):
        blend_scores(left, np.array([0.1]), lightgbm_weight=0.5)


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_calibration_variants_leave_the_base_frozen(method: str) -> None:
    features = frame()
    target = pd.Series([index % 2 for index in range(len(features))])
    base = FakeClassifier()
    before = predict_positive(base, features, family="lightgbm")
    model = build_calibration_variant(
        base,
        family="lightgbm",
        method=method,  # type: ignore[arg-type]
        calibration_features=features,
        calibration_target=target,
    )
    scores = model.predict_proba(features)[:, 1]
    assert scores.shape == (len(features),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert predict_positive(base, features, family="lightgbm") == pytest.approx(before)
    if method == "none":
        assert scores == pytest.approx(before)


def test_unsupported_calibration_method_is_refused() -> None:
    features = frame()
    target = pd.Series([index % 2 for index in range(len(features))])
    with pytest.raises(V3ModelError, match="unsupported"):
        build_calibration_variant(
            FakeClassifier(),
            family="lightgbm",
            method="platt",  # type: ignore[arg-type]
            calibration_features=features,
            calibration_target=target,
        )


@pytest.mark.parametrize("weight", [0.25, 0.50, 0.75])
@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_ensembles_need_no_additional_base_fit(weight: float, method: str) -> None:
    features = frame()
    target = pd.Series([index % 2 for index in range(len(features))])
    lightgbm = FakeClassifier()
    catboost = FakeClassifier()
    model = build_ensemble_variant(
        lightgbm,
        catboost,
        lightgbm_weight=weight,
        method=method,  # type: ignore[arg-type]
        calibration_features=features,
        calibration_target=target,
    )
    scores = model.predict_proba(features)[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()
    # Neither base was refitted by ensembling or calibrating.
    assert lightgbm.fitted_rows == 0
    assert catboost.fitted_rows == 0
