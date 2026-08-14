from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v2.models import (
    CalibratedV2Model,
    V2ModelError,
    build_calibration_variant,
    build_candidate,
    candidate_specs,
    fit_candidate,
    native_categorical_frame,
    predict_positive,
    require_versions,
    validate_constructor_contract,
)
from flight_delay.modeling.v2.protocol import CATEGORICAL_FEATURES, V2_FEATURES


class FakeClassifier:
    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.fit_kwargs: dict[str, Any] = {}
        self.classes_ = np.asarray([0, 1])

    def get_params(self) -> dict[str, Any]:
        return dict(self.parameters)

    def fit(self, features: pd.DataFrame, target: pd.Series, **kwargs: Any) -> FakeClassifier:
        self.fit_kwargs = kwargs
        self.fit_dtypes = features.dtypes.astype(str).to_dict()
        self.target = target.copy()
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.where(pd.to_numeric(features["DayofMonth"]).isin((9, 25)), 0.95, 0.05)
        return np.column_stack((1.0 - positive, positive))


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, V2_FEATURES[:20]].copy()
    for name in V2_FEATURES[20:]:
        result[name] = 0.5
    return result.loc[:, V2_FEATURES]


def test_exact_model_specs_and_backend_separation(v2_protocol: dict[str, Any]) -> None:
    lightgbm = candidate_specs(v2_protocol, family="lightgbm", backend="CPU")
    gpu = candidate_specs(v2_protocol, family="catboost", backend="GPU")
    cpu = candidate_specs(v2_protocol, family="catboost", backend="CPU")
    assert len(lightgbm) == 16
    assert len(gpu) == len(cpu) == 12
    assert [spec.identity_parameters for spec in gpu] == [spec.identity_parameters for spec in cpu]
    assert all(spec.constructor_parameters["task_type"] == "GPU" for spec in gpu)
    assert all(spec.constructor_parameters["devices"] == "0" for spec in gpu)
    assert all(spec.constructor_parameters["task_type"] == "CPU" for spec in cpu)
    assert all("devices" not in spec.constructor_parameters for spec in cpu)
    assert all(spec.constructor_parameters["n_jobs"] == 20 for spec in lightgbm)
    with pytest.raises(V2ModelError, match="CPU-only"):
        candidate_specs(v2_protocol, family="lightgbm", backend="GPU")


def test_constructor_contract_uses_all_frozen_candidates(v2_protocol: dict[str, Any]) -> None:
    observed = validate_constructor_contract(v2_protocol, classifier_type=FakeClassifier)
    assert len(observed) == 40
    assert "LGBM01:CPU" in observed
    assert "CB01:GPU" in observed
    assert "CB01:CPU" in observed
    model = build_candidate(
        candidate_specs(v2_protocol, family="catboost", backend="GPU")[0],
        classifier_type=FakeClassifier,
    )
    assert model.get_params()["task_type"] == "GPU"


def test_native_categorical_fit_and_prediction(
    v2_protocol: dict[str, Any], synthetic_v2_frame: pd.DataFrame
) -> None:
    features = _features(synthetic_v2_frame.iloc[:8])
    target = synthetic_v2_frame.iloc[:8]["target"]
    dates = synthetic_v2_frame.iloc[:8]["flight_date"]
    for family in ("lightgbm", "catboost"):
        spec = candidate_specs(v2_protocol, family=family, backend="CPU")[0]
        model = FakeClassifier(**spec.constructor_parameters)
        fitted = fit_candidate(model, spec, features, target, dates)
        keyword = "categorical_feature" if family == "lightgbm" else "cat_features"
        assert fitted.fit_kwargs[keyword] == list(CATEGORICAL_FEATURES)
        if family == "lightgbm":
            assert all(fitted.fit_dtypes[name] == "category" for name in CATEGORICAL_FEATURES)
        scores = predict_positive(fitted, features, family=family)
        assert scores.shape == (8,)
        assert set(scores) == {0.05, 0.95}


def test_model_input_and_probability_guards(
    v2_protocol: dict[str, Any], synthetic_v2_frame: pd.DataFrame
) -> None:
    features = _features(synthetic_v2_frame.iloc[:4])
    with pytest.raises(V2ModelError, match="missing"):
        native_categorical_frame(features.drop(columns="Origin"), family="catboost")
    invalid = features.copy()
    invalid.loc[invalid.index[0], "Origin"] = ""
    with pytest.raises(V2ModelError, match="invalid"):
        native_categorical_frame(invalid, family="lightgbm")
    spec = candidate_specs(v2_protocol, family="catboost", backend="CPU")[0]
    descending = synthetic_v2_frame.iloc[:4]["flight_date"].sort_values(ascending=False)
    with pytest.raises(V2ModelError, match="chronologically"):
        fit_candidate(
            FakeClassifier(), spec, features, synthetic_v2_frame.iloc[:4]["target"], descending
        )

    class BadProbabilities(FakeClassifier):
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.ones((len(features), 1))

    with pytest.raises(V2ModelError, match="binary row"):
        predict_positive(BadProbabilities(), features, family="catboost")


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_calibration_variants_freeze_the_base(
    synthetic_v2_frame: pd.DataFrame, method: str
) -> None:
    features = _features(synthetic_v2_frame.iloc[:8])
    target = synthetic_v2_frame.iloc[:8]["target"]
    base = FakeClassifier()
    before = base.predict_proba(features).copy()
    calibrated = build_calibration_variant(
        base,
        family="catboost",
        method=method,
        calibration_features=features,
        calibration_target=target,
    )
    assert isinstance(calibrated, CalibratedV2Model)
    assert calibrated.predict_proba(features).shape == (8, 2)
    np.testing.assert_array_equal(base.predict_proba(features), before)


def test_version_guard_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flight_delay.modeling.v2.models.installed_version",
        lambda name: "4.7.0" if name == "lightgbm" else "1.2.10",
    )
    assert require_versions() == {"lightgbm": "4.7.0", "catboost": "1.2.10"}
    monkeypatch.setattr("flight_delay.modeling.v2.models.installed_version", lambda _name: None)
    with pytest.raises(V2ModelError, match="exact modeling versions"):
        require_versions()
