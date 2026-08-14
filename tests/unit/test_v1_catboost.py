from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from flight_delay.modeling import v1_catboost
from flight_delay.modeling.v1_catboost import (
    CATBOOST_CANDIDATE_IDS,
    V1CatBoostError,
    build_calibration_variant,
    build_catboost_candidate,
    candidate_specs,
    fit_catboost_base,
    require_catboost_version,
    validate_catboost_runtime_contract,
)
from flight_delay.modeling.v1_data import V1_CATEGORICAL_FEATURES, V1_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())


class _FakeClassifier:
    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.fit_arguments: tuple[Any, ...] | None = None

    def fit(self, *arguments: Any, **keywords: Any) -> _FakeClassifier:
        self.fit_arguments = (*arguments, keywords)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = np.linspace(0.2, 0.8, len(features))
        return np.column_stack((1 - scores, scores))


def _features(rows: int = 20) -> pd.DataFrame:
    payload: dict[str, Any] = {}
    for column in V1_FEATURES:
        if column in V1_CATEGORICAL_FEATURES:
            if column == "Reporting_Airline":
                payload[column] = ["UA" if index % 2 else "AA" for index in range(rows)]
            elif column == "Origin":
                payload[column] = ["DEN" if index % 2 else "SFO" for index in range(rows)]
            elif column == "Dest":
                payload[column] = ["SFO" if index % 2 else "DEN" for index in range(rows)]
            else:
                payload[column] = ["DEN-SFO" if index % 2 else "SFO-DEN" for index in range(rows)]
        else:
            payload[column] = np.arange(rows, dtype=float) + 1
    return pd.DataFrame(payload, columns=V1_FEATURES)


def test_exact_four_candidate_constructor_mapping() -> None:
    specs = candidate_specs(_protocol())
    assert tuple(item.candidate_id for item in specs) == CATBOOST_CANDIDATE_IDS
    assert len(specs) == 4
    for spec in specs:
        model = build_catboost_candidate(
            _protocol(), spec.candidate_id, classifier_type=_FakeClassifier
        )
        assert model.parameters == spec.parameters
        assert model.parameters["has_time"] is True
        assert model.parameters["allow_writing_files"] is False
        assert "early_stopping" not in model.parameters
        assert "class_weights" not in model.parameters
        assert "auto_class_weights" not in model.parameters
    with pytest.raises(V1CatBoostError, match="unauthorized"):
        build_catboost_candidate(_protocol(), "CB5", classifier_type=_FakeClassifier)


def test_candidate_spec_rejects_each_protocol_constructor_drift() -> None:
    early = copy.deepcopy(_protocol())
    early["catboost_search"]["common_parameters"]["early_stopping"] = "enabled"
    with pytest.raises(V1CatBoostError, match="early stopping"):
        candidate_specs(early)
    weighted = copy.deepcopy(_protocol())
    weighted["catboost_search"]["common_parameters"]["class_weights"] = [1, 2]
    with pytest.raises(V1CatBoostError, match="weighting"):
        candidate_specs(weighted)
    expanded = copy.deepcopy(_protocol())
    expanded["catboost_search"]["candidates"][0]["id"] = "CB5"
    with pytest.raises(V1CatBoostError, match="CB1-CB4"):
        candidate_specs(expanded)


def test_exact_catboost_version_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_catboost, "installed_catboost_version", lambda: None)
    with pytest.raises(V1CatBoostError, match="requires catboost==1.2.10"):
        require_catboost_version()
    monkeypatch.setattr(v1_catboost, "installed_catboost_version", lambda: "1.2.10")
    assert require_catboost_version() == "1.2.10"


def test_runtime_constructor_contract_is_validated_before_fit() -> None:
    validated = validate_catboost_runtime_contract(_protocol(), classifier_type=_FakeClassifier)
    assert tuple(validated) == CATBOOST_CANDIDATE_IDS

    class Rejecting:
        def __init__(self, **_parameters: Any) -> None:
            raise TypeError("rejected")

    with pytest.raises(V1CatBoostError, match="rejected CB1 parameters"):
        validate_catboost_runtime_contract(_protocol(), classifier_type=Rejecting)

    class Drifting(_FakeClassifier):
        def get_params(self) -> dict[str, Any]:
            return {}

    with pytest.raises(V1CatBoostError, match="constructor drifted"):
        validate_catboost_runtime_contract(_protocol(), classifier_type=Drifting)


def test_fit_is_chronological_and_uses_native_categorical_names() -> None:
    model = _FakeClassifier()
    features = _features(4)
    target = pd.Series([0, 1, 0, 1])
    dates = pd.Series(pd.date_range("2025-01-01", periods=4))
    fit_catboost_base(model, features, target, dates)
    assert model.fit_arguments is not None
    assert model.fit_arguments[-1] == {"cat_features": list(V1_CATEGORICAL_FEATURES)}
    assert "eval_set" not in model.fit_arguments[-1]
    with pytest.raises(V1CatBoostError, match="chronologically"):
        fit_catboost_base(model, features, target, dates.iloc[::-1])


def test_raw_calibration_variant_never_touches_calibration_labels() -> None:
    base = _FakeClassifier()

    class _ExplodingTarget:
        def __getattribute__(self, _name: str) -> Any:
            raise AssertionError("raw variant accessed calibration target")

    assert (
        build_calibration_variant(
            base,
            method="none",
            calibration_target=_ExplodingTarget(),  # type: ignore[arg-type]
        )
        is base
    )


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_uses_frozen_factory_without_mutating_base(method: str) -> None:
    base = _FakeClassifier()
    features = _features(4)
    target = pd.Series([0, 1, 0, 1])
    calls: list[tuple[Any, str]] = []

    def factory(model: Any, _features: Any, _target: Any, *, method: str) -> Any:
        calls.append((model, method))
        return _FakeClassifier()

    before = base.predict_proba(features)
    calibrated = build_calibration_variant(
        base,
        method=method,  # type: ignore[arg-type]
        calibration_features=features,
        calibration_target=target,
        calibrator_factory=factory,
    )
    assert calls == [(base, method)]
    np.testing.assert_array_equal(before, base.predict_proba(features))
    assert calibrated is not base


def test_calibration_rejects_invalid_missing_and_mutating_variants() -> None:
    base = _FakeClassifier()
    features = _features(4)
    target = pd.Series([0, 1, 0, 1])
    with pytest.raises(V1CatBoostError, match="unsupported"):
        build_calibration_variant(base, method="beta")  # type: ignore[arg-type]
    with pytest.raises(V1CatBoostError, match="require the locked"):
        build_calibration_variant(base, method="sigmoid")

    def mutating_factory(model: Any, *_args: Any, **_kwargs: Any) -> Any:
        model.predict_proba = lambda frame: np.column_stack(  # type: ignore[method-assign]
            (np.zeros(len(frame)), np.ones(len(frame)))
        )
        return _FakeClassifier()

    with pytest.raises(V1CatBoostError, match="mutated"):
        build_calibration_variant(
            base,
            method="sigmoid",
            calibration_features=features,
            calibration_target=target,
            calibrator_factory=mutating_factory,
        )


def test_tiny_real_catboost_fit_predict_smoke() -> None:
    pytest.importorskip("catboost")
    features = _features(20)
    target = pd.Series([0, 1] * 10)
    dates = pd.Series(pd.date_range("2025-01-01", periods=20))
    model = build_catboost_candidate(_protocol(), "CB1")
    fit_catboost_base(model, features, target, dates)
    probabilities = model.predict_proba(features)[:, 1]
    assert probabilities.shape == (20,)
    assert np.isfinite(probabilities).all()


def test_optional_dependency_is_absent_from_runtime_dockerfiles() -> None:
    project = (ROOT / "pyproject.toml").read_text()
    assert 'v1 = [\n    "catboost==1.2.10"' in project
    assert "catboost" not in project.split("dependencies = [", 1)[1].split("]", 1)[0]
    for path in (
        ROOT / "services/api/Dockerfile",
        ROOT / "services/user_ui/Dockerfile",
        ROOT / "services/monitor_ui/Dockerfile",
    ):
        source = path.read_text().casefold()
        assert "catboost" not in source
        assert "requirements-v1" not in source


def test_ci_installs_modeling_locks_and_probes_every_runtime_image() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert (
        '-c requirements.lock -c requirements-v1.lock -c requirements-v2.lock ".[dev,v1,v2]"'
        in workflow
    )
    assert 'find_spec("catboost") is None' in workflow
    assert 'find_spec("lightgbm") is None' in workflow
    for image in (
        "flight-delay-api:scaffold",
        "flight-delay-user-ui:scaffold",
        "flight-delay-monitor-ui:scaffold",
    ):
        assert image in workflow
