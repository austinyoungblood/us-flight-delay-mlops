from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v2.data import PreparedV2Data
from flight_delay.modeling.v2.features import (
    build_historical_state,
    transform_training_rows,
    transform_with_state,
)
from flight_delay.modeling.v2.models import CandidateSpec, fit_candidate
from flight_delay.modeling.v2.tracking import NullTracker
from flight_delay.modeling.v2.workflow import (
    V2WorkflowError,
    bundle_evidence,
    run_candidate_stage,
    run_refit_and_november,
    run_screening_and_cpu_confirmation,
    sanitized_workflow_result,
)

FIT_EVENTS: list[tuple[str, str, str]] = []


class SyntheticClassifier:
    def __init__(self, spec: CandidateSpec) -> None:
        self.spec = spec
        self.classes_ = np.asarray([0, 1])

    def fit(self, features: pd.DataFrame, target: pd.Series, **_kwargs: Any) -> SyntheticClassifier:
        FIT_EVENTS.append((self.spec.family, self.spec.backend, self.spec.candidate_id))
        self.fitted_rows = len(features)
        self.target_mean = float(target.mean())
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.where(pd.to_numeric(features["DayofMonth"]).isin((9, 25)), 0.98, 0.02)
        return np.column_stack((1.0 - positive, positive))


def _builder(spec: CandidateSpec) -> SyntheticClassifier:
    return SyntheticClassifier(spec)


def _prepared(frame: pd.DataFrame) -> PreparedV2Data:
    dates = pd.to_datetime(frame["flight_date"])
    train = frame.loc[dates.dt.month.le(10)].copy()
    model_rows = train.loc[pd.to_datetime(train["flight_date"]).dt.month.ge(2)].copy()
    transformed = transform_training_rows(train, model_rows)
    state = build_historical_state(train, as_of="2025-10-31")
    november = frame.loc[dates.dt.month.eq(11)].copy()
    november_dates = pd.to_datetime(november["flight_date"])
    calibration = november.loc[november_dates.dt.day.lt(16)].copy()
    selection = november.loc[november_dates.dt.day.ge(16)].copy()
    return PreparedV2Data(
        search=transformed,
        full_refit=transformed,
        calibration_features=transform_with_state(calibration, state),
        calibration_target=calibration["target"],
        calibration_date=calibration["flight_date"],
        selection_features=transform_with_state(selection, state),
        selection_target=selection["target"],
        selection_date=selection["flight_date"],
        november_state=state,
        raw_train=train,
        raw_november=november,
        lineage={"november_state_sha256": state.sha256},
    )


def test_gpu_screening_then_authoritative_cpu_confirmation(
    v2_protocol: dict[str, Any], synthetic_v2_frame: pd.DataFrame
) -> None:
    FIT_EVENTS.clear()
    prepared = _prepared(synthetic_v2_frame)
    tracker = NullTracker()
    result = run_screening_and_cpu_confirmation(
        protocol=v2_protocol,
        transformed=prepared.search,
        tracker=tracker,
        metadata={"group": "synthetic"},
        builder=_builder,
        fitter=fit_candidate,
    )
    assert len(result["screening"]) == 28
    assert len(result["cpu_confirmation"]) == 8
    assert len(result["screening_cpu_differences"]) == 8
    assert [row["candidate_id"] for row in result["advanced_to_refit"]] == [
        "LGBM01",
        "LGBM02",
        "CB01",
        "CB02",
    ]
    assert FIT_EVENTS[:64] == [
        ("lightgbm", "CPU", f"LGBM{candidate:02d}")
        for candidate in range(1, 17)
        for _fold in range(4)
    ]
    assert FIT_EVENTS[64:112] == [
        ("catboost", "GPU", f"CB{candidate:02d}")
        for candidate in range(1, 13)
        for _fold in range(4)
    ]
    assert all(row["backend"] == "CPU" for row in result["cpu_confirmation"])
    assert all(run.metadata["group"] == "synthetic" for run in tracker.runs)


def test_full_synthetic_workflow_creates_twelve_finalists_and_winner(
    v2_protocol: dict[str, Any], synthetic_v2_frame: pd.DataFrame
) -> None:
    prepared = _prepared(synthetic_v2_frame)
    tracker = NullTracker()
    search = run_screening_and_cpu_confirmation(
        protocol=v2_protocol,
        transformed=prepared.search,
        tracker=tracker,
        metadata={"group": "synthetic"},
        builder=_builder,
        fitter=fit_candidate,
    )
    result = run_refit_and_november(
        prepared=prepared,
        protocol=v2_protocol,
        advanced=search["advanced_to_refit"],
        tracker=tracker,
        metadata={"group": "synthetic"},
        r3_reconstruction_passed=True,
        builder=_builder,
        fitter=fit_candidate,
    )
    assert result["decision"] == "winner"
    assert len(result["finalists"]) == 12
    assert result["winner"]["passed"] is True
    assert result["production_remains"] == "v0"
    assert result["stopped_before_december"] is True
    assert {row["calibration_method"] for row in result["finalists"]} == {
        "none",
        "sigmoid",
        "isotonic",
    }
    sanitized = sanitized_workflow_result(result)
    assert isinstance(sanitized["winner"], str)
    assert all("model" not in row for row in sanitized["finalists"])

    with pytest.raises(V2WorkflowError, match="R3 reconstruction"):
        run_refit_and_november(
            prepared=prepared,
            protocol=v2_protocol,
            advanced=search["advanced_to_refit"],
            tracker=tracker,
            metadata={"group": "synthetic"},
            r3_reconstruction_passed=False,
            builder=_builder,
            fitter=fit_candidate,
        )


def test_candidate_subset_and_bundle_guards(
    v2_protocol: dict[str, Any], synthetic_v2_frame: pd.DataFrame
) -> None:
    prepared = _prepared(synthetic_v2_frame)
    with pytest.raises(V2WorkflowError, match="unauthorized"):
        run_candidate_stage(
            protocol=v2_protocol,
            transformed=prepared.search,
            family="catboost",
            backend="CPU",
            candidate_ids=("CB99",),
            tracker=NullTracker(),
            metadata={"group": "synthetic"},
            builder=_builder,
            fitter=fit_candidate,
        )
    spec_row = {
        "candidate_id": "CB01",
        "family": "catboost",
        "backend": "CPU",
    }
    spec = next(
        item
        for item in __import__(
            "flight_delay.modeling.v2.models", fromlist=["candidate_specs"]
        ).candidate_specs(v2_protocol, family="catboost", backend="CPU")
        if item.candidate_id == spec_row["candidate_id"]
    )
    model = _builder(spec)
    fit_candidate(
        model,
        spec,
        prepared.full_refit.features,
        prepared.full_refit.target,
        prepared.full_refit.flight_date,
    )
    evidence = bundle_evidence(model, prepared.selection_features, prepared.november_state)
    assert evidence["serialized_bundle_bytes"] > 0
    assert evidence["historical_state_sha256"] == prepared.november_state.sha256
    assert evidence["serialization_load_inference_passed"] is True
