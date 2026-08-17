"""End-to-end v3 staging on synthetic data: no real model runtime is imported."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v3.data import PreparedV3Data
from flight_delay.modeling.v3.features import (
    build_v3_historical_state,
    transform_v3_training_rows,
    transform_with_v3_state,
)
from flight_delay.modeling.v3.protocol import V3_FEATURES
from flight_delay.modeling.v3.tracking import NullTracker
from flight_delay.modeling.v3.workflow import (
    V3WorkflowError,
    run_candidate_stage,
    run_refit_and_november,
    run_screening_and_cpu_confirmation,
    sanitized_workflow_result,
)
from tests.conftest import make_v3_frame


class SeededModel:
    """A deterministic scorer whose separation varies by identity, so ranking is exercised."""

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.strength = 0.5
        self.fitted_rows = 0
        self.sample_weight: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self.parameters)

    def fit(self, features: pd.DataFrame, target: pd.Series, sample_weight: Any = None, **_: Any):
        self.fitted_rows = len(features)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        self.target_mean = float(pd.Series(target).mean())
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        base = np.asarray(features["prior_route_delay_rate"], dtype=float)
        jitter = np.linspace(0.0, 0.2, len(features))
        positive = np.clip(self.strength * base + (1 - self.strength) * jitter, 0.01, 0.99)
        return np.column_stack((1.0 - positive, positive))


@pytest.fixture(scope="module")
def prepared() -> PreparedV3Data:
    history = make_v3_frame(start="2024-01-01", end="2025-10-31")
    november = make_v3_frame(start="2025-11-01", end="2025-11-30")
    model_rows = history.loc[pd.to_datetime(history["flight_date"]).ge("2024-02-01")].copy()
    fold_rows = pd.concat([model_rows, november], ignore_index=True).sort_values(
        "flight_date", kind="stable"
    )
    search = transform_v3_training_rows(history, fold_rows)
    full_refit = transform_v3_training_rows(history, model_rows)
    state = build_v3_historical_state(history, as_of="2025-10-31")
    dates = pd.to_datetime(november["flight_date"])
    calibration = november.loc[dates.lt("2025-11-16")]
    selection = november.loc[dates.ge("2025-11-16")]
    return PreparedV3Data(
        search=search,
        full_refit=full_refit,
        calibration_features=transform_with_v3_state(calibration, state),
        calibration_target=calibration["target"].astype(int),
        calibration_date=dates.loc[calibration.index],
        selection_features=transform_with_v3_state(selection, state),
        selection_target=selection["target"].astype(int),
        selection_date=dates.loc[selection.index],
        november_state=state,
        raw_history=history,
        raw_november=november,
        lineage={"november_state_sha256": state.sha256, "december_decoded": False},
    )


def builder(spec: Any) -> SeededModel:
    model = SeededModel(**spec.constructor_parameters)
    # Give EXP120 identities a slight edge so advancement is deterministic and observable.
    model.strength = 0.9 if spec.weight_policy == "EXPONENTIAL_120D" else 0.6
    return model


def fitter(model: Any, spec: Any, features, target, flight_date, *, fit_cutoff: str):
    from flight_delay.modeling.v3.weighting import fit_weights

    weights = fit_weights(
        pd.to_datetime(flight_date), policy=spec.weight_policy, fit_cutoff=fit_cutoff
    )
    model.fit(features, target, sample_weight=weights)
    return model, weights


def test_search_matrix_covers_all_four_folds(prepared: PreparedV3Data) -> None:
    dates = pd.to_datetime(prepared.search.flight_date)
    assert dates.min() == pd.Timestamp("2024-02-01")
    assert dates.max() == pd.Timestamp("2025-11-30")
    assert tuple(prepared.search.features.columns) == V3_FEATURES


def test_full_refit_matrix_stops_before_november(prepared: PreparedV3Data) -> None:
    dates = pd.to_datetime(prepared.full_refit.flight_date)
    assert dates.max() < pd.Timestamp("2025-11-01")
    assert dates.min() == pd.Timestamp("2024-02-01")


def test_candidate_stage_runs_four_folds_and_records_runtime(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    tracker = NullTracker()
    rows = run_candidate_stage(
        protocol=v3_protocol,
        transformed=prepared.search,
        family="lightgbm",
        backend="CPU",
        candidate_ids=None,
        tracker=tracker,
        metadata={"group": "test"},
        builder=builder,
        fitter=fitter,
    )
    assert len(rows) == 4
    for row in rows:
        assert [fold["fold_id"] for fold in row["folds"]] == [
            "FOLD_1",
            "FOLD_2",
            "FOLD_3",
            "FOLD_4",
        ]
        assert all(fold["stage_runtime_seconds"] >= 0 for fold in row["folds"])
        assert row["candidate_runtime_seconds"] >= 0
        assert set(row["weight_summary"]) == {"FOLD_1", "FOLD_2", "FOLD_3", "FOLD_4"}


def test_no_fold_ever_fits_on_november(prepared: PreparedV3Data, v3_protocol: dict) -> None:
    dates = pd.to_datetime(prepared.search.flight_date).dt.normalize()
    for fold in v3_protocol["rolling_origin"]["folds"]:
        fit_mask = dates.ge(fold["fit_start"]) & dates.lt(fold["fit_end_exclusive"])
        assert dates.loc[fit_mask].max() < pd.Timestamp("2025-11-01")
    last = v3_protocol["rolling_origin"]["folds"][-1]
    evaluation = dates.ge(last["evaluation_start"]) & dates.lt(last["evaluation_end_exclusive"])
    assert evaluation.any()


def test_uniform_and_exponential_identities_fit_differently(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    captured: dict[str, Any] = {}

    def recording_fitter(model, spec, features, target, flight_date, *, fit_cutoff):
        result = fitter(model, spec, features, target, flight_date, fit_cutoff=fit_cutoff)
        captured[spec.candidate_id] = result[1]
        return result

    run_candidate_stage(
        protocol=v3_protocol,
        transformed=prepared.search,
        family="lightgbm",
        backend="CPU",
        candidate_ids=("LGBM12-UNIFORM", "LGBM12-EXP120"),
        tracker=NullTracker(),
        metadata={"group": "test"},
        builder=builder,
        fitter=recording_fitter,
    )
    assert captured["LGBM12-UNIFORM"] is None
    assert captured["LGBM12-EXP120"] is not None
    assert float(captured["LGBM12-EXP120"].mean()) == pytest.approx(1.0, abs=1e-9)


def test_unauthorized_candidate_subset_is_refused(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    with pytest.raises(V3WorkflowError, match="unauthorized"):
        run_candidate_stage(
            protocol=v3_protocol,
            transformed=prepared.search,
            family="lightgbm",
            backend="CPU",
            candidate_ids=("LGBM99-UNIFORM",),
            tracker=NullTracker(),
            metadata={"group": "test"},
            builder=builder,
            fitter=fitter,
        )


def test_screening_advances_eight_to_four_to_two(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    result = run_screening_and_cpu_confirmation(
        protocol=v3_protocol,
        transformed=prepared.search,
        tracker=NullTracker(),
        metadata={"group": "test"},
        builder=builder,
        fitter=fitter,
    )
    assert len(result["screening"]) == 8
    assert len(result["cpu_confirmation"]) == 4
    assert len(result["advanced_to_refit"]) == 2
    assert {row["family"] for row in result["advanced_to_refit"]} == {"lightgbm", "catboost"}
    assert len(result["screening_cpu_differences"]) == 4
    # CatBoost screens on GPU but is confirmed and advanced on CPU only.
    screened = {row["candidate_id"]: row for row in result["screening"]}
    assert screened["CB04-UNIFORM"]["backend"] == "GPU"
    assert all(row["backend"] == "CPU" for row in result["cpu_confirmation"])


def test_refit_builds_six_base_and_nine_ensemble_finalists(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    advanced = [
        {"family": "lightgbm", "candidate_id": "LGBM12-EXP120"},
        {"family": "catboost", "candidate_id": "CB04-EXP120"},
    ]
    result = run_refit_and_november(
        prepared=prepared,
        protocol=v3_protocol,
        advanced=advanced,
        tracker=NullTracker(),
        metadata={"group": "test"},
        r3_reconstruction_passed=True,
        builder=builder,
        fitter=fitter,
    )
    assert len(result["finalists"]) == 15
    kinds = [row["kind"] for row in result["finalists"]]
    assert kinds.count("base") == 6
    assert kinds.count("ensemble") == 9
    ensembles = {row["finalist_id"] for row in result["finalists"] if row["kind"] == "ensemble"}
    assert ensembles == {
        f"{name}-{method}"
        for name in ("ENS25", "ENS50", "ENS75")
        for method in ("none", "sigmoid", "isotonic")
    }
    assert result["decision"] in {"winner", "governed_stop"}
    assert result["production_remains"] == "v0"
    assert result["stopped_before_december"] is True
    assert set(result["base_refits"]) == {"lightgbm", "catboost"}


def test_refit_requires_one_candidate_per_family(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    with pytest.raises(V3WorkflowError, match="one candidate per family"):
        run_refit_and_november(
            prepared=prepared,
            protocol=v3_protocol,
            advanced=[{"family": "lightgbm", "candidate_id": "LGBM12-EXP120"}],
            tracker=NullTracker(),
            metadata={"group": "test"},
            r3_reconstruction_passed=True,
            builder=builder,
            fitter=fitter,
        )


def test_refit_refuses_without_r3_reconstruction(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    with pytest.raises(V3WorkflowError, match="R3 reconstruction"):
        run_refit_and_november(
            prepared=prepared,
            protocol=v3_protocol,
            advanced=[
                {"family": "lightgbm", "candidate_id": "LGBM12-EXP120"},
                {"family": "catboost", "candidate_id": "CB04-EXP120"},
            ],
            tracker=NullTracker(),
            metadata={"group": "test"},
            r3_reconstruction_passed=False,
            builder=builder,
            fitter=fitter,
        )


def test_sanitized_result_drops_models_and_is_json_safe(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    import json

    result = run_refit_and_november(
        prepared=prepared,
        protocol=v3_protocol,
        advanced=[
            {"family": "lightgbm", "candidate_id": "LGBM12-EXP120"},
            {"family": "catboost", "candidate_id": "CB04-EXP120"},
        ],
        tracker=NullTracker(),
        metadata={"group": "test"},
        r3_reconstruction_passed=True,
        builder=builder,
        fitter=fitter,
    )
    sanitized = sanitized_workflow_result(result)
    assert all("model" not in row for row in sanitized["finalists"])
    assert json.dumps(sanitized, default=float)


def test_seasonal_prior_year_check_requires_real_prior_year_state() -> None:
    from flight_delay.modeling.v3.workflow import seasonal_prior_year_check

    class Fake:
        def __init__(self, ledger):
            self.same_calendar_month_max_year = ledger

    assert seasonal_prior_year_check(Fake({11: 2024})) is True
    # Same-year contribution must fail.
    assert seasonal_prior_year_check(Fake({11: 2025})) is False
    # An absent November entry must fail rather than pass vacuously.
    assert seasonal_prior_year_check(Fake({10: 2025})) is False


def test_weight_policy_gate_reflects_the_actual_refit_weights(
    prepared: PreparedV3Data, v3_protocol: dict
) -> None:
    result = run_refit_and_november(
        prepared=prepared,
        protocol=v3_protocol,
        advanced=[
            {"family": "lightgbm", "candidate_id": "LGBM12-EXP120"},
            {"family": "catboost", "candidate_id": "CB04-EXP120"},
        ],
        tracker=NullTracker(),
        metadata={"group": "test"},
        r3_reconstruction_passed=True,
        builder=builder,
        fitter=fitter,
    )
    for base in result["base_refits"].values():
        assert base["weight_summary"]["normalized_to_mean_one"] is True


def test_governance_flags_track_real_evidence(prepared: PreparedV3Data) -> None:
    from flight_delay.modeling.v3.workflow import _governance

    bundle = {
        "historical_state_integrity_passed": True,
        "serialization_load_inference_passed": True,
    }
    passing = _governance(
        prepared=prepared,
        bundle=bundle,
        r3_reconstruction_passed=True,
        weight_policies_normalized=True,
    )
    assert passing["seasonal_prior_year_check_passed"] is True
    assert passing["weight_policy_check_passed"] is True
    assert passing["no_december_access_during_development"] is True
    assert passing["schema_check_passed"] is True

    # The weight-policy gate is real evidence, not a constant.
    failing = _governance(
        prepared=prepared,
        bundle=bundle,
        r3_reconstruction_passed=True,
        weight_policies_normalized=False,
    )
    assert failing["weight_policy_check_passed"] is False
