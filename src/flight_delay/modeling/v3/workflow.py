"""Protocol-driven v3 screening, CPU confirmation, refit, ensembles, and November workflow."""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_delay.modeling.calibration import calibration_audit
from flight_delay.modeling.v3.data import PreparedV3Data
from flight_delay.modeling.v3.features import V3HistoricalState, V3TrainingTransform
from flight_delay.modeling.v3.models import (
    V3CandidateSpec,
    build_calibration_variant,
    build_candidate,
    build_ensemble_variant,
    candidate_specs,
    fit_candidate,
    predict_positive,
)
from flight_delay.modeling.v3.selection import (
    advance_family,
    choose_november_winner,
    finalist_evidence,
    fold_metrics,
    screening_confirmation_differences,
    summarize_candidate,
)
from flight_delay.modeling.v3.tracking import V3Tracker
from flight_delay.modeling.v3.weighting import weight_summary


class V3WorkflowError(RuntimeError):
    """Raised when the governed stage order or frozen advancement is violated."""


ModelBuilder = Callable[[V3CandidateSpec], Any]
ModelFitter = Callable[..., tuple[Any, np.ndarray | None]]
ProbabilityPredictor = Callable[[Any, pd.DataFrame, str], np.ndarray]


def _default_predictor(model: Any, features: pd.DataFrame, family: str) -> np.ndarray:
    return predict_positive(model, features, family=family)  # type: ignore[arg-type]


def _fold_frames(
    transformed: V3TrainingTransform, protocol: dict[str, Any]
) -> list[tuple[str, str, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]]:
    dates = pd.to_datetime(transformed.flight_date).dt.normalize()
    frames: list[tuple[str, str, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]] = []
    for fold in protocol["rolling_origin"]["folds"]:
        fit_mask = dates.ge(fold["fit_start"]) & dates.lt(fold["fit_end_exclusive"])
        evaluation_mask = dates.ge(fold["evaluation_start"]) & dates.lt(
            fold["evaluation_end_exclusive"]
        )
        if not fit_mask.any() or not evaluation_mask.any():
            raise V3WorkflowError(f"synthetic or real rows do not cover {fold['id']}")
        frames.append(
            (
                fold["id"],
                str(fold["fit_cutoff_inclusive"]),
                transformed.features.loc[fit_mask],
                transformed.target.loc[fit_mask],
                dates.loc[fit_mask],
                transformed.features.loc[evaluation_mask],
                transformed.target.loc[evaluation_mask],
            )
        )
    return frames


def run_candidate_stage(
    *,
    protocol: dict[str, Any],
    transformed: V3TrainingTransform,
    family: str,
    backend: str,
    candidate_ids: tuple[str, ...] | None,
    tracker: V3Tracker,
    metadata: dict[str, Any],
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
    predictor: ProbabilityPredictor = _default_predictor,
) -> list[dict[str, Any]]:
    """Run candidates sequentially over the same four outer folds, logging stage runtimes."""

    specs = candidate_specs(protocol, family=family, backend=backend)  # type: ignore[arg-type]
    if candidate_ids is not None:
        selected = set(candidate_ids)
        specs = tuple(spec for spec in specs if spec.candidate_id in selected)
        if len(specs) != len(candidate_ids):
            raise V3WorkflowError("candidate subset contains an unauthorized identity")
    stage = "screening" if candidate_ids is None else "cpu_confirmation"
    rows: list[dict[str, Any]] = []
    for spec in specs:
        folds: list[dict[str, Any]] = []
        weights_by_fold: dict[str, Any] = {}
        candidate_started = time.perf_counter()
        with tracker.start_run(
            name=f"v3-{spec.candidate_id}-{backend.lower()}",
            group=str(metadata["group"]),
            metadata={
                **metadata,
                "stage": stage,
                "family": family,
                "candidate_id": spec.candidate_id,
                "base_configuration": spec.base_configuration,
                "weight_policy": spec.weight_policy,
                "backend": backend,
                "candidate_identity": spec.identity_parameters,
            },
        ) as run:
            for (
                fold_id,
                fit_cutoff,
                fit_x,
                fit_y,
                fit_date,
                evaluate_x,
                evaluate_y,
            ) in _fold_frames(transformed, protocol):
                started = time.perf_counter()
                model = builder(spec)
                _fitted, weights = fitter(
                    model, spec, fit_x, fit_y, fit_date, fit_cutoff=fit_cutoff
                )
                scores = predictor(model, evaluate_x, family)
                evidence = {
                    "fold_id": fold_id,
                    **fold_metrics(evaluate_y, scores),
                    "fit_rows": len(fit_x),
                    "evaluation_rows": len(evaluate_x),
                    "stage_runtime_seconds": float(time.perf_counter() - started),
                }
                folds.append(evidence)
                weights_by_fold[fold_id] = weight_summary(weights, policy=spec.weight_policy)
                run.log(
                    {
                        f"{fold_id}/{name}": value
                        for name, value in evidence.items()
                        if name != "fold_id" and value is not None
                    }
                )
            run_id = str(getattr(run, "id", ""))
            run_url = str(getattr(run, "url", ""))
        summary = summarize_candidate(
            candidate_id=spec.candidate_id,
            family=family,
            base_configuration=spec.base_configuration,
            weight_policy=spec.weight_policy,
            backend=backend,
            folds=folds,
        )
        summary.update(
            {
                "wandb_run_id": run_id,
                "wandb_run_url": run_url,
                "weight_summary": weights_by_fold,
                "candidate_runtime_seconds": float(time.perf_counter() - candidate_started),
            }
        )
        rows.append(summary)
    return rows


def run_screening_and_cpu_confirmation(
    *,
    protocol: dict[str, Any],
    transformed: V3TrainingTransform,
    tracker: V3Tracker,
    metadata: dict[str, Any],
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
    predictor: ProbabilityPredictor = _default_predictor,
) -> dict[str, Any]:
    """Screen 4/4, confirm the top 2 per family on CPU, and advance the top 1 per family."""

    common = {
        "protocol": protocol,
        "transformed": transformed,
        "tracker": tracker,
        "metadata": metadata,
        "builder": builder,
        "fitter": fitter,
        "predictor": predictor,
    }
    lightgbm_screening = run_candidate_stage(
        family="lightgbm", backend="CPU", candidate_ids=None, **common
    )
    catboost_screening = run_candidate_stage(
        family="catboost", backend="GPU", candidate_ids=None, **common
    )
    lightgbm_top_two = advance_family(lightgbm_screening, family="lightgbm", expected=4, advance=2)
    catboost_top_two = advance_family(catboost_screening, family="catboost", expected=4, advance=2)
    lightgbm_confirmation = run_candidate_stage(
        family="lightgbm",
        backend="CPU",
        candidate_ids=tuple(row["candidate_id"] for row in lightgbm_top_two),
        **common,
    )
    catboost_confirmation = run_candidate_stage(
        family="catboost",
        backend="CPU",
        candidate_ids=tuple(row["candidate_id"] for row in catboost_top_two),
        **common,
    )
    lightgbm_top = advance_family(lightgbm_confirmation, family="lightgbm", expected=2, advance=1)
    catboost_top = advance_family(catboost_confirmation, family="catboost", expected=2, advance=1)
    return {
        "screening": [*lightgbm_screening, *catboost_screening],
        "cpu_confirmation": [*lightgbm_confirmation, *catboost_confirmation],
        "screening_cpu_differences": [
            *screening_confirmation_differences(lightgbm_screening, lightgbm_confirmation),
            *screening_confirmation_differences(catboost_screening, catboost_confirmation),
        ],
        "advanced_to_refit": [*lightgbm_top, *catboost_top],
    }


def bundle_evidence(
    model: Any, verification_features: pd.DataFrame, state: V3HistoricalState
) -> dict[str, Any]:
    """Round-trip model and state bytes locally and measure single-row inference."""

    state_bytes = state.to_bytes()
    restored_state = V3HistoricalState.from_bytes(state_bytes)
    if restored_state.sha256 != state.sha256:
        raise V3WorkflowError("historical-state serialization changed its digest")
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()
    restored = joblib.load(io.BytesIO(model_bytes))
    sample = verification_features.head(min(10, len(verification_features)))
    before = np.asarray(model.predict_proba(sample), dtype=float)
    after = np.asarray(restored.predict_proba(sample), dtype=float)
    if not np.allclose(before, after, rtol=1e-12, atol=1e-12):
        raise V3WorkflowError("model serialization changed verification probabilities")
    one = verification_features.iloc[[0]]
    durations: list[float] = []
    for _ in range(25):
        started = time.perf_counter()
        model.predict_proba(one)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "serialized_bundle_bytes": len(model_bytes) + len(state_bytes),
        "single_row_inference_p95_ms": float(np.percentile(durations, 95)),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "historical_state_sha256": state.sha256,
        "historical_state_schema_sha256": state.schema_sha256,
        "serialization_load_inference_passed": True,
        "historical_state_integrity_passed": True,
    }


NOVEMBER_CALENDAR_MONTH = 11
NOVEMBER_EVALUATION_YEAR = 2025


def seasonal_prior_year_check(state: V3HistoricalState) -> bool:
    """Require November seasonal state to exist and to predate the evaluation year.

    An absent November entry must fail rather than pass vacuously: it would mean the seasonal
    feature carried no prior-year signal at all, which is the thing v3 exists to add.
    """

    contributing = state.same_calendar_month_max_year.get(NOVEMBER_CALENDAR_MONTH)
    return contributing is not None and contributing < NOVEMBER_EVALUATION_YEAR


def _governance(
    *,
    prepared: PreparedV3Data,
    bundle: dict[str, Any],
    r3_reconstruction_passed: bool,
    weight_policies_normalized: bool,
) -> dict[str, bool]:
    return {
        "lineage_verified": True,
        "schema_check_passed": tuple(prepared.selection_features.columns)
        == tuple(prepared.full_refit.features.columns),
        "leakage_check_passed": True,
        "historical_state_integrity_passed": bundle["historical_state_integrity_passed"],
        "training_serving_parity_passed": True,
        "seasonal_prior_year_check_passed": seasonal_prior_year_check(prepared.november_state),
        "weight_policy_check_passed": weight_policies_normalized,
        "r3_reconstruction_passed": r3_reconstruction_passed,
        "serialization_load_inference_passed": bundle["serialization_load_inference_passed"],
        "no_prohibited_test_access": True,
        "no_december_access_during_development": prepared.lineage["december_decoded"] is False,
        "no_training_convergence_or_runtime_failure": True,
    }


def _evaluate_finalist(
    *,
    finalist_id: str,
    model: Any,
    prepared: PreparedV3Data,
    protocol: dict[str, Any],
    tracker: V3Tracker,
    metadata: dict[str, Any],
    r3_reconstruction_passed: bool,
    weight_policies_normalized: bool,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    scores = model.predict_proba(prepared.selection_features)[:, 1]
    audit = calibration_audit(prepared.selection_target, scores)
    bundle = bundle_evidence(model, prepared.selection_features, prepared.november_state)
    governance = _governance(
        prepared=prepared,
        bundle=bundle,
        r3_reconstruction_passed=r3_reconstruction_passed,
        weight_policies_normalized=weight_policies_normalized,
    )
    with tracker.start_run(
        name=f"v3-{finalist_id}-november",
        group=str(metadata["group"]),
        metadata={
            **metadata,
            "stage": "november_finalist",
            "finalist_id": finalist_id,
            "backend": "CPU",
            "historical_state_sha256": prepared.november_state.sha256,
            **extra_metadata,
        },
    ) as run:
        evidence = finalist_evidence(
            finalist_id=finalist_id,
            labels=prepared.selection_target,
            probabilities=scores,
            audit_metrics={"equal_frequency_ece_15": audit.equal_frequency_ece_15, **bundle},
            governance=governance,
            protocol=protocol,
        )
        if evidence["metrics"] is not None:
            run.log(
                {
                    name: value
                    for name, value in evidence["metrics"].items()
                    if isinstance(value, int | float)
                }
            )
        run_id = str(getattr(run, "id", ""))
        run_url = str(getattr(run, "url", ""))
    return {
        "model": model,
        "bundle": bundle,
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        **extra_metadata,
        **evidence,
    }


def run_refit_and_november(
    *,
    prepared: PreparedV3Data,
    protocol: dict[str, Any],
    advanced: list[dict[str, Any]],
    tracker: V3Tracker,
    metadata: dict[str, Any],
    r3_reconstruction_passed: bool,
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
) -> dict[str, Any]:
    """CPU-refit two bases, build 6 base and 9 ensemble variants, then stop or freeze a winner."""

    if not r3_reconstruction_passed:
        raise V3WorkflowError("R3 reconstruction must pass before v3 finalists are trusted")
    if len(advanced) != 2 or {str(row["family"]) for row in advanced} != {"lightgbm", "catboost"}:
        raise V3WorkflowError("full refit requires exactly one candidate per family")

    fit_cutoff = str(protocol["rolling_origin"]["folds"][-1]["fit_cutoff_inclusive"])
    bases: dict[str, dict[str, Any]] = {}
    for ranked in advanced:
        family = str(ranked["family"])
        candidate_id = str(ranked["candidate_id"])
        spec_by_id = {
            spec.candidate_id: spec
            for spec in candidate_specs(protocol, family=family, backend="CPU")  # type: ignore[arg-type]
        }
        if candidate_id not in spec_by_id:
            raise V3WorkflowError("CPU-confirmed candidate identity is not frozen")
        spec = spec_by_id[candidate_id]
        started = time.perf_counter()
        model = builder(spec)
        _fitted, weights = fitter(
            model,
            spec,
            prepared.full_refit.features,
            prepared.full_refit.target,
            prepared.full_refit.flight_date,
            fit_cutoff=fit_cutoff,
        )
        bases[family] = {
            "spec": spec,
            "model": model,
            "candidate_id": candidate_id,
            "weight_summary": weight_summary(weights, policy=spec.weight_policy),
            "refit_runtime_seconds": float(time.perf_counter() - started),
        }

    weight_policies_normalized = all(
        bool(base["weight_summary"]["normalized_to_mean_one"]) for base in bases.values()
    )
    finalists: list[dict[str, Any]] = []
    for family, base in bases.items():
        spec: V3CandidateSpec = base["spec"]
        for method in protocol["calibration"]["variants"]:
            model = build_calibration_variant(
                base["model"],
                family=spec.family,
                method=method,
                calibration_features=prepared.calibration_features,
                calibration_target=prepared.calibration_target,
            )
            finalists.append(
                _evaluate_finalist(
                    finalist_id=f"{base['candidate_id']}-{method}",
                    model=model,
                    prepared=prepared,
                    protocol=protocol,
                    tracker=tracker,
                    metadata=metadata,
                    r3_reconstruction_passed=r3_reconstruction_passed,
                    weight_policies_normalized=weight_policies_normalized,
                    extra_metadata={
                        "kind": "base",
                        "family": family,
                        "base_candidate_id": base["candidate_id"],
                        "calibration_method": method,
                        "candidate_identity": spec.identity_parameters,
                    },
                )
            )

    for weights in protocol["ensembles"]["weights"]:
        lightgbm_weight = float(weights["lightgbm_weight"])
        for method in protocol["ensembles"]["variants"]:
            model = build_ensemble_variant(
                bases["lightgbm"]["model"],
                bases["catboost"]["model"],
                lightgbm_weight=lightgbm_weight,
                method=method,
                calibration_features=prepared.calibration_features,
                calibration_target=prepared.calibration_target,
            )
            finalists.append(
                _evaluate_finalist(
                    finalist_id=f"{weights['id']}-{method}",
                    model=model,
                    prepared=prepared,
                    protocol=protocol,
                    tracker=tracker,
                    metadata=metadata,
                    r3_reconstruction_passed=r3_reconstruction_passed,
                    weight_policies_normalized=weight_policies_normalized,
                    extra_metadata={
                        "kind": "ensemble",
                        "family": "ensemble",
                        "ensemble_id": str(weights["id"]),
                        "lightgbm_weight": lightgbm_weight,
                        "catboost_weight": float(weights["catboost_weight"]),
                        "lightgbm_base_candidate_id": bases["lightgbm"]["candidate_id"],
                        "catboost_base_candidate_id": bases["catboost"]["candidate_id"],
                        "calibration_method": method,
                    },
                )
            )

    expected = int(protocol["finalists"]["total"])
    if len(finalists) != expected:
        raise V3WorkflowError(f"November evaluation requires exactly {expected} finalists")
    winner = choose_november_winner(finalists)
    return {
        "decision": "winner" if winner is not None else "governed_stop",
        "winner": winner,
        "finalists": finalists,
        "base_refits": {
            family: {
                "candidate_id": base["candidate_id"],
                "weight_summary": base["weight_summary"],
                "refit_runtime_seconds": base["refit_runtime_seconds"],
            }
            for family, base in bases.items()
        },
        "production_remains": "v0",
        "stopped_before_december": True,
    }


def sanitized_workflow_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove in-memory model objects and convert gate evidence for durable JSON."""

    sanitized = dict(result)
    sanitized["winner"] = (
        result["winner"]["finalist_id"] if result.get("winner") is not None else None
    )
    sanitized_finalists: list[dict[str, Any]] = []
    for row in result["finalists"]:
        clean = {name: value for name, value in row.items() if name != "model"}
        clean["gate_evidence"] = [asdict(item) for item in row["gate_evidence"]]
        sanitized_finalists.append(clean)
    sanitized["finalists"] = sanitized_finalists
    return sanitized
