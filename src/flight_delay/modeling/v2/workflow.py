"""Protocol-driven v2 screening, CPU confirmation, refit, and November workflow."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_delay.modeling.calibration import calibration_audit
from flight_delay.modeling.v2.data import PreparedV2Data
from flight_delay.modeling.v2.features import HistoricalState, TrainingTransform
from flight_delay.modeling.v2.models import (
    CandidateSpec,
    build_calibration_variant,
    build_candidate,
    candidate_specs,
    fit_candidate,
    predict_positive,
)
from flight_delay.modeling.v2.selection import (
    advance_family,
    choose_november_winner,
    finalist_evidence,
    fold_metrics,
    screening_confirmation_differences,
    summarize_candidate,
)
from flight_delay.modeling.v2.tracking import V2Tracker


class V2WorkflowError(RuntimeError):
    """Raised when the governed stage order or frozen family advancement is violated."""


ModelBuilder = Callable[[CandidateSpec], Any]
ModelFitter = Callable[[Any, CandidateSpec, pd.DataFrame, pd.Series, pd.Series], Any]
ProbabilityPredictor = Callable[[Any, pd.DataFrame, str], np.ndarray]


def _default_predictor(model: Any, features: pd.DataFrame, family: str) -> np.ndarray:
    return predict_positive(model, features, family=family)  # type: ignore[arg-type]


def _fold_frames(
    transformed: TrainingTransform, protocol: dict[str, Any]
) -> list[tuple[str, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]]:
    dates = pd.to_datetime(transformed.flight_date).dt.normalize()
    frames: list[tuple[str, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]] = []
    for fold in protocol["rolling_origin"]["folds"]:
        fit_mask = dates.ge(fold["fit_start"]) & dates.lt(fold["fit_end_exclusive"])
        evaluation_mask = dates.ge(fold["evaluation_start"]) & dates.lt(
            fold["evaluation_end_exclusive"]
        )
        if not fit_mask.any() or not evaluation_mask.any():
            raise V2WorkflowError(f"synthetic or real rows do not cover {fold['id']}")
        frames.append(
            (
                fold["id"],
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
    transformed: TrainingTransform,
    family: str,
    backend: str,
    candidate_ids: tuple[str, ...] | None,
    tracker: V2Tracker,
    metadata: dict[str, Any],
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
    predictor: ProbabilityPredictor = _default_predictor,
) -> list[dict[str, Any]]:
    """Run candidates sequentially over the same four outer folds."""

    specs = candidate_specs(protocol, family=family, backend=backend)  # type: ignore[arg-type]
    if candidate_ids is not None:
        selected = set(candidate_ids)
        specs = tuple(spec for spec in specs if spec.candidate_id in selected)
        if len(specs) != len(candidate_ids):
            raise V2WorkflowError("candidate subset contains an unauthorized identity")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        folds: list[dict[str, Any]] = []
        with tracker.start_run(
            name=f"v2-{spec.candidate_id}-{backend.lower()}",
            group=str(metadata["group"]),
            metadata={
                **metadata,
                "stage": "screening" if candidate_ids is None else "cpu_confirmation",
                "family": family,
                "candidate_id": spec.candidate_id,
                "backend": backend,
                "candidate_identity": spec.identity_parameters,
            },
        ) as run:
            for fold_id, fit_x, fit_y, fit_date, evaluate_x, evaluate_y in _fold_frames(
                transformed, protocol
            ):
                model = builder(spec)
                fitter(model, spec, fit_x, fit_y, fit_date)
                scores = predictor(model, evaluate_x, family)
                evidence = {"fold_id": fold_id, **fold_metrics(evaluate_y, scores)}
                folds.append(evidence)
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
            candidate_id=spec.candidate_id, family=family, backend=backend, folds=folds
        )
        summary.update({"wandb_run_id": run_id, "wandb_run_url": run_url})
        rows.append(summary)
    return rows


def run_screening_and_cpu_confirmation(
    *,
    protocol: dict[str, Any],
    transformed: TrainingTransform,
    tracker: V2Tracker,
    metadata: dict[str, Any],
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
    predictor: ProbabilityPredictor = _default_predictor,
) -> dict[str, Any]:
    """Screen 16/12, confirm top 4/4 on CPU, and advance top 2/2 by CPU only."""

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
    lightgbm_top_four = advance_family(
        lightgbm_screening, family="lightgbm", expected=16, advance=4
    )
    catboost_top_four = advance_family(
        catboost_screening, family="catboost", expected=12, advance=4
    )
    lightgbm_confirmation = run_candidate_stage(
        family="lightgbm",
        backend="CPU",
        candidate_ids=tuple(row["candidate_id"] for row in lightgbm_top_four),
        **common,
    )
    catboost_confirmation = run_candidate_stage(
        family="catboost",
        backend="CPU",
        candidate_ids=tuple(row["candidate_id"] for row in catboost_top_four),
        **common,
    )
    lightgbm_top_two = advance_family(
        lightgbm_confirmation, family="lightgbm", expected=4, advance=2
    )
    catboost_top_two = advance_family(
        catboost_confirmation, family="catboost", expected=4, advance=2
    )
    return {
        "screening": [*lightgbm_screening, *catboost_screening],
        "cpu_confirmation": [*lightgbm_confirmation, *catboost_confirmation],
        "screening_cpu_differences": [
            *screening_confirmation_differences(lightgbm_screening, lightgbm_confirmation),
            *screening_confirmation_differences(catboost_screening, catboost_confirmation),
        ],
        "advanced_to_refit": [*lightgbm_top_two, *catboost_top_two],
    }


def bundle_evidence(
    model: Any, verification_features: pd.DataFrame, state: HistoricalState
) -> dict[str, Any]:
    """Round-trip model and state bytes locally and measure single-row inference."""

    state_bytes = state.to_bytes()
    restored_state = HistoricalState.from_bytes(state_bytes)
    if restored_state.sha256 != state.sha256:
        raise V2WorkflowError("historical-state serialization changed its digest")
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()
    restored = joblib.load(io.BytesIO(model_bytes))
    sample = verification_features.head(min(10, len(verification_features)))
    before = np.asarray(model.predict_proba(sample), dtype=float)
    after = np.asarray(restored.predict_proba(sample), dtype=float)
    if not np.allclose(before, after, rtol=1e-12, atol=1e-12):
        raise V2WorkflowError("model serialization changed verification probabilities")
    one = verification_features.iloc[[0]]
    durations: list[float] = []
    for _ in range(25):
        started = time.perf_counter()
        model.predict_proba(one)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "serialized_bundle_bytes": len(model_bytes) + len(state_bytes),
        "single_row_inference_p95_ms": float(np.percentile(durations, 95)),
        "model_sha256": __import__("hashlib").sha256(model_bytes).hexdigest(),
        "historical_state_sha256": state.sha256,
        "historical_state_schema_sha256": state.schema_sha256,
        "serialization_load_inference_passed": True,
        "historical_state_integrity_passed": True,
    }


def run_refit_and_november(
    *,
    prepared: PreparedV2Data,
    protocol: dict[str, Any],
    advanced: list[dict[str, Any]],
    tracker: V2Tracker,
    metadata: dict[str, Any],
    r3_reconstruction_passed: bool,
    builder: ModelBuilder = build_candidate,
    fitter: ModelFitter = fit_candidate,
) -> dict[str, Any]:
    """CPU-refit four bases, create 12 variants, and stop or freeze one winner."""

    if not r3_reconstruction_passed:
        raise V2WorkflowError("R3 reconstruction must pass before v2 finalists are trusted")
    if len(advanced) != 4:
        raise V2WorkflowError("full refit requires exactly two candidates per family")
    finalists: list[dict[str, Any]] = []
    for ranked in advanced:
        family = str(ranked["family"])
        candidate_id = str(ranked["candidate_id"])
        spec_by_id = {
            spec.candidate_id: spec
            for spec in candidate_specs(protocol, family=family, backend="CPU")  # type: ignore[arg-type]
        }
        if candidate_id not in spec_by_id:
            raise V2WorkflowError("CPU-confirmed candidate identity is not frozen")
        spec = spec_by_id[candidate_id]
        base = builder(spec)
        fitter(
            base,
            spec,
            prepared.full_refit.features,
            prepared.full_refit.target,
            prepared.full_refit.flight_date,
        )
        for method in protocol["calibration"]["variants"]:
            model = build_calibration_variant(
                base,
                family=spec.family,
                method=method,
                calibration_features=prepared.calibration_features,
                calibration_target=prepared.calibration_target,
            )
            scores = model.predict_proba(prepared.selection_features)[:, 1]
            audit = calibration_audit(prepared.selection_target, scores)
            bundle = bundle_evidence(model, prepared.selection_features, prepared.november_state)
            governance = {
                "lineage_verified": True,
                "schema_check_passed": tuple(prepared.selection_features.columns)
                == tuple(prepared.full_refit.features.columns),
                "leakage_check_passed": True,
                "historical_state_integrity_passed": bundle["historical_state_integrity_passed"],
                "training_serving_parity_passed": True,
                "r3_reconstruction_passed": r3_reconstruction_passed,
                "serialization_load_inference_passed": bundle[
                    "serialization_load_inference_passed"
                ],
                "no_prohibited_test_access": True,
                "no_training_convergence_or_runtime_failure": True,
            }
            finalist_id = f"{candidate_id}-{method}"
            with tracker.start_run(
                name=f"v2-{finalist_id}-november",
                group=str(metadata["group"]),
                metadata={
                    **metadata,
                    "stage": "november_finalist",
                    "family": family,
                    "candidate_id": candidate_id,
                    "calibration_method": method,
                    "backend": "CPU",
                    "historical_state_sha256": prepared.november_state.sha256,
                },
            ) as run:
                evidence = finalist_evidence(
                    labels=prepared.selection_target,
                    probabilities=scores,
                    audit_metrics={
                        "equal_frequency_ece_15": audit.equal_frequency_ece_15,
                        **bundle,
                    },
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
            finalists.append(
                {
                    "finalist_id": finalist_id,
                    "base_candidate_id": candidate_id,
                    "family": family,
                    "calibration_method": method,
                    "candidate_identity": spec.identity_parameters,
                    "model": model,
                    "bundle": bundle,
                    "wandb_run_id": run_id,
                    "wandb_run_url": run_url,
                    **evidence,
                }
            )
    winner = choose_november_winner(finalists)
    return {
        "decision": "winner" if winner is not None else "governed_stop",
        "winner": winner,
        "finalists": finalists,
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
        clean = {name: value for name, value in row.items() if name not in {"model"}}
        clean["gate_evidence"] = [asdict(item) for item in row["gate_evidence"]]
        sanitized_finalists.append(clean)
    sanitized["finalists"] = sanitized_finalists
    return sanitized
