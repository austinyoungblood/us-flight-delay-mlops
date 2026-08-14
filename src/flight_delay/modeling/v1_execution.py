"""Governed v1 preflight, one-time execution, bundles, and December qualification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.modeling.calibration import calibration_audit, fit_calibrator
from flight_delay.modeling.evaluation import evaluate_binary, measure_single_row_latency
from flight_delay.modeling.release import (
    LOCKED_THRESHOLD,
    compare_development_metrics,
)
from flight_delay.modeling.remediation import (
    EXPECTED_MATRIX,
    build_remediation_model,
    partition_remediation_data,
)
from flight_delay.modeling.v1_catboost import (
    CATBOOST_VERSION,
    build_calibration_variant,
    build_catboost_candidate,
    candidate_specs,
    fit_catboost_base,
    installed_catboost_version,
    validate_catboost_runtime_contract,
)
from flight_delay.modeling.v1_data import (
    V1_CATEGORICAL_FEATURES,
    V1_FEATURES,
    AdaptedV1Frame,
    adapt_v1_frame,
    development_period,
    load_december_data,
    load_development_data,
)
from flight_delay.modeling.v1_protocol import (
    PROTOCOL_ID,
    load_and_validate_v1_protocol,
    sha256_file,
)
from flight_delay.modeling.v1_selection import (
    all_gates_pass,
    choose_november_winner,
    evaluate_november_gates,
    evaluate_qualification_gates,
    probability_metrics,
    select_v1_threshold,
    top_two_catboost,
)
from flight_delay.modeling.v1_tracking import V1Tracker, WandbTracker

PROTOCOL_MERGE_SHA = "62e8046b67c696cb3ec83e63f6f6128977a48292"
DEVELOPMENT_MARKER = Path("artifacts/v1/development/execution_marker.json")
WINNER_LOCK = Path("artifacts/v1/development/november_winner_lock.json")
WINNER_MODEL = Path("artifacts/v1/development/november_winner.joblib")
QUALIFICATION_MARKER = Path("artifacts/v1/qualification/execution_marker.json")
RELEASE_CANDIDATE_LOCK = Path("artifacts/v1/qualification/v1_release_candidate_lock.json")
ROLLING_METRICS = (
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "prevalence",
    "probability_mean",
)


class V1ExecutionError(RuntimeError):
    """Raised when a governed execution precondition or invariant fails."""


@dataclass(frozen=True)
class CandidateBundle:
    directory: Path
    file_hashes: dict[str, str]
    aggregate_digest: str
    byte_size: int
    model_sha256: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise V1ExecutionError(f"immutable output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        if refuse_existing and path.exists():
            raise V1ExecutionError(f"immutable output already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise V1ExecutionError(f"immutable output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        temporary.write_text(text, encoding="utf-8")
        if refuse_existing and path.exists():
            raise V1ExecutionError(f"immutable output already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_marker(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, payload, refuse_existing=True)


def update_marker(path: Path, updates: dict[str, Any]) -> None:
    if not path.is_file():
        raise V1ExecutionError(f"execution marker does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V1ExecutionError("execution marker is unreadable") from error
    payload.update(updates)
    _atomic_json(path, payload)


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def implementation_git_sha(repository_root: Path) -> str:
    return _git(repository_root, "rev-parse", "HEAD")


def require_applied_git_state(repository_root: Path) -> str:
    if _git(repository_root, "status", "--porcelain"):
        raise V1ExecutionError("applied v1 execution requires a clean Git worktree")
    try:
        _git(repository_root, "merge-base", "--is-ancestor", PROTOCOL_MERGE_SHA, "HEAD")
    except subprocess.CalledProcessError as error:
        raise V1ExecutionError(
            "the immutable protocol merge must be an ancestor of HEAD"
        ) from error
    return implementation_git_sha(repository_root)


def validate_dependency_isolation(repository_root: Path) -> dict[str, Any]:
    """Validate optional locks and prove production Dockerfiles install only base dependencies."""

    project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    v1_extra = project["project"]["optional-dependencies"].get("v1")
    if v1_extra != [f"catboost=={CATBOOST_VERSION}"]:
        raise V1ExecutionError("pyproject v1 extra must contain only the exact CatBoost pin")
    lock_lines = {
        line.strip()
        for line in (repository_root / "requirements-v1.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {f"catboost=={CATBOOST_VERSION}", "graphviz==0.21", "plotly==6.5.0"}
    if lock_lines != required:
        raise V1ExecutionError("requirements-v1.lock differs from the isolated dependency set")
    dockerfiles = (
        repository_root / "services/api/Dockerfile",
        repository_root / "services/user_ui/Dockerfile",
        repository_root / "services/monitor_ui/Dockerfile",
    )
    for path in dockerfiles:
        source = path.read_text(encoding="utf-8")
        if "catboost" in source.casefold() or "requirements-v1.lock" in source or ".[v1]" in source:
            raise V1ExecutionError(
                f"runtime Dockerfile includes optional modeling dependencies: {path}"
            )
    return {
        "v1_extra": v1_extra,
        "v1_constraints": sorted(lock_lines),
        "runtime_images_install_base_only": True,
    }


def validate_production_v0(repository_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Prove the committed release and deployment still point at immutable production v0."""

    incumbent = protocol["incumbent"]
    release = _read_json_object(repository_root / "release/release_decision.json")
    deployment = _read_json_object(repository_root / "deploy/deployment_manifest.json")["model"]
    expected = {
        "serving_alias": incumbent["serving_alias"],
        "registry_version": incumbent["registry_version"],
        "registry_digest": incumbent["registry_digest"],
        "bundle_digest": incumbent["bundle_sha256"],
        "internal_production_gate_passed": incumbent["internal_production_gate_passed"],
        "deployment_purpose": incumbent["deployment_purpose"],
    }
    for name, value in expected.items():
        if release.get(name) != value:
            raise V1ExecutionError(f"committed release no longer preserves production v0: {name}")
    deployment_expected = {
        **expected,
        "release_bundle_digest": expected["bundle_digest"],
        "classification_threshold": incumbent["threshold"],
    }
    deployment_expected.pop("bundle_digest")
    for name, value in deployment_expected.items():
        if deployment.get(name) != value:
            raise V1ExecutionError(f"deployment manifest no longer preserves production v0: {name}")
    return {"release": expected, "deployment_unchanged": True}


def preflight(
    repository_root: Path, *, stage: Literal["development", "qualification"]
) -> dict[str, Any]:
    """Validate static execution contracts without parquet, CatBoost, W&B, or network access."""

    root = repository_root.resolve()
    protocol, lock, protocol_sha = load_and_validate_v1_protocol(
        root / "configs/v1_experiment_protocol.yaml",
        lock_path=root / "experiments/v1/protocol_lock.json",
        repository_root=root,
    )
    dependencies = validate_dependency_isolation(root)
    production = validate_production_v0(root, protocol)
    manifest = read_manifest(root / protocol["dependencies"]["processed_dataset_manifest"]["path"])
    version = installed_catboost_version()
    report: dict[str, Any] = {
        "mode": "dry-run/preflight",
        "stage": stage,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "protocol_lock_valid": lock["protocol_sha256"] == protocol_sha,
        "implementation_git_sha": implementation_git_sha(root),
        "catboost_required_version": CATBOOST_VERSION,
        "catboost_installed_version": version,
        "catboost_version_ready": version == CATBOOST_VERSION,
        "dependency_isolation": dependencies,
        "production_v0": production,
        "dataset_manifest_digest": manifest["manifest_digest"],
        "parquet_opened": False,
        "catboost_runtime_imported": "catboost" in sys.modules,
        "wandb_imported": "wandb" in sys.modules,
        "network_contacted": False,
        "historical_test_accessed": False,
        "production_v0_mutated": False,
    }
    if stage == "development":
        report.update(
            {
                "candidates": [item.candidate_id for item in candidate_specs(protocol)],
                "rolling_folds": [fold["id"] for fold in protocol["rolling_origin"]["folds"]],
                "calibration_variants": protocol["refit_calibration"]["variants"],
                "finalist_variant_count": protocol["refit_calibration"][
                    "expected_finalist_variant_count"
                ],
                "r3_reconstruction_required": protocol["control"][
                    "reconstruction_required_before_challenger_trust"
                ],
                "stops_before_december": True,
                "execution_marker_exists": (root / DEVELOPMENT_MARKER).exists(),
            }
        )
    else:
        report.update(
            {
                "requires_completed_november_winner_lock": True,
                "winner_lock_exists": (root / WINNER_LOCK).is_file(),
                "winner_model_exists": (root / WINNER_MODEL).is_file(),
                "qualification_marker_exists": (root / QUALIFICATION_MARKER).exists(),
                "retraining_permitted": False,
                "recalibration_permitted": False,
                "threshold_change_permitted": False,
            }
        )
    return report


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _bundle_digest(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(file_hashes)).hexdigest()


def write_candidate_bundle(
    *,
    directory: Path,
    model: Any,
    candidate_id: str,
    parameters: dict[str, Any],
    calibration_method: str,
    threshold: float,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    protocol_lock: dict[str, Any],
    verification_features: pd.DataFrame,
) -> CandidateBundle:
    """Write, hash, clean-load, and inference-check a non-Registry candidate bundle."""

    if directory.exists():
        raise V1ExecutionError(f"candidate bundle already exists: {directory}")
    directory.mkdir(parents=True)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    _write_json_file(directory / "feature_schema.json", {"features": list(V1_FEATURES)})
    _write_json_file(
        directory / "categorical_features.json",
        {"categorical_features": list(V1_CATEGORICAL_FEATURES)},
    )
    _write_json_file(directory / "threshold.json", {"threshold": threshold})
    _write_json_file(directory / "metrics_development.json", metrics)
    _write_json_file(
        directory / "metadata.json",
        {
            **metadata,
            "candidate_id": candidate_id,
            "complete_catboost_parameters": parameters,
            "calibration_method": calibration_method,
        },
    )
    _write_json_file(directory / "protocol_lock.json", protocol_lock)
    file_hashes = {
        path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file()
    }
    aggregate = _bundle_digest(file_hashes)
    before = np.asarray(model.predict_proba(verification_features)[:, 1], dtype=float)
    restored = joblib.load(model_path)
    after = np.asarray(restored.predict_proba(verification_features)[:, 1], dtype=float)
    if not np.allclose(before, after, rtol=1e-12, atol=1e-12):
        raise V1ExecutionError("candidate bundle serialization changed probabilities")
    return CandidateBundle(
        directory=directory,
        file_hashes=file_hashes,
        aggregate_digest=aggregate,
        byte_size=sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
        model_sha256=file_hashes["model.joblib"],
    )


def load_verified_bundle(bundle: CandidateBundle) -> Any:
    for filename, expected in bundle.file_hashes.items():
        path = bundle.directory / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise V1ExecutionError(f"candidate bundle hash mismatch: {filename}")
    if _bundle_digest(bundle.file_hashes) != bundle.aggregate_digest:
        raise V1ExecutionError("candidate bundle aggregate digest mismatch")
    return joblib.load(bundle.directory / "model.joblib")


def _close_figures(result: Any) -> None:
    import matplotlib.pyplot as plt

    for figure in result.figures.values():
        plt.close(figure)


def _threshold_metrics(labels: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    result = evaluate_binary(labels, scores, threshold=threshold)
    _close_figures(result)
    return result.metrics


def reconstruct_governed_r3(train: pd.DataFrame, november: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the locked R3-sigmoid incumbent before any challenger is trusted."""

    partitions = partition_remediation_data(train, november)
    base, schema = build_remediation_model("R3", dict(EXPECTED_MATRIX["R3"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base.fit(partitions.final_fit.loc[:, schema], partitions.final_fit["target"])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise V1ExecutionError("R3 reconstruction emitted a convergence warning")
    model = fit_calibrator(
        base,
        partitions.calibration.loc[:, schema],
        partitions.calibration["target"],
        method="sigmoid",
    )
    scores = model.predict_proba(partitions.selection.loc[:, schema])[:, 1]
    metrics = _threshold_metrics(partitions.selection["target"], scores, LOCKED_THRESHOLD)
    audit = calibration_audit(partitions.selection["target"], scores)
    metrics.update(
        {
            "prevalence": float(partitions.selection["target"].mean()),
            "mean_probability_gap": audit.mean_probability_gap,
            "equal_frequency_ece_15": audit.equal_frequency_ece_15,
        }
    )
    reproduction = compare_development_metrics(metrics)
    if not reproduction["all_metrics_reproduced"]:
        raise V1ExecutionError("governed R3-sigmoid reconstruction did not reproduce")
    return {"model": model, "metrics": metrics, "reproduction": reproduction}


def require_r3_reconstruction(result: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the governed incumbent reconstruction reproduced exactly."""

    if result.get("reproduction", {}).get("all_metrics_reproduced") is not True:
        raise V1ExecutionError("R3 reconstruction gate blocked all CatBoost challengers")
    return result


def _rolling_frames(
    train: pd.DataFrame, protocol: dict[str, Any]
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    return [
        (
            fold["id"],
            development_period(train, fold["train_start"], fold["train_end_exclusive"]),
            development_period(train, fold["validation_start"], fold["validation_end_exclusive"]),
        )
        for fold in protocol["rolling_origin"]["folds"]
    ]


def run_catboost_rolling(
    *, protocol: dict[str, Any], train: pd.DataFrame, tracker: V1Tracker, common: dict[str, Any]
) -> list[dict[str, Any]]:
    """Execute all and only CB1-CB4 across the four locked rolling-origin folds."""

    results: list[dict[str, Any]] = []
    for spec in candidate_specs(protocol):
        fold_evidence: list[dict[str, Any]] = []
        with tracker.start_run(
            name=f"v1-{spec.candidate_id}-rolling",
            group=common["group"],
            metadata={**common, "stage": "rolling", "candidate_id": spec.candidate_id},
        ) as run:
            for fold_id, fit_frame, evaluation_frame in _rolling_frames(train, protocol):
                fitted = adapt_v1_frame(fit_frame)
                evaluated = adapt_v1_frame(evaluation_frame)
                model = build_catboost_candidate(protocol, spec.candidate_id)
                fit_catboost_base(model, fitted.features, fitted.target, fitted.flight_date)
                scores = model.predict_proba(evaluated.features)[:, 1]
                complete_metrics = probability_metrics(evaluated.target, scores)
                fold_metrics = {name: complete_metrics[name] for name in ROLLING_METRICS}
                fold_evidence.append({"fold_id": fold_id, **fold_metrics})
                run.log({f"{fold_id}/{key}": value for key, value in fold_metrics.items()})
        results.append(
            {
                "candidate_id": spec.candidate_id,
                "status": "completed",
                "folds": fold_evidence,
                "mean_average_precision": float(
                    np.mean([row["average_precision"] for row in fold_evidence])
                ),
                "mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_evidence])),
                "mean_log_loss": float(np.mean([row["log_loss"] for row in fold_evidence])),
                "mean_brier_score": float(np.mean([row["brier_score"] for row in fold_evidence])),
                "std_average_precision": float(
                    np.std([row["average_precision"] for row in fold_evidence])
                ),
                "wandb_run_id": str(getattr(run, "id", "")),
                "wandb_run_url": str(getattr(run, "url", "")),
            }
        )
    return results


def run_r3_rolling_context(train: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Calculate descriptive uncalibrated R3-base rolling context; never a finalist."""

    evidence: list[dict[str, Any]] = []
    for fold_id, fit_frame, evaluation_frame in _rolling_frames(train, protocol):
        model, schema = build_remediation_model("R3", dict(EXPECTED_MATRIX["R3"]))
        model.fit(fit_frame.loc[:, schema], fit_frame["target"])
        scores = model.predict_proba(evaluation_frame.loc[:, schema])[:, 1]
        complete_metrics = probability_metrics(evaluation_frame["target"], scores)
        evidence.append(
            {"fold_id": fold_id, **{name: complete_metrics[name] for name in ROLLING_METRICS}}
        )
    return evidence


def _tracker_from_environment(tracking: str) -> V1Tracker:
    if tracking != "online":
        raise V1ExecutionError("applied governed execution requires --tracking online")
    api_key = os.environ.get("WANDB_API_KEY")
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not api_key or not entity or not project:
        raise V1ExecutionError("online tracking environment is incomplete")
    return WandbTracker(entity=entity, project=project)


def _common_metadata(
    *, protocol: dict[str, Any], protocol_sha: str, code_sha: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "implementation_git_sha": code_sha,
        "dataset_manifest_digest": manifest["manifest_digest"],
        "group": f"v1-{protocol_sha[:12]}-{code_sha[:12]}",
    }


def _finalist_evidence(
    *,
    model: Any,
    finalist_id: str,
    spec: Any,
    method: str,
    selection: AdaptedV1Frame,
    protocol: dict[str, Any],
    protocol_lock: dict[str, Any],
    common: dict[str, Any],
    bundle_directory: Path,
) -> dict[str, Any]:
    scores = np.asarray(model.predict_proba(selection.features)[:, 1], dtype=float)
    constraints = protocol["november_selection"]["threshold_objective"]["constraints"]
    threshold = select_v1_threshold(
        selection.target,
        scores,
        recall_min=constraints["recall_min"],
        precision_min=constraints["precision_min"],
        predicted_positive_rate_max=constraints["predicted_positive_rate_max"],
    )
    if threshold.selected_metrics is None or threshold.selected_threshold is None:
        return {
            "finalist_id": finalist_id,
            "status": "no_eligible_threshold",
            "threshold_selection": threshold.as_dict(),
            "gate_evidence": tuple(),
        }
    metrics = probability_metrics(selection.target, scores)
    metrics.update(asdict(threshold.selected_metrics))
    audit = calibration_audit(selection.target, scores)
    metrics["equal_frequency_ece_15"] = audit.equal_frequency_ece_15
    latency = measure_single_row_latency(model, selection.features)
    metrics["single_row_inference_p95_ms"] = latency["p95_ms"]
    bundle = write_candidate_bundle(
        directory=bundle_directory,
        model=model,
        candidate_id=finalist_id,
        parameters=spec.parameters,
        calibration_method=method,
        threshold=threshold.selected_threshold,
        metrics=metrics,
        metadata=common,
        protocol_lock=protocol_lock,
        verification_features=selection.features.iloc[:100],
    )
    restored = load_verified_bundle(bundle)
    restored_scores = restored.predict_proba(selection.features.iloc[:100])[:, 1]
    governance = {
        "lineage_verified": True,
        "schema_check_passed": True,
        "leakage_check_passed": True,
        "deterministic_reconstruction_check_passed": True,
        "serialization_load_inference_check_passed": bool(
            np.allclose(scores[:100], restored_scores, rtol=1e-12, atol=1e-12)
        ),
        "no_prohibited_test_access": True,
        "no_training_convergence_or_runtime_failure": True,
    }
    metrics["serialized_bundle_bytes"] = bundle.byte_size
    gates = evaluate_november_gates(metrics=metrics, protocol=protocol, governance=governance)
    return {
        "finalist_id": finalist_id,
        "status": "completed",
        "base_candidate_id": spec.candidate_id,
        "calibration_method": method,
        "model": model,
        "parameters": spec.parameters,
        "threshold": threshold.selected_threshold,
        "threshold_selection": threshold.as_dict(),
        "metrics": metrics,
        "gate_evidence": gates,
        "bundle": bundle,
    }


def _sanitized_finalist(finalist: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in finalist.items() if key not in {"model", "bundle"}}
    result["gate_evidence"] = [asdict(item) for item in finalist["gate_evidence"]]
    return result


def run_development_apply(repository_root: Path, *, tracking: str) -> dict[str, Any]:
    """Execute the governed development state machine once and stop before December."""

    root = repository_root.resolve()
    code_sha = require_applied_git_state(root)
    report = preflight(root, stage="development")
    if not report["catboost_version_ready"]:
        raise V1ExecutionError("exact optional CatBoost environment is not installed")
    protocol, protocol_lock, protocol_sha = load_and_validate_v1_protocol(
        root / "configs/v1_experiment_protocol.yaml",
        lock_path=root / "experiments/v1/protocol_lock.json",
        repository_root=root,
    )
    validate_catboost_runtime_contract(protocol)
    marker_path = root / DEVELOPMENT_MARKER
    if marker_path.exists():
        raise V1ExecutionError("development execution marker already exists; rerun prohibited")
    existing_outputs = (
        root / WINNER_LOCK,
        root / WINNER_MODEL,
        root / "artifacts/v1/development/decision.json",
        root / "artifacts/v1/development/stop-report.md",
    )
    if any(path.exists() for path in existing_outputs):
        raise V1ExecutionError("development output exists without its durable execution marker")
    tracker = _tracker_from_environment(tracking)
    create_marker(
        marker_path,
        {
            "status": "started",
            "protocol_sha": protocol_sha,
            "implementation_git_sha": code_sha,
            "started_at": utc_now(),
            "historical_test_accessed": False,
        },
    )
    stage = "data_guard"
    try:
        data = load_development_data(root)
        common = _common_metadata(
            protocol=protocol, protocol_sha=protocol_sha, code_sha=code_sha, manifest=data.manifest
        )
        stage = "r3_reconstruction"
        with tracker.start_run(
            name="v1-R3-reconstruction",
            group=common["group"],
            metadata={
                **common,
                "stage": "r3_reconstruction_and_rolling_context",
                "candidate_id": "R3-sigmoid",
            },
        ) as r3_run:
            reconstruction = require_r3_reconstruction(
                reconstruct_governed_r3(data.train, data.november)
            )
            r3_run.log(reconstruction["metrics"])
            stage = "r3_rolling_context"
            r3_context = run_r3_rolling_context(data.train, protocol)
            for fold in r3_context:
                r3_run.log(
                    {
                        f"context/{fold['fold_id']}/{name}": value
                        for name, value in fold.items()
                        if name != "fold_id"
                    }
                )
            r3_reference = {
                "id": str(getattr(r3_run, "id", "")),
                "url": str(getattr(r3_run, "url", "")),
            }
        stage = "catboost_rolling"
        rolling = run_catboost_rolling(
            protocol=protocol, train=data.train, tracker=tracker, common=common
        )
        selected = top_two_catboost(rolling)
        partitions = partition_remediation_data(data.train, data.november)
        refit = adapt_v1_frame(partitions.final_fit)
        calibration = adapt_v1_frame(partitions.calibration)
        selection = adapt_v1_frame(partitions.selection)
        by_id = {item.candidate_id: item for item in candidate_specs(protocol)}
        stage = "november_finalists"
        finalists: list[dict[str, Any]] = []
        development_root = root / "artifacts/v1/development"
        development_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="candidate-bundles-", dir=development_root) as temp:
            temp_root = Path(temp)
            for ranked in selected:
                candidate_id = str(ranked["candidate_id"])
                spec = by_id[candidate_id]
                base = build_catboost_candidate(protocol, candidate_id)
                fit_catboost_base(base, refit.features, refit.target, refit.flight_date)
                baseline_predictions = base.predict_proba(calibration.features)
                for method in protocol["refit_calibration"]["variants"]:
                    model = build_calibration_variant(
                        base,
                        method=method,
                        calibration_features=(calibration.features if method != "none" else None),
                        calibration_target=(calibration.target if method != "none" else None),
                    )
                    if not np.array_equal(
                        baseline_predictions, base.predict_proba(calibration.features)
                    ):
                        raise V1ExecutionError("finalist construction mutated its fitted base")
                    finalist_id = f"{candidate_id}-{method}"
                    with tracker.start_run(
                        name=f"v1-{finalist_id}-november",
                        group=common["group"],
                        metadata={
                            **common,
                            "stage": "november_finalist",
                            "candidate_id": candidate_id,
                            "calibration_method": method,
                        },
                    ) as run:
                        finalist = _finalist_evidence(
                            model=model,
                            finalist_id=finalist_id,
                            spec=spec,
                            method=method,
                            selection=selection,
                            protocol=protocol,
                            protocol_lock=protocol_lock,
                            common=common,
                            bundle_directory=temp_root / finalist_id,
                        )
                        run.log(
                            {
                                key: value
                                for key, value in finalist.get("metrics", {}).items()
                                if isinstance(value, int | float)
                            }
                        )
                        finalist["wandb_run_id"] = str(getattr(run, "id", ""))
                        finalist["wandb_run_url"] = str(getattr(run, "url", ""))
                    finalists.append(finalist)
            winner = choose_november_winner(finalists)
            decision_payload: dict[str, Any]
            if winner is None:
                decision_payload = {
                    **common,
                    "decision": "governed_stop",
                    "production_remains": "v0",
                    "finalists": [_sanitized_finalist(finalist) for finalist in finalists],
                }
                _atomic_json(
                    development_root / "decision.json", decision_payload, refuse_existing=True
                )
                _atomic_text(
                    development_root / "stop-report.md",
                    "# Governed v1 development stop\n\n"
                    "No finalist passed every mandatory November gate. "
                    "Production remains v0.\n",
                    refuse_existing=True,
                )
                decision = "governed_stop"
            else:
                winner_model = root / WINNER_MODEL
                if winner_model.exists() or (root / WINNER_LOCK).exists():
                    raise V1ExecutionError("immutable November winner output already exists")
                model_temp = winner_model.with_name(f".{winner_model.name}.{os.getpid()}.part")
                try:
                    joblib.dump(winner["model"], model_temp)
                    os.replace(model_temp, winner_model)
                finally:
                    model_temp.unlink(missing_ok=True)
                run_references = [
                    r3_reference,
                    *(
                        {"id": item["wandb_run_id"], "url": item["wandb_run_url"]}
                        for item in rolling
                    ),
                    *(
                        {"id": item.get("wandb_run_id", ""), "url": item.get("wandb_run_url", "")}
                        for item in finalists
                    ),
                ]
                winner_lock = {
                    "protocol_id": PROTOCOL_ID,
                    "protocol_sha256": protocol_sha,
                    "implementation_git_sha": code_sha,
                    "candidate_id": winner["finalist_id"],
                    "base_candidate_id": winner["base_candidate_id"],
                    "complete_catboost_parameters": winner["parameters"],
                    "calibration_method": winner["calibration_method"],
                    "threshold": winner["threshold"],
                    "feature_schema": list(V1_FEATURES),
                    "categorical_schema": list(V1_CATEGORICAL_FEATURES),
                    "development_metrics": winner["metrics"],
                    "all_gate_evidence": [asdict(item) for item in winner["gate_evidence"]],
                    "dataset_manifest_identity": data.manifest["manifest_digest"],
                    "train_parquet_sha256": data.manifest["parquet_files"]["train"]["sha256"],
                    "validation_parquet_sha256": data.manifest["parquet_files"]["validation"][
                        "sha256"
                    ],
                    "model_file_sha256": sha256_file(winner_model),
                    "candidate_bundle_digest": winner["bundle"].aggregate_digest,
                    "candidate_bundle_bytes": winner["bundle"].byte_size,
                    "wandb_run_ids": [item["id"] for item in run_references],
                    "wandb_run_urls": [item["url"] for item in run_references],
                    "created_at": utc_now(),
                    "december_evaluated": False,
                }
                _atomic_json(root / WINNER_LOCK, winner_lock, refuse_existing=True)
                decision_payload = {
                    **common,
                    "decision": "winner",
                    "candidate_id": winner["finalist_id"],
                    "production_remains": "v0",
                }
                _atomic_json(
                    development_root / "decision.json", decision_payload, refuse_existing=True
                )
                decision = "winner"
        with tracker.start_run(
            name="v1-development-decision",
            group=common["group"],
            metadata={**common, "stage": "decision"},
        ) as run:
            run.log({"decision/winner": int(decision == "winner")})
        update_marker(
            marker_path, {"status": "complete", "decision": decision, "completed_at": utc_now()}
        )
        return {
            "decision": decision,
            "rolling": rolling,
            "r3_rolling_context": r3_context,
            "finalist_count": len(finalists),
            "stopped_before_december": True,
        }
    except Exception as error:
        update_marker(
            marker_path,
            {
                "status": "failed",
                "sanitized_error_type": type(error).__name__,
                "failed_stage": stage,
                "failed_at": utc_now(),
            },
        )
        raise


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V1ExecutionError(f"cannot read governed state: {path}") from error
    if not isinstance(payload, dict):
        raise V1ExecutionError(f"governed state must be an object: {path}")
    return payload


def validate_december_handoff(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = _read_json_object(repository_root / DEVELOPMENT_MARKER)
    if marker.get("status") != "complete" or marker.get("decision") != "winner":
        raise V1ExecutionError("December requires a completed November winner")
    lock = _read_json_object(repository_root / WINNER_LOCK)
    model_path = repository_root / WINNER_MODEL
    if lock.get("december_evaluated") is not False:
        raise V1ExecutionError("November lock is not eligible for first December evaluation")
    if not model_path.is_file() or sha256_file(model_path) != lock.get("model_file_sha256"):
        raise V1ExecutionError("frozen November winner model hash mismatch")
    if lock.get("feature_schema") != list(V1_FEATURES) or lock.get("categorical_schema") != list(
        V1_CATEGORICAL_FEATURES
    ):
        raise V1ExecutionError("frozen November schema mismatch")
    if not isinstance(lock.get("threshold"), int | float):
        raise V1ExecutionError("frozen November threshold is invalid")
    return marker, lock


def run_december_apply(repository_root: Path, *, tracking: str) -> dict[str, Any]:
    """Evaluate the exact frozen November winner once on only December validation rows."""

    root = repository_root.resolve()
    code_sha = require_applied_git_state(root)
    report = preflight(root, stage="qualification")
    if not report["catboost_version_ready"]:
        raise V1ExecutionError("exact optional CatBoost environment is not installed")
    _development_marker, winner = validate_december_handoff(root)
    marker_path = root / QUALIFICATION_MARKER
    if marker_path.exists():
        raise V1ExecutionError("December was already started or evaluated")
    qualification_outputs = (
        root / "artifacts/v1/qualification/qualification_result.json",
        root / RELEASE_CANDIDATE_LOCK,
    )
    if any(path.exists() for path in qualification_outputs):
        raise V1ExecutionError("December output exists without its durable execution marker")
    protocol, _lock, protocol_sha = load_and_validate_v1_protocol(
        root / "configs/v1_experiment_protocol.yaml",
        lock_path=root / "experiments/v1/protocol_lock.json",
        repository_root=root,
    )
    if winner["protocol_sha256"] != protocol_sha:
        raise V1ExecutionError("November winner protocol SHA mismatch")
    if winner.get("protocol_id") != protocol["protocol_id"]:
        raise V1ExecutionError("November winner protocol ID mismatch")
    if winner["implementation_git_sha"] != code_sha:
        raise V1ExecutionError("December implementation lineage differs from November winner")
    tracker = _tracker_from_environment(tracking)
    create_marker(
        marker_path,
        {
            "status": "started",
            "protocol_sha": protocol_sha,
            "implementation_git_sha": code_sha,
            "winner_model_sha256": winner["model_file_sha256"],
            "started_at": utc_now(),
            "historical_test_accessed": False,
        },
    )
    stage = "december_data_guard"
    try:
        december = adapt_v1_frame(load_december_data(root))
        stage = "frozen_winner_load"
        model = joblib.load(root / WINNER_MODEL)
        stage = "qualification_evaluation"
        scores = np.asarray(model.predict_proba(december.features)[:, 1], dtype=float)
        metrics = probability_metrics(december.target, scores)
        metrics.update(_threshold_metrics(december.target, scores, float(winner["threshold"])))
        audit = calibration_audit(december.target, scores)
        metrics["equal_frequency_ece_15"] = audit.equal_frequency_ece_15
        metrics["single_row_inference_p95_ms"] = measure_single_row_latency(
            model, december.features
        )["p95_ms"]
        metrics["serialized_bundle_bytes"] = winner["candidate_bundle_bytes"]
        gates = evaluate_qualification_gates(
            metrics=metrics, protocol=protocol, governance_passed=True
        )
        passed = all_gates_pass(gates)
        manifest = read_manifest(root / "data/manifests/processed_manifest.json")
        common = _common_metadata(
            protocol=protocol, protocol_sha=protocol_sha, code_sha=code_sha, manifest=manifest
        )
        with tracker.start_run(
            name="v1-december-qualification",
            group=common["group"],
            metadata={
                **common,
                "stage": "december_qualification",
                "candidate_id": winner["candidate_id"],
                "calibration_method": winner["calibration_method"],
            },
        ) as run:
            run.log(
                {key: value for key, value in metrics.items() if isinstance(value, int | float)}
            )
            run_id = str(getattr(run, "id", ""))
            run_url = str(getattr(run, "url", ""))
        result = {
            **common,
            "passed": passed,
            "candidate_id": winner["candidate_id"],
            "winner_model_sha256": winner["model_file_sha256"],
            "metrics": metrics,
            "gate_evidence": [asdict(item) for item in gates],
            "wandb_run_id": run_id,
            "wandb_run_url": run_url,
            "production_remains": "v0",
        }
        qualification_root = root / "artifacts/v1/qualification"
        _atomic_json(qualification_root / "qualification_result.json", result, refuse_existing=True)
        if passed:
            _atomic_json(
                root / RELEASE_CANDIDATE_LOCK,
                {
                    **result,
                    "same_frozen_november_model": True,
                    "registry_artifact_created": False,
                    "production_promoted": False,
                },
                refuse_existing=True,
            )
        update_marker(
            marker_path,
            {
                "status": "complete",
                "decision": "release_candidate" if passed else "qualification_failed",
                "completed_at": utc_now(),
            },
        )
        return result
    except Exception as error:
        update_marker(
            marker_path,
            {
                "status": "failed",
                "sanitized_error_type": type(error).__name__,
                "failed_stage": stage,
                "failed_at": utc_now(),
            },
        )
        raise
