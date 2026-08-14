"""Governed v2 preflight, one-time development, and separate December qualification."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.modeling.calibration import calibration_audit
from flight_delay.modeling.v1_selection import (
    GateEvidence,
    evaluate_qualification_gates,
    probability_metrics,
)
from flight_delay.modeling.v2.data import load_december_features, prepare_development_data
from flight_delay.modeling.v2.features import HistoricalState
from flight_delay.modeling.v2.models import (
    CATBOOST_VERSION,
    LIGHTGBM_VERSION,
    installed_version,
    require_versions,
)
from flight_delay.modeling.v2.protocol import (
    CATEGORICAL_FEATURES,
    PROTOCOL_COMMIT_SHA,
    V2_FEATURES,
    load_and_validate_v2_protocol,
    sha256_file,
)
from flight_delay.modeling.v2.tracking import WandbTracker
from flight_delay.modeling.v2.workflow import (
    bundle_evidence,
    run_refit_and_november,
    run_screening_and_cpu_confirmation,
    sanitized_workflow_result,
)

DEVELOPMENT_MARKER = Path("artifacts/v2/development/execution_marker.json")
DECISION_PATH = Path("artifacts/v2/development/decision.json")
STATE_PATH = Path("artifacts/v2/development/historical_state.json")
WINNER_LOCK = Path("artifacts/v2/development/winner_lock.json")
WINNER_MODEL = Path("artifacts/v2/development/winner.joblib")
QUALIFICATION_MARKER = Path("artifacts/v2/qualification/execution_marker.json")
QUALIFICATION_RESULT = Path("artifacts/v2/qualification/qualification_result.json")


class V2ExecutionError(RuntimeError):
    """Raised when a durable v2 execution invariant fails."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def implementation_git_sha(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def require_merged_applied_state(root: Path) -> str:
    """Prevent any applied fit from a feature branch or dirty/unreviewed checkout."""

    if _git(root, "status", "--porcelain"):
        raise V2ExecutionError("applied v2 execution requires a clean Git worktree")
    if _git(root, "branch", "--show-current") != "main":
        raise V2ExecutionError("applied v2 execution is locked until the reviewed PR is on main")
    try:
        _git(root, "merge-base", "--is-ancestor", PROTOCOL_COMMIT_SHA, "HEAD")
    except subprocess.CalledProcessError as error:
        raise V2ExecutionError("the frozen v2 protocol commit must be an ancestor") from error
    return implementation_git_sha(root)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ExecutionError(f"cannot read governed state: {path}") from error
    if not isinstance(payload, dict):
        raise V2ExecutionError(f"governed state must be an object: {path}")
    return payload


def _atomic_bytes(path: Path, encoded: bytes, *, refuse_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise V2ExecutionError(f"immutable output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        temporary.write_bytes(encoded)
        if refuse_existing and path.exists():
            raise V2ExecutionError(f"immutable output already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any], *, refuse_existing: bool) -> None:
    _atomic_bytes(path, canonical_json_bytes(payload) + b"\n", refuse_existing=refuse_existing)


def create_marker(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, payload, refuse_existing=True)


def update_marker(path: Path, updates: dict[str, Any]) -> None:
    payload = _read_json(path)
    payload.update(updates)
    _atomic_json(path, payload, refuse_existing=False)


def validate_dependency_isolation(root: Path) -> dict[str, Any]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected_extra = [f"lightgbm=={LIGHTGBM_VERSION}", f"catboost=={CATBOOST_VERSION}"]
    if project["project"]["optional-dependencies"].get("v2") != expected_extra:
        raise V2ExecutionError("pyproject v2 extra differs from the exact modeling-only pins")
    constraints = {
        line.strip()
        for line in (root / "requirements-v2.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if constraints != set(expected_extra):
        raise V2ExecutionError("requirements-v2.lock differs from the exact v2 constraints")
    dockerfiles = (
        root / "services/api/Dockerfile",
        root / "services/user_ui/Dockerfile",
        root / "services/monitor_ui/Dockerfile",
    )
    for path in dockerfiles:
        source = path.read_text(encoding="utf-8").casefold()
        if any(token in source for token in ("catboost", "lightgbm", "requirements-v2", ".[v2]")):
            raise V2ExecutionError(f"runtime image includes a modeling-only dependency: {path}")
    return {
        "v2_extra": expected_extra,
        "v2_constraints": sorted(constraints),
        "runtime_images_install_base_only": True,
    }


def validate_production_v0(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    incumbent = protocol["incumbent"]
    release = _read_json(root / "release/release_decision.json")
    deployment = _read_json(root / "deploy/deployment_manifest.json")["model"]
    expected_release = {
        "serving_alias": incumbent["serving_alias"],
        "registry_version": incumbent["registry_version"],
        "registry_digest": incumbent["registry_digest"],
        "bundle_digest": incumbent["bundle_sha256"],
    }
    expected_deployment = {
        "serving_alias": incumbent["serving_alias"],
        "registry_version": incumbent["registry_version"],
        "registry_digest": incumbent["registry_digest"],
        "release_bundle_digest": incumbent["bundle_sha256"],
        "classification_threshold": incumbent["threshold"],
    }
    if any(release.get(name) != value for name, value in expected_release.items()):
        raise V2ExecutionError("committed release no longer preserves production v0")
    if any(deployment.get(name) != value for name, value in expected_deployment.items()):
        raise V2ExecutionError("deployment manifest no longer preserves production v0")
    return {"release": expected_release, "deployment": expected_deployment, "unchanged": True}


def preflight(
    repository_root: Path, *, stage: Literal["development", "qualification"]
) -> dict[str, Any]:
    """Validate v2 statically without parquet, model-runtime imports, W&B, or network."""

    root = repository_root.resolve()
    protocol, lock, protocol_sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    manifest = read_manifest(root / protocol["dependencies"]["processed_dataset_manifest"]["path"])
    report: dict[str, Any] = {
        "mode": "dry-run/preflight",
        "stage": stage,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "protocol_lock_valid": lock["protocol_sha256"] == protocol_sha,
        "implementation_git_sha": implementation_git_sha(root),
        "dependency_isolation": validate_dependency_isolation(root),
        "production_v0": validate_production_v0(root, protocol),
        "dataset_manifest_digest": manifest["manifest_digest"],
        "lightgbm_required_version": LIGHTGBM_VERSION,
        "lightgbm_installed_version": installed_version("lightgbm"),
        "catboost_required_version": CATBOOST_VERSION,
        "catboost_installed_version": installed_version("catboost"),
        "parquet_opened": False,
        "december_opened": False,
        "historical_test_accessed": False,
        "network_contacted": False,
        "wandb_imported": "wandb" in sys.modules,
        "lightgbm_runtime_imported": "lightgbm" in sys.modules,
        "catboost_runtime_imported": "catboost" in sys.modules,
        "production_v0_mutated": False,
        "registry_mutated": False,
        "aws_contacted": False,
    }
    if stage == "development":
        report.update(
            {
                "lightgbm_candidate_count": 16,
                "catboost_candidate_count": 12,
                "gpu_screening_sequential": True,
                "cpu_confirmation_authoritative": True,
                "historical_feature_count": 17,
                "total_feature_count": 37,
                "november_state_as_of": "2025-10-31",
                "stops_before_december": True,
                "execution_marker_exists": (root / DEVELOPMENT_MARKER).exists(),
            }
        )
    else:
        report.update(
            {
                "requires_frozen_november_winner": True,
                "winner_lock_exists": (root / WINNER_LOCK).is_file(),
                "winner_model_exists": (root / WINNER_MODEL).is_file(),
                "historical_state_exists": (root / STATE_PATH).is_file(),
                "qualification_marker_exists": (root / QUALIFICATION_MARKER).exists(),
                "refitting_permitted": False,
                "recalibration_permitted": False,
                "threshold_change_permitted": False,
                "historical_state_update_permitted": False,
            }
        )
    return report


def _online_tracker() -> WandbTracker:
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not entity or not project:
        raise V2ExecutionError("online v2 execution requires WANDB_ENTITY and WANDB_PROJECT")
    return WandbTracker(entity=entity, project=project)


def _common_metadata(
    *, protocol_sha: str, code_sha: str, lineage: dict[str, Any]
) -> dict[str, Any]:
    return {
        "group": f"v2-{protocol_sha}-{code_sha}",
        "protocol_sha256": protocol_sha,
        "implementation_git_sha": code_sha,
        "hardware_identity": platform.platform(),
        "cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "screening_backends": {"lightgbm": "CPU", "catboost": "GPU:0"},
        "cpu_confirmation_backend": "CPU",
        "catboost_version": CATBOOST_VERSION,
        "lightgbm_version": LIGHTGBM_VERSION,
        "feature_state_digest": lineage["november_state_sha256"],
        "dataset_lineage": lineage,
    }


def _reconstruct_r3(train: Any, november: Any) -> dict[str, Any]:
    """Lazily import the incumbent reconstruction only inside applied execution."""

    from flight_delay.modeling.v1_execution import (
        reconstruct_governed_r3,
        require_r3_reconstruction,
    )

    return require_r3_reconstruction(reconstruct_governed_r3(train, november))


def _write_winner_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V2ExecutionError("immutable winner model already exists")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(model, temporary)
        if path.exists():
            raise V2ExecutionError("immutable winner model already exists")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_development_apply(repository_root: Path, *, tracking: str) -> dict[str, Any]:
    """Run the one-time reviewed v2 development workflow and stop before December."""

    root = repository_root.resolve()
    if tracking != "online":
        raise V2ExecutionError("applied v2 development requires online governed tracking")
    code_sha = require_merged_applied_state(root)
    preflight(root, stage="development")
    require_versions()
    protocol, _lock, protocol_sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    outputs = (DEVELOPMENT_MARKER, DECISION_PATH, STATE_PATH, WINNER_LOCK, WINNER_MODEL)
    if any((root / path).exists() for path in outputs):
        raise V2ExecutionError("v2 development marker or output already exists")
    marker = root / DEVELOPMENT_MARKER
    create_marker(
        marker,
        {
            "status": "started",
            "protocol_sha": protocol_sha,
            "implementation_git_sha": code_sha,
            "started_at": utc_now(),
            "december_opened": False,
            "historical_test_accessed": False,
        },
    )
    stage = "data_guard"
    try:
        prepared = prepare_development_data(root)
        _atomic_bytes(root / STATE_PATH, prepared.november_state.to_bytes(), refuse_existing=True)
        metadata = _common_metadata(
            protocol_sha=protocol_sha, code_sha=code_sha, lineage=prepared.lineage
        )
        tracker = _online_tracker()
        stage = "r3_reconstruction"
        r3 = _reconstruct_r3(prepared.raw_train, prepared.raw_november)
        stage = "screening_and_cpu_confirmation"
        search = run_screening_and_cpu_confirmation(
            protocol=protocol,
            transformed=prepared.search,
            tracker=tracker,
            metadata=metadata,
        )
        stage = "cpu_refit_calibration_november"
        november = run_refit_and_november(
            prepared=prepared,
            protocol=protocol,
            advanced=search["advanced_to_refit"],
            tracker=tracker,
            metadata=metadata,
            r3_reconstruction_passed=r3["reproduction"]["all_metrics_reproduced"],
        )
        sanitized = sanitized_workflow_result(november)
        decision = {
            **metadata,
            "decision": november["decision"],
            "production_remains": "v0",
            "stopped_before_december": True,
            "r3_reconstruction": r3["reproduction"],
            "screening": search["screening"],
            "cpu_confirmation": search["cpu_confirmation"],
            "screening_cpu_differences": search["screening_cpu_differences"],
            "advanced_to_refit": [row["candidate_id"] for row in search["advanced_to_refit"]],
            "november": sanitized,
        }
        _atomic_json(root / DECISION_PATH, decision, refuse_existing=True)
        winner = november["winner"]
        if winner is not None:
            _write_winner_model(root / WINNER_MODEL, winner["model"])
            _atomic_json(
                root / WINNER_LOCK,
                {
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_sha,
                    "implementation_git_sha": code_sha,
                    "finalist_id": winner["finalist_id"],
                    "family": winner["family"],
                    "base_candidate_id": winner["base_candidate_id"],
                    "candidate_identity": winner["candidate_identity"],
                    "calibration_method": winner["calibration_method"],
                    "threshold": winner["metrics"]["threshold"],
                    "feature_schema": list(V2_FEATURES),
                    "categorical_schema": list(CATEGORICAL_FEATURES),
                    "historical_state_sha256": prepared.november_state.sha256,
                    "historical_state_as_of": prepared.november_state.as_of.isoformat(),
                    "model_sha256": sha256_file(root / WINNER_MODEL),
                    "serialized_bundle_bytes": winner["bundle"]["serialized_bundle_bytes"],
                    "development_metrics": winner["metrics"],
                    "gate_evidence": [asdict(item) for item in winner["gate_evidence"]],
                    "december_evaluated": False,
                    "production_remains": "v0",
                },
                refuse_existing=True,
            )
        update_marker(
            marker,
            {
                "status": "complete",
                "decision": november["decision"],
                "completed_at": utc_now(),
            },
        )
        return decision
    except Exception as error:
        update_marker(
            marker,
            {
                "status": "failed",
                "failed_stage": stage,
                "sanitized_error_type": type(error).__name__,
                "failed_at": utc_now(),
            },
        )
        raise


def validate_december_handoff(root: Path) -> tuple[dict[str, Any], HistoricalState]:
    marker = _read_json(root / DEVELOPMENT_MARKER)
    if marker.get("status") != "complete" or marker.get("decision") != "winner":
        raise V2ExecutionError("December requires a completed frozen November winner")
    winner = _read_json(root / WINNER_LOCK)
    if winner.get("december_evaluated") is not False:
        raise V2ExecutionError("December has already been evaluated")
    if winner.get("historical_state_as_of") != "2025-10-31":
        raise V2ExecutionError("winner does not reference the frozen October-31 state")
    if not (root / WINNER_MODEL).is_file() or sha256_file(root / WINNER_MODEL) != winner.get(
        "model_sha256"
    ):
        raise V2ExecutionError("frozen winner model hash mismatch")
    state_bytes = (root / STATE_PATH).read_bytes()
    state = HistoricalState.from_bytes(state_bytes)
    if state.sha256 != winner.get("historical_state_sha256"):
        raise V2ExecutionError("frozen historical-state hash mismatch")
    return winner, state


def _fixed_threshold_metrics(labels: Any, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "predicted_positive_rate": float(predicted.mean()),
    }


def run_december_apply(repository_root: Path, *, tracking: str) -> dict[str, Any]:
    """Evaluate the exact November winner once, without fit, calibration, or state update."""

    root = repository_root.resolve()
    if tracking != "online":
        raise V2ExecutionError("applied December qualification requires online governed tracking")
    code_sha = require_merged_applied_state(root)
    preflight(root, stage="qualification")
    require_versions()
    winner, state = validate_december_handoff(root)
    if winner.get("implementation_git_sha") != code_sha:
        raise V2ExecutionError("December code lineage differs from the frozen November winner")
    if (root / QUALIFICATION_MARKER).exists() or (root / QUALIFICATION_RESULT).exists():
        raise V2ExecutionError("December qualification already started or produced output")
    protocol, _lock, protocol_sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    marker = root / QUALIFICATION_MARKER
    create_marker(
        marker,
        {
            "status": "started",
            "started_at": utc_now(),
            "protocol_sha": protocol_sha,
            "implementation_git_sha": code_sha,
            "historical_test_accessed": False,
        },
    )
    stage = "december_data_guard"
    try:
        features, target, _dates = load_december_features(root, state=state)
        stage = "frozen_winner_load"
        model = joblib.load(root / WINNER_MODEL)
        stage = "qualification_evaluation"
        scores = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        metrics = probability_metrics(target, scores)
        metrics.update(_fixed_threshold_metrics(target, scores, float(winner["threshold"])))
        audit = calibration_audit(target, scores)
        metrics["equal_frequency_ece_15"] = audit.equal_frequency_ece_15
        runtime = bundle_evidence(model, features, state)
        metrics.update(runtime)
        gates = list(
            evaluate_qualification_gates(metrics=metrics, protocol=protocol, governance_passed=True)
        )
        gates.append(
            GateEvidence(
                "historical_state_integrity_passed",
                "is True",
                runtime["historical_state_integrity_passed"],
                runtime["historical_state_integrity_passed"] is True,
            )
        )
        passed = all(item.passed for item in gates)
        tracker = _online_tracker()
        with tracker.start_run(
            name="v2-december-qualification",
            group=f"v2-{protocol_sha}-{code_sha}",
            metadata={
                "stage": "december_qualification",
                "candidate_id": winner["finalist_id"],
                "protocol_sha256": protocol_sha,
                "implementation_git_sha": code_sha,
                "historical_state_sha256": state.sha256,
                "refit_performed": False,
                "recalibration_performed": False,
                "threshold_changed": False,
                "historical_state_updated": False,
            },
        ) as run:
            run.log(
                {name: value for name, value in metrics.items() if isinstance(value, int | float)}
            )
            run_id = str(getattr(run, "id", ""))
            run_url = str(getattr(run, "url", ""))
        result = {
            "passed": passed,
            "candidate_id": winner["finalist_id"],
            "metrics": metrics,
            "gate_evidence": [asdict(item) for item in gates],
            "wandb_run_id": run_id,
            "wandb_run_url": run_url,
            "same_frozen_november_model": True,
            "same_frozen_october_31_state": True,
            "production_remains": "v0",
        }
        _atomic_json(root / QUALIFICATION_RESULT, result, refuse_existing=True)
        update_marker(
            marker,
            {
                "status": "complete",
                "decision": "qualified" if passed else "qualification_failed",
                "completed_at": utc_now(),
            },
        )
        return result
    except Exception as error:
        update_marker(
            marker,
            {
                "status": "failed",
                "failed_stage": stage,
                "sanitized_error_type": type(error).__name__,
                "failed_at": utc_now(),
            },
        )
        raise
