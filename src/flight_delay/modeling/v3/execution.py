"""Governed v3 preflight, runtime estimation, one-time development, and December qualification."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.data.prepare_v3 import (
    DECEMBER_AUTHORIZATION,
    V3_PROCESSED_MANIFEST,
    materialize_december_qualification_data,
)
from flight_delay.modeling.calibration import calibration_audit
from flight_delay.modeling.v1_selection import (
    GateEvidence,
    evaluate_qualification_gates,
    probability_metrics,
)
from flight_delay.modeling.v2.models import (
    CATBOOST_VERSION,
    LIGHTGBM_VERSION,
    installed_version,
    require_versions,
)
from flight_delay.modeling.v3.data import load_december_features, prepare_development_data
from flight_delay.modeling.v3.features import V3HistoricalState
from flight_delay.modeling.v3.protocol import (
    CANDIDATE_IDENTITY_IDS,
    CATEGORICAL_FEATURES,
    V3_FEATURES,
    load_and_validate_v3_protocol,
    sha256_file,
)
from flight_delay.modeling.v3.tracking import WandbTracker
from flight_delay.modeling.v3.workflow import (
    bundle_evidence,
    run_refit_and_november,
    run_screening_and_cpu_confirmation,
    sanitized_workflow_result,
)

PROTOCOL_COMMIT_SHA = "cbeab0cf2fb3506875971433ac087cba7cfa9158"

DEVELOPMENT_MARKER = Path("artifacts/v3/development/execution_marker.json")
DECISION_PATH = Path("artifacts/v3/development/decision.json")
STATE_PATH = Path("artifacts/v3/development/historical_state.json")
WINNER_LOCK = Path("artifacts/v3/development/winner_lock.json")
WINNER_MODEL = Path("artifacts/v3/development/winner.joblib")
QUALIFICATION_MARKER = Path("artifacts/v3/qualification/execution_marker.json")
QUALIFICATION_RESULT = Path("artifacts/v3/qualification/qualification_result.json")

# Conservative planning throughputs for this host (24 vCPU, RTX 3060) at the v3 48-feature width.
# These are engineering estimates, NOT measurements of a real v3 fit: no v3 model has been trained.
# The applied run logs true per-stage runtimes, which should replace these numbers afterwards.
# They only shape the advisory dry-run estimate and never influence governed behaviour.
THROUGHPUT_ROWS_PER_SECOND: dict[str, float] = {
    "lightgbm_cpu": 26000.0,
    "catboost_gpu": 41000.0,
    "catboost_cpu": 5200.0,
}


class V3ExecutionError(RuntimeError):
    """Raised when a durable v3 execution invariant fails."""


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
        raise V3ExecutionError("applied v3 execution requires a clean Git worktree")
    if _git(root, "branch", "--show-current") != "main":
        raise V3ExecutionError("applied v3 execution is locked until the reviewed PR is on main")
    try:
        _git(root, "merge-base", "--is-ancestor", PROTOCOL_COMMIT_SHA, "HEAD")
    except subprocess.CalledProcessError as error:
        raise V3ExecutionError("the frozen v3 protocol commit must be an ancestor") from error
    return implementation_git_sha(root)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V3ExecutionError(f"cannot read governed state: {path}") from error
    if not isinstance(payload, dict):
        raise V3ExecutionError(f"governed state must be an object: {path}")
    return payload


def _atomic_bytes(path: Path, encoded: bytes, *, refuse_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise V3ExecutionError(f"immutable output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        temporary.write_bytes(encoded)
        if refuse_existing and path.exists():
            raise V3ExecutionError(f"immutable output already exists: {path}")
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
    """Prove v3 adds no modeling dependency and the runtime images stay clean."""

    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected_extra = [f"lightgbm=={LIGHTGBM_VERSION}", f"catboost=={CATBOOST_VERSION}"]
    if project["project"]["optional-dependencies"].get("v2") != expected_extra:
        raise V3ExecutionError("pyproject v2 extra differs from the exact modeling-only pins")
    if "v3" in project["project"]["optional-dependencies"]:
        raise V3ExecutionError("v3 must reuse the existing modeling extra, not add a new one")
    constraints = {
        line.strip()
        for line in (root / "requirements-v2.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if constraints != set(expected_extra):
        raise V3ExecutionError("requirements-v2.lock differs from the exact modeling constraints")
    dockerfiles = (
        root / "services/api/Dockerfile",
        root / "services/user_ui/Dockerfile",
        root / "services/monitor_ui/Dockerfile",
    )
    for path in dockerfiles:
        source = path.read_text(encoding="utf-8").casefold()
        forbidden = ("catboost", "lightgbm", "requirements-v2", "requirements-v3", ".[v2]", ".[v3]")
        if any(token in source for token in forbidden):
            raise V3ExecutionError(f"runtime image includes a modeling-only dependency: {path}")
    return {
        "modeling_extra": expected_extra,
        "modeling_constraints": sorted(constraints),
        "v3_added_modeling_dependency": False,
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
        raise V3ExecutionError("committed release no longer preserves production v0")
    if any(deployment.get(name) != value for name, value in expected_deployment.items()):
        raise V3ExecutionError("deployment manifest no longer preserves production v0")
    return {"release": expected_release, "deployment": expected_deployment, "unchanged": True}


def _monthly_model_rows(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        str(row["month"]): int(row["model_eligible_rows"])
        for row in manifest.get("monthly_counts", [])
    }


def _months_between(start: str, end_exclusive: str) -> list[str]:
    months: list[str] = []
    year, month = int(start[:4]), int(start[5:7])
    end_year, end_month = int(end_exclusive[:4]), int(end_exclusive[5:7])
    while (year, month) < (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def estimate_applied_runtime(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    *,
    throughput: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate the governed campaign from real row counts and the frozen stage structure.

    This is advisory only: it reads the v3 processed manifest, never the parquet rows, and is
    reported before any applied execution so the overnight timebox can be checked in advance.
    """

    rates = {**THROUGHPUT_ROWS_PER_SECOND, **(throughput or {})}
    monthly = _monthly_model_rows(manifest)
    if not monthly:
        raise V3ExecutionError("runtime estimation requires v3 monthly row counts")
    cap = int(protocol["sampling"]["search_rows_per_month_max"])
    search_monthly = {month: min(cap, rows) for month, rows in monthly.items()}

    folds = protocol["rolling_origin"]["folds"]
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fit_months = _months_between(str(fold["fit_start"]), str(fold["fit_end_exclusive"]))
        evaluation_months = _months_between(
            str(fold["evaluation_start"]), str(fold["evaluation_end_exclusive"])
        )
        fold_rows.append(
            {
                "fold_id": fold["id"],
                "search_fit_rows": sum(search_monthly.get(month, 0) for month in fit_months),
                "evaluation_rows": sum(monthly.get(month, 0) for month in evaluation_months),
            }
        )
    search_fit_rows_per_candidate = sum(row["search_fit_rows"] for row in fold_rows)

    refit = protocol["advancement"]["full_refit"]
    refit_months = _months_between(str(refit["start"]), str(refit["end_exclusive"]))
    full_refit_rows = sum(monthly.get(month, 0) for month in refit_months)

    screening = protocol["advancement"]["screening"]
    stages = [
        {
            "stage": "screening_lightgbm_cpu",
            "identities": int(screening["lightgbm_identities"]),
            "rows_per_identity": search_fit_rows_per_candidate,
            "rate_key": "lightgbm_cpu",
        },
        {
            "stage": "screening_catboost_gpu",
            "identities": int(screening["catboost_identities"]),
            "rows_per_identity": search_fit_rows_per_candidate,
            "rate_key": "catboost_gpu",
        },
        {
            "stage": "cpu_confirmation_lightgbm",
            "identities": int(screening["top_per_family_to_cpu_confirmation"]),
            "rows_per_identity": search_fit_rows_per_candidate,
            "rate_key": "lightgbm_cpu",
        },
        {
            "stage": "cpu_confirmation_catboost",
            "identities": int(screening["top_per_family_to_cpu_confirmation"]),
            "rows_per_identity": search_fit_rows_per_candidate,
            "rate_key": "catboost_cpu",
        },
        {
            "stage": "full_refit_lightgbm_cpu",
            "identities": int(refit["lightgbm_bases"]),
            "rows_per_identity": full_refit_rows,
            "rate_key": "lightgbm_cpu",
        },
        {
            "stage": "full_refit_catboost_cpu",
            "identities": int(refit["catboost_bases"]),
            "rows_per_identity": full_refit_rows,
            "rate_key": "catboost_cpu",
        },
    ]
    for stage in stages:
        rows = int(stage["identities"]) * int(stage["rows_per_identity"])
        stage["total_fit_rows"] = rows
        stage["estimated_seconds"] = round(rows / rates[str(stage["rate_key"])], 1)

    fit_seconds = sum(float(stage["estimated_seconds"]) for stage in stages)
    # Feature transformation runs twice (search matrix and full-refit matrix) and the November
    # finalists re-score a fixed 15 variants; both are small next to the fits but are counted.
    transform_seconds = round((search_fit_rows_per_candidate + full_refit_rows) / 90000.0, 1)
    finalist_seconds = round(int(protocol["finalists"]["total"]) * 45.0, 1)
    total_seconds = fit_seconds + transform_seconds + finalist_seconds
    return {
        "basis": "v3 processed manifest row counts and the frozen stage structure",
        "advisory_only": True,
        "search_rows_per_month_cap": cap,
        "fold_rows": fold_rows,
        "search_fit_rows_per_candidate": search_fit_rows_per_candidate,
        "full_refit_rows": full_refit_rows,
        "throughput_rows_per_second": rates,
        "stages": stages,
        "estimated_fit_seconds": round(fit_seconds, 1),
        "estimated_transform_seconds": transform_seconds,
        "estimated_finalist_seconds": finalist_seconds,
        "estimated_total_seconds": round(total_seconds, 1),
        "estimated_total_hours": round(total_seconds / 3600.0, 2),
        "fits_overnight_budget": total_seconds <= 12 * 3600,
    }


def preflight(
    repository_root: Path, *, stage: Literal["development", "qualification"]
) -> dict[str, Any]:
    """Validate v3 statically without parquet rows, model runtimes, W&B, or network."""

    root = repository_root.resolve()
    protocol, lock, protocol_sha = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
        repository_root=root,
    )
    report: dict[str, Any] = {
        "mode": "dry-run/preflight",
        "stage": stage,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "protocol_lock_valid": lock["protocol_sha256"] == protocol_sha,
        "implementation_git_sha": implementation_git_sha(root),
        "dependency_isolation": validate_dependency_isolation(root),
        "production_v0": validate_production_v0(root, protocol),
        "v3_source_manifest_digest": read_manifest(
            root / protocol["dependencies"]["v3_source_manifest"]["path"]
        )["manifest_digest"],
        "lightgbm_required_version": LIGHTGBM_VERSION,
        "lightgbm_installed_version": installed_version("lightgbm"),
        "catboost_required_version": CATBOOST_VERSION,
        "catboost_installed_version": installed_version("catboost"),
        "parquet_opened": False,
        "december_opened": False,
        "historical_test_accessed": False,
        "january_may_2026_accessed": False,
        "network_contacted": False,
        "wandb_imported": "wandb" in sys.modules,
        "lightgbm_runtime_imported": "lightgbm" in sys.modules,
        "catboost_runtime_imported": "catboost" in sys.modules,
        "production_v0_mutated": False,
        "registry_mutated": False,
        "aws_contacted": False,
    }
    processed_manifest_path = root / V3_PROCESSED_MANIFEST
    if processed_manifest_path.is_file():
        processed = read_manifest(processed_manifest_path)
        report["v3_processed_manifest_digest"] = processed["manifest_digest"]
        report["v3_processed_december_decoded"] = processed.get("december_2025_decoded")
        report["runtime_estimate"] = estimate_applied_runtime(protocol, processed)
    else:
        report["v3_processed_manifest_digest"] = None
        report["runtime_estimate"] = None

    if stage == "development":
        report.update(
            {
                "candidate_identities": list(CANDIDATE_IDENTITY_IDS),
                "candidate_identity_count": len(CANDIDATE_IDENTITY_IDS),
                "weight_policies": [row["id"] for row in protocol["weight_policies"]["policies"]],
                "gpu_screening_sequential": True,
                "cpu_confirmation_authoritative": True,
                "total_feature_count": len(V3_FEATURES),
                "native_categorical_count": len(CATEGORICAL_FEATURES),
                "finalist_count": int(protocol["finalists"]["total"]),
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
                "candidate_switching_permitted": False,
            }
        )
    return report


def _online_tracker() -> WandbTracker:
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not entity or not project:
        raise V3ExecutionError("online v3 execution requires WANDB_ENTITY and WANDB_PROJECT")
    return WandbTracker(entity=entity, project=project)


def _common_metadata(
    *, protocol_sha: str, code_sha: str, lineage: dict[str, Any]
) -> dict[str, Any]:
    return {
        "group": f"v3-{protocol_sha}-{code_sha}",
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


def reconstruct_r3_control(
    repository_root: Path,
    *,
    loader: Callable[..., Any] | None = None,
    reconstructor: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce the frozen R3 incumbent from the ORIGINAL canonical v1/v2 control dataset.

    This is a control check on the incumbent, not a v3 challenger fit, so it must run on the exact
    data that historically produced the frozen R3 metrics. Feeding it the v3 population — a
    different year range at a different sampling density — would compare the incumbent against
    numbers it was never measured on and silently invalidate the gate.

    The canonical v1 loader is used deliberately: it validates the v1 protocol, verifies
    ``data/manifests/processed_manifest.json``, reads only ``train.parquet`` and
    ``validation.parquet``, and refuses ``test.parquet`` outright.
    """

    if loader is None:  # pragma: no cover - exercised through the injected loader in tests
        from flight_delay.modeling.v1_data import load_development_data as loader
    if reconstructor is None:  # pragma: no cover - real reconstruction only in applied execution

        def reconstructor(train: Any, november: Any) -> dict[str, Any]:
            from flight_delay.modeling.v1_execution import (
                reconstruct_governed_r3,
                require_r3_reconstruction,
            )

            return require_r3_reconstruction(reconstruct_governed_r3(train, november))

    control = loader(repository_root)
    lineage = {
        "r3_control_dataset_manifest_digest": control.manifest["manifest_digest"],
        "r3_control_protocol_sha256": control.protocol_sha256,
        "r3_control_sources": [
            "data/processed/train.parquet",
            "data/processed/validation.parquet",
        ],
        "r3_control_train_rows": len(control.train),
        "r3_control_november_rows": len(control.november),
        "r3_control_used_v3_population": False,
        "historical_test_accessed": False,
    }
    return reconstructor(control.train, control.november), lineage


def _write_winner_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V3ExecutionError("immutable winner model already exists")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(model, temporary)
        if path.exists():
            raise V3ExecutionError("immutable winner model already exists")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _winner_lock_payload(
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    code_sha: str,
    winner: dict[str, Any],
    state: V3HistoricalState,
    model_sha: str,
) -> dict[str, Any]:
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "implementation_git_sha": code_sha,
        "finalist_id": winner["finalist_id"],
        "kind": winner["kind"],
        "family": winner["family"],
        "calibration_method": winner["calibration_method"],
        "threshold": winner["metrics"]["threshold"],
        "feature_schema": list(V3_FEATURES),
        "categorical_schema": list(CATEGORICAL_FEATURES),
        "historical_state_sha256": state.sha256,
        "historical_state_schema_sha256": state.schema_sha256,
        "historical_state_as_of": state.as_of.isoformat(),
        "model_sha256": model_sha,
        "serialized_bundle_bytes": winner["bundle"]["serialized_bundle_bytes"],
        "development_metrics": winner["metrics"],
        "gate_evidence": [asdict(item) for item in winner["gate_evidence"]],
        "december_evaluated": False,
        "production_remains": "v0",
    }
    if winner["kind"] == "ensemble":
        payload.update(
            {
                "ensemble_id": winner["ensemble_id"],
                "lightgbm_weight": winner["lightgbm_weight"],
                "catboost_weight": winner["catboost_weight"],
                "lightgbm_base_candidate_id": winner["lightgbm_base_candidate_id"],
                "catboost_base_candidate_id": winner["catboost_base_candidate_id"],
            }
        )
    else:
        payload.update(
            {
                "base_candidate_id": winner["base_candidate_id"],
                "candidate_identity": winner["candidate_identity"],
            }
        )
    return payload


def run_development_apply(repository_root: Path, *, tracking: str) -> dict[str, Any]:
    """Run the one-time reviewed v3 development workflow and stop before December."""

    root = repository_root.resolve()
    if tracking != "online":
        raise V3ExecutionError("applied v3 development requires online governed tracking")
    code_sha = require_merged_applied_state(root)
    preflight(root, stage="development")
    require_versions()
    protocol, _lock, protocol_sha = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
        repository_root=root,
    )
    outputs = (DEVELOPMENT_MARKER, DECISION_PATH, STATE_PATH, WINNER_LOCK, WINNER_MODEL)
    if any((root / path).exists() for path in outputs):
        raise V3ExecutionError("v3 development marker or output already exists")
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
        stage = "r3_control_reconstruction"
        # The incumbent control runs on the canonical v1/v2 dataset, never on prepared.raw_history
        # or prepared.raw_november, which belong to the v3 challenger population.
        r3, r3_lineage = reconstruct_r3_control(root)
        metadata["dataset_lineage"] = {**prepared.lineage, **r3_lineage}
        stage = "screening_and_cpu_confirmation"
        search = run_screening_and_cpu_confirmation(
            protocol=protocol,
            transformed=prepared.search,
            tracker=tracker,
            metadata=metadata,
        )
        stage = "cpu_refit_calibration_ensembles_november"
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
            "r3_control_reconstruction": r3["reproduction"],
            "r3_control_lineage": r3_lineage,
            "dataset_lineage_separation": {
                "r3_control_dataset_manifest_digest": r3_lineage[
                    "r3_control_dataset_manifest_digest"
                ],
                "v3_dataset_manifest_digest": prepared.lineage["v3_dataset_manifest_digest"],
                "distinct": r3_lineage["r3_control_dataset_manifest_digest"]
                != prepared.lineage["v3_dataset_manifest_digest"],
            },
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
                _winner_lock_payload(
                    protocol=protocol,
                    protocol_sha=protocol_sha,
                    code_sha=code_sha,
                    winner=winner,
                    state=prepared.november_state,
                    model_sha=sha256_file(root / WINNER_MODEL),
                ),
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


def validate_december_handoff(root: Path) -> tuple[dict[str, Any], V3HistoricalState]:
    marker = _read_json(root / DEVELOPMENT_MARKER)
    if marker.get("status") != "complete" or marker.get("decision") != "winner":
        # A recovery adoption never rewrites the historical source marker. Its independently
        # hashed adoption record is the only supported bridge to the canonical handoff paths.
        try:
            from flight_delay.modeling.v3.recovery import validate_recovery_adoption_for_december

            validate_recovery_adoption_for_december(root, marker)
        except Exception as error:
            raise V3ExecutionError(
                "December requires a completed frozen November winner"
            ) from error
    winner = _read_json(root / WINNER_LOCK)
    if winner.get("december_evaluated") is not False:
        raise V3ExecutionError("December has already been evaluated")
    if winner.get("historical_state_as_of") != "2025-10-31":
        raise V3ExecutionError("winner does not reference the frozen October-31 state")
    if not (root / WINNER_MODEL).is_file() or sha256_file(root / WINNER_MODEL) != winner.get(
        "model_sha256"
    ):
        raise V3ExecutionError("frozen winner model hash mismatch")
    state = V3HistoricalState.from_bytes((root / STATE_PATH).read_bytes())
    if state.sha256 != winner.get("historical_state_sha256"):
        raise V3ExecutionError("frozen historical-state hash mismatch")
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
        raise V3ExecutionError("applied December qualification requires online governed tracking")
    code_sha = require_merged_applied_state(root)
    preflight(root, stage="qualification")
    require_versions()
    winner, state = validate_december_handoff(root)
    if winner.get("implementation_git_sha") != code_sha:
        raise V3ExecutionError("December code lineage differs from the frozen November winner")
    if (root / QUALIFICATION_MARKER).exists() or (root / QUALIFICATION_RESULT).exists():
        raise V3ExecutionError("December qualification already started or produced output")
    protocol, _lock, protocol_sha = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
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
    stage = "december_materialization"
    try:
        # Only now, after the frozen winner and its lineage have been validated, is December
        # decoded at all -- and only into the Git-ignored qualification workspace.
        materialized = materialize_december_qualification_data(
            root, december_authorization=DECEMBER_AUTHORIZATION
        )
        stage = "december_data_guard"
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
            name="v3-december-qualification",
            group=f"v3-{protocol_sha}-{code_sha}",
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
                "candidate_switched": False,
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
            "retrospective_not_genuine_final_test": True,
            "december_materialization": {
                "parquet_path": str(materialized.parquet_path.relative_to(root)),
                "manifest_path": str(materialized.manifest_path.relative_to(root)),
                "manifest_digest": materialized.manifest["manifest_digest"],
                "tracked_development_manifest_mutated": False,
                "workspace_is_git_ignored": True,
            },
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
