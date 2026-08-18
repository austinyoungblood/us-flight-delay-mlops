"""Fail-closed evidence reconstruction and governed v3 recovery orchestration."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.data.prepare_v3 import V3_PROCESSED_MANIFEST
from flight_delay.modeling.v2.models import require_versions
from flight_delay.modeling.v3.data import PreparedV3Data, prepare_development_data
from flight_delay.modeling.v3.execution import (
    DECISION_PATH,
    DEVELOPMENT_MARKER,
    STATE_PATH,
    THROUGHPUT_ROWS_PER_SECOND,
    WINNER_LOCK,
    WINNER_MODEL,
    V3ExecutionError,
    _atomic_bytes,
    _atomic_json,
    _read_json,
    _winner_lock_payload,
    _write_winner_model,
    preflight,
    require_merged_applied_state,
    update_marker,
    utc_now,
    validate_production_v0,
)
from flight_delay.modeling.v3.models import candidate_specs
from flight_delay.modeling.v3.protocol import FOLD_IDS, load_and_validate_v3_protocol, sha256_file
from flight_delay.modeling.v3.selection import (
    advance_family,
    screening_confirmation_differences,
    summarize_candidate,
)
from flight_delay.modeling.v3.tracking import WandbTracker
from flight_delay.modeling.v3.workflow import run_refit_and_november, sanitized_workflow_result

SOURCE_IMPLEMENTATION_SHA = "3dea562f06365df166c89af6e851a817a2db00fc"
SOURCE_PROTOCOL_SHA = "061be599fd84a4ddbf06229c300fe4670272d176b22899f1515332923376ecff"
SOURCE_MARKER_SHA256 = "90ae06f5bc81a3b86393d077866a2a2d65e3478d5920802dc4bee285a7fe9c1d"
SOURCE_LOG_SHA256 = "161f71825a752753206d54f72bc081f20573949ffaf492a001bf650002d67024"
CORRECTIVE_COMMIT_SHA = "d5cf5da6e01787aca7838265e5bfd28818f37d5d"
RECOVERY_REASON = "threshold_sweep_performance_defect"
SOURCE_GROUP = f"v3-{SOURCE_PROTOCOL_SHA}-{SOURCE_IMPLEMENTATION_SHA}"

RECOVERY_ROOT = Path("artifacts/v3/recovery")
RECOVERY_ADOPTION = Path("artifacts/v3/development/recovery_adoption.json")
SOURCE_EVIDENCE_NAME = "source_evidence.json"
SOURCE_EVIDENCE_DIGEST_NAME = "source_evidence.sha256"
TERMINATION_RECORD_NAME = "termination_record.json"
AUTHORIZATION_NAME = "authorization.json"
RECOVERY_MARKER_NAME = "recovery_marker.json"
DEVELOPMENT_RESULT_NAME = "development_result.json"
RECOVERY_DECISION_NAME = "decision.json"
RECOVERY_WINNER_LOCK_NAME = "winner_lock.json"
RECOVERY_WINNER_MODEL_NAME = "winner.joblib"
RECOVERY_STATE_NAME = "historical_state.json"

_RECOVERY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_RANKING_FOLD_METRICS = (
    "max_precision_at_operating_recall",
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "fit_rows",
    "evaluation_rows",
    "stage_runtime_seconds",
)


class V3RecoveryError(V3ExecutionError):
    """Raised when recovery evidence, authorization, or output fails closed."""


def recovery_directory(root: Path, recovery_id: str) -> Path:
    if not _RECOVERY_ID.fullmatch(recovery_id):
        raise V3RecoveryError("recovery_id must be a safe 1-80 character identifier")
    return root.resolve() / RECOVERY_ROOT / recovery_id


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    unsigned = {name: value for name, value in payload.items() if name != digest_field}
    return _sha256_bytes(canonical_json_bytes(unsigned))


def _freeze_self_hashed_json(
    path: Path, payload: dict[str, Any], *, digest_field: str
) -> dict[str, Any]:
    frozen = dict(payload)
    frozen[digest_field] = _payload_digest(frozen, digest_field)
    try:
        _atomic_json(path, frozen, refuse_existing=True)
    except V3ExecutionError as error:
        raise V3RecoveryError(str(error)) from error
    return frozen


def _read_recovery_json(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except V3ExecutionError as error:
        raise V3RecoveryError(str(error)) from error


def _require_self_hash(payload: Mapping[str, Any], digest_field: str, description: str) -> None:
    observed = payload.get(digest_field)
    if not isinstance(observed, str) or observed != _payload_digest(payload, digest_field):
        raise V3RecoveryError(f"{description} is tampered or lacks its payload digest")


def _require_hex_digest(value: Any, description: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise V3RecoveryError(f"{description} must be a lowercase SHA256 digest")
    return value


def _as_dict(value: Any, description: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise V3RecoveryError(f"{description} must be an object")


def _finite_number(value: Any, description: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise V3RecoveryError(f"{description} must be a finite number")
    return value


def _expected_spec(
    protocol: dict[str, Any], *, family: str, backend: str, candidate_id: str
) -> Any:
    if family not in {"lightgbm", "catboost"}:
        raise V3RecoveryError(f"wrong candidate family for {candidate_id}: {family}")
    expected_backend = "CPU" if family == "lightgbm" else backend
    try:
        specs = candidate_specs(protocol, family=family, backend=expected_backend)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise V3RecoveryError(f"wrong backend for {candidate_id}: {backend}") from error
    matches = [spec for spec in specs if spec.candidate_id == candidate_id]
    if len(matches) != 1:
        raise V3RecoveryError(f"wrong or unknown candidate identity: {candidate_id}")
    return matches[0]


def _folds_from_summary(summary: Mapping[str, Any], *, run_id: str) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold_id in FOLD_IDS:
        prefix = f"{fold_id}/"
        row: dict[str, Any] = {"fold_id": fold_id}
        for name, value in summary.items():
            if isinstance(name, str) and name.startswith(prefix) and not name.startswith("_"):
                metric = name[len(prefix) :]
                row[metric] = _finite_number(value, f"{run_id} {fold_id}/{metric}")
        missing = set(_RANKING_FOLD_METRICS) - set(row)
        if missing:
            raise V3RecoveryError(
                f"incomplete fold evidence for {run_id} {fold_id}: {sorted(missing)}"
            )
        folds.append(row)
    return folds


def _validated_tracking_run(
    protocol: dict[str, Any], raw: Mapping[str, Any], *, source_group: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _as_dict(raw, "tracking run")
    run_id = str(run.get("run_id", "")).strip()
    run_url = str(run.get("run_url", "")).strip()
    if not run_id or not run_url:
        raise V3RecoveryError("tracking evidence requires run ID and run URL")
    if run.get("state") != "finished":
        raise V3RecoveryError(f"source tracking run is not finished: {run_id}")
    if not str(run.get("created_at", "")).strip() or not str(run.get("updated_at", "")).strip():
        raise V3RecoveryError(f"tracking run lacks relevant timestamps: {run_id}")
    if run.get("group") != source_group:
        raise V3RecoveryError(f"source execution/group provenance mismatch: {run_id}")

    config = _as_dict(run.get("config"), f"tracking config for {run_id}")
    stage = str(config.get("stage", ""))
    if stage not in {"screening", "cpu_confirmation"}:
        raise V3RecoveryError(f"unexpected source stage in advancement evidence: {stage}")
    candidate_id = str(config.get("candidate_id", ""))
    family = str(config.get("family", ""))
    backend = str(config.get("backend", ""))
    required_backend = ("CPU" if family == "lightgbm" else "GPU") if stage == "screening" else "CPU"
    if backend != required_backend:
        raise V3RecoveryError(f"wrong backend for {candidate_id}: {backend}")
    spec = _expected_spec(protocol, family=family, backend=backend, candidate_id=candidate_id)

    expected = {
        "group": source_group,
        "protocol_sha256": SOURCE_PROTOCOL_SHA,
        "implementation_git_sha": SOURCE_IMPLEMENTATION_SHA,
        "candidate_id": candidate_id,
        "family": family,
        "backend": backend,
        "base_configuration": spec.base_configuration,
        "weight_policy": spec.weight_policy,
        "candidate_identity": spec.identity_parameters,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise V3RecoveryError(f"source run metadata mismatch for {candidate_id}: {name}")
    lineage = _as_dict(config.get("dataset_lineage"), f"dataset lineage for {run_id}")
    if config.get("feature_state_digest") != lineage.get("november_state_sha256"):
        raise V3RecoveryError(f"feature-state lineage mismatch for {run_id}")

    summary = _as_dict(run.get("summary"), f"tracking summary for {run_id}")
    folds = _folds_from_summary(summary, run_id=run_id)
    candidate = summarize_candidate(
        candidate_id=candidate_id,
        family=family,
        base_configuration=spec.base_configuration,
        weight_policy=spec.weight_policy,
        backend=backend,
        folds=folds,
    )
    candidate.update({"wandb_run_id": run_id, "wandb_run_url": run_url})
    normalized = {
        "run_id": run_id,
        "run_url": run_url,
        "name": str(run.get("name", "")),
        "state": "finished",
        "created_at": str(run["created_at"]),
        "updated_at": str(run["updated_at"]),
        "group": source_group,
        "config": config,
        "summary": summary,
    }
    return normalized, candidate


def build_source_evidence(
    *,
    protocol: dict[str, Any],
    recovery_id: str,
    tracking_runs: Sequence[Mapping[str, Any]],
    source_group: str = SOURCE_GROUP,
    exported_at: str | None = None,
) -> dict[str, Any]:
    """Validate original tracking rows and recompute both frozen advancement stages."""

    recovery_directory(Path.cwd(), recovery_id)
    if source_group != SOURCE_GROUP:
        raise V3RecoveryError("source group differs from the original governed execution")
    normalized: list[dict[str, Any]] = []
    candidate_rows: list[tuple[str, dict[str, Any]]] = []
    seen_run_ids: set[str] = set()
    seen_stage_candidates: set[tuple[str, str]] = set()
    lineage_bytes: bytes | None = None
    source_lineage: dict[str, Any] | None = None
    for raw in tracking_runs:
        run, candidate = _validated_tracking_run(protocol, raw, source_group=source_group)
        run_id = run["run_id"]
        stage = str(run["config"]["stage"])
        identity = (stage, str(candidate["candidate_id"]))
        if run_id in seen_run_ids or identity in seen_stage_candidates:
            raise V3RecoveryError("duplicate tracking run or candidate identity exists")
        seen_run_ids.add(run_id)
        seen_stage_candidates.add(identity)
        current_lineage = _as_dict(run["config"]["dataset_lineage"], "source lineage")
        if (
            current_lineage.get("december_decoded") is not False
            or current_lineage.get("january_may_2026_accessed") is not False
        ):
            raise V3RecoveryError("source tracking lineage crossed a prohibited period")
        current_lineage_bytes = canonical_json_bytes(current_lineage)
        if lineage_bytes is None:
            lineage_bytes, source_lineage = current_lineage_bytes, current_lineage
        elif current_lineage_bytes != lineage_bytes:
            raise V3RecoveryError("source runs do not share one exact dataset lineage")
        normalized.append(run)
        candidate_rows.append((stage, candidate))

    screening = [row for stage, row in candidate_rows if stage == "screening"]
    confirmation = [row for stage, row in candidate_rows if stage == "cpu_confirmation"]
    if len(screening) != 8 or len(confirmation) != 4:
        raise V3RecoveryError("expected eight screening and four CPU-confirmation runs")
    lightgbm_two = advance_family(screening, family="lightgbm", expected=4, advance=2)
    catboost_two = advance_family(screening, family="catboost", expected=4, advance=2)
    expected_confirmation = {str(row["candidate_id"]) for row in (*lightgbm_two, *catboost_two)}
    if {str(row["candidate_id"]) for row in confirmation} != expected_confirmation:
        raise V3RecoveryError("CPU confirmation identities do not match reconstructed screening")
    lightgbm_one = advance_family(confirmation, family="lightgbm", expected=2, advance=1)
    catboost_one = advance_family(confirmation, family="catboost", expected=2, advance=1)
    advanced = [*lightgbm_one, *catboost_one]
    if len(advanced) != 2 or {row["family"] for row in advanced} != {"lightgbm", "catboost"}:
        raise V3RecoveryError("reconstruction did not yield exactly one candidate per family")

    normalized.sort(key=lambda row: (str(row["config"]["stage"]), str(row["run_id"])))
    screening.sort(key=lambda row: str(row["candidate_id"]))
    confirmation.sort(key=lambda row: str(row["candidate_id"]))
    return {
        "schema_version": 1,
        "execution_mode": "governed_recovery_source_evidence",
        "recovery_id": recovery_id,
        "exported_at": exported_at or datetime.now(UTC).isoformat(),
        "source_execution_group": source_group,
        "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "source_execution_protocol_sha": SOURCE_PROTOCOL_SHA,
        "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
        "source_execution_log_sha256": SOURCE_LOG_SHA256,
        "source_dataset_lineage": source_lineage,
        "source_tracking_runs": normalized,
        "reconstruction": {
            "screening": screening,
            "advanced_to_cpu_confirmation": sorted(expected_confirmation),
            "cpu_confirmation": confirmation,
            "screening_cpu_differences": screening_confirmation_differences(
                screening, confirmation
            ),
            "advanced_to_refit": [
                {"candidate_id": row["candidate_id"], "family": row["family"]} for row in advanced
            ],
            "screening_repeated": False,
            "cpu_confirmation_repeated": False,
            "r3_reconstruction_passed_by_source_control_flow": True,
            "partial_november_finalist_used": False,
        },
    }


def freeze_source_evidence(
    root: Path,
    *,
    protocol: dict[str, Any],
    recovery_id: str,
    tracking_runs: Sequence[Mapping[str, Any]],
    source_group: str = SOURCE_GROUP,
    exported_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    directory = recovery_directory(root, recovery_id)
    payload = build_source_evidence(
        protocol=protocol,
        recovery_id=recovery_id,
        tracking_runs=tracking_runs,
        source_group=source_group,
        exported_at=exported_at,
    )
    path = directory / SOURCE_EVIDENCE_NAME
    encoded = canonical_json_bytes(payload) + b"\n"
    digest = _sha256_bytes(encoded)
    try:
        _atomic_bytes(path, encoded, refuse_existing=True)
        _atomic_bytes(
            directory / SOURCE_EVIDENCE_DIGEST_NAME,
            f"{digest}\n".encode(),
            refuse_existing=True,
        )
    except V3ExecutionError as error:
        raise V3RecoveryError(str(error)) from error
    return payload, digest


def wandb_source_runs(
    *, entity: str, project: str, source_group: str = SOURCE_GROUP, api_factory: Any | None = None
) -> list[dict[str, Any]]:
    """Read source runs through the public W&B API without mutating or resuming them."""

    if not entity or not project:
        raise V3RecoveryError("evidence export requires WANDB_ENTITY and WANDB_PROJECT")
    if source_group != SOURCE_GROUP:
        raise V3RecoveryError("evidence export is locked to the original source group")
    if api_factory is None:
        api_factory = importlib.import_module("wandb").Api
    api = api_factory()
    runs = api.runs(f"{entity}/{project}", filters={"group": source_group})
    exported: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        if config.get("stage") not in {"screening", "cpu_confirmation"}:
            continue
        summary = getattr(run.summary, "_json_dict", run.summary)
        exported.append(
            {
                "run_id": str(run.id),
                "run_url": str(run.url),
                "name": str(run.name),
                "state": str(run.state),
                "created_at": str(run.created_at),
                "updated_at": str(run.updated_at),
                "group": str(run.group),
                "config": config,
                "summary": dict(summary),
            }
        )
    return exported


def load_source_evidence(
    root: Path, *, protocol: dict[str, Any], recovery_id: str, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    directory = recovery_directory(root, recovery_id)
    path = directory / SOURCE_EVIDENCE_NAME
    sidecar = directory / SOURCE_EVIDENCE_DIGEST_NAME
    digest = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").strip() != digest:
        raise V3RecoveryError("source tracking evidence digest sidecar mismatch")
    if expected_sha256 is not None and digest != expected_sha256:
        raise V3RecoveryError("source tracking evidence authorization digest mismatch")
    payload = _read_recovery_json(path)
    rebuilt = build_source_evidence(
        protocol=protocol,
        recovery_id=recovery_id,
        tracking_runs=payload.get("source_tracking_runs", []),
        source_group=str(payload.get("source_execution_group", "")),
        exported_at=str(payload.get("exported_at", "")),
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(payload):
        raise V3RecoveryError("source tracking evidence cannot be independently reconstructed")
    return payload, digest


def create_termination_record(
    root: Path,
    *,
    recovery_id: str,
    source_root: Path,
    source_log: Path,
    original_pid: int | None,
    wrapper_exit_status: int,
    termination_mechanism: str,
    termination_reason: str,
    original_execution_terminated: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Freeze operator-supplied handoff facts after termination; never terminates a process."""

    if not original_execution_terminated:
        raise V3RecoveryError("termination record requires explicit terminated attestation")
    if not termination_mechanism.strip() or not termination_reason.strip():
        raise V3RecoveryError("termination mechanism and reason are required")
    source_root = source_root.resolve()
    marker_path = source_root / DEVELOPMENT_MARKER
    marker_digest = sha256_file(marker_path)
    if marker_digest != SOURCE_MARKER_SHA256:
        raise V3RecoveryError("source marker digest mismatch")
    marker = _read_recovery_json(marker_path)
    if (
        marker.get("status") != "started"
        or marker.get("implementation_git_sha") != SOURCE_IMPLEMENTATION_SHA
        or marker.get("protocol_sha") != SOURCE_PROTOCOL_SHA
    ):
        raise V3RecoveryError("source marker lineage or status mismatch")
    if (
        marker.get("december_opened") is not False
        or marker.get("historical_test_accessed") is not False
    ):
        raise V3RecoveryError("source marker does not preserve unopened prohibited periods")
    if (source_root / DECISION_PATH).exists() or (source_root / WINNER_LOCK).exists():
        raise V3RecoveryError("termination handoff requires absent decision and winner lock")
    log_after = sha256_file(source_log.resolve())
    payload = {
        "schema_version": 1,
        "recovery_id": recovery_id,
        "recorded_at": created_at or datetime.now(UTC).isoformat(),
        "original_execution_terminated": True,
        "original_pid": original_pid,
        "original_marker_sha256": marker_digest,
        "original_log_sha256_before_termination": SOURCE_LOG_SHA256,
        "original_log_sha256_after_termination": log_after,
        "wrapper_exit_status": wrapper_exit_status,
        "termination_mechanism": termination_mechanism,
        "termination_reason": termination_reason,
        "source_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "protocol_sha": SOURCE_PROTOCOL_SHA,
        "source_started_at": marker.get("started_at"),
        "source_marker_status": "started",
        "decision_absent": True,
        "winner_lock_absent": True,
        "december_unopened": True,
        "historical_final_test_unopened": True,
        "source_marker_preserved": True,
        "process_inspected_or_signaled_by_recorder": False,
    }
    return _freeze_self_hashed_json(
        recovery_directory(root, recovery_id) / TERMINATION_RECORD_NAME,
        payload,
        digest_field="termination_payload_sha256",
    )


def load_termination_record(root: Path, *, recovery_id: str) -> tuple[dict[str, Any], str]:
    path = recovery_directory(root, recovery_id) / TERMINATION_RECORD_NAME
    payload = _read_recovery_json(path)
    _require_self_hash(payload, "termination_payload_sha256", "termination record")
    expected = {
        "recovery_id": recovery_id,
        "original_execution_terminated": True,
        "original_marker_sha256": SOURCE_MARKER_SHA256,
        "original_log_sha256_before_termination": SOURCE_LOG_SHA256,
        "source_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "protocol_sha": SOURCE_PROTOCOL_SHA,
        "source_marker_status": "started",
        "decision_absent": True,
        "winner_lock_absent": True,
        "december_unopened": True,
        "historical_final_test_unopened": True,
        "source_marker_preserved": True,
        "process_inspected_or_signaled_by_recorder": False,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise V3RecoveryError("termination record does not match the governed source handoff")
    _require_hex_digest(
        payload.get("original_log_sha256_after_termination"), "post-termination log"
    )
    return payload, sha256_file(path)


def create_authorization(
    root: Path,
    *,
    protocol: dict[str, Any],
    recovery_id: str,
    corrected_selector_test_evidence: Mapping[str, Any],
    corrected_selector_benchmark_evidence: Mapping[str, Any],
    authorized_at: str | None = None,
) -> dict[str, Any]:
    termination, termination_digest = load_termination_record(root, recovery_id=recovery_id)
    _evidence, evidence_digest = load_source_evidence(
        root, protocol=protocol, recovery_id=recovery_id
    )
    if not corrected_selector_test_evidence or not corrected_selector_benchmark_evidence:
        raise V3RecoveryError("selector test and benchmark evidence are required")
    payload = {
        "schema_version": 1,
        "authorized_at": authorized_at or datetime.now(UTC).isoformat(),
        "recovery_id": recovery_id,
        "reason": RECOVERY_REASON,
        "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "source_execution_protocol_sha": SOURCE_PROTOCOL_SHA,
        "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
        "source_execution_log_sha256": SOURCE_LOG_SHA256,
        "source_started_at": termination["source_started_at"],
        "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
        "corrected_selector_test_evidence": dict(corrected_selector_test_evidence),
        "corrected_selector_benchmark_evidence": dict(corrected_selector_benchmark_evidence),
        "original_execution_terminated": True,
        "original_exit_status": termination["wrapper_exit_status"],
        "original_termination_record_sha256": termination_digest,
        "reconstructed_tracking_evidence_sha256": evidence_digest,
        "december_access_authorized": False,
        "historical_test_access_authorized": False,
        "production_mutation_authorized": False,
    }
    return _freeze_self_hashed_json(
        recovery_directory(root, recovery_id) / AUTHORIZATION_NAME,
        payload,
        digest_field="authorization_payload_sha256",
    )


def load_authorization(
    root: Path, *, protocol: dict[str, Any], recovery_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = recovery_directory(root, recovery_id) / AUTHORIZATION_NAME
    authorization = _read_recovery_json(path)
    _require_self_hash(authorization, "authorization_payload_sha256", "recovery authorization")
    expected = {
        "recovery_id": recovery_id,
        "reason": RECOVERY_REASON,
        "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "source_execution_protocol_sha": SOURCE_PROTOCOL_SHA,
        "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
        "source_execution_log_sha256": SOURCE_LOG_SHA256,
        "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
        "original_execution_terminated": True,
        "december_access_authorized": False,
        "historical_test_access_authorized": False,
        "production_mutation_authorized": False,
    }
    if any(authorization.get(name) != value for name, value in expected.items()):
        raise V3RecoveryError("authorization does not match the governed recovery incident")
    if not authorization.get("corrected_selector_test_evidence") or not authorization.get(
        "corrected_selector_benchmark_evidence"
    ):
        raise V3RecoveryError("authorization lacks corrected selector evidence")
    termination, termination_digest = load_termination_record(root, recovery_id=recovery_id)
    if authorization.get("original_termination_record_sha256") != termination_digest:
        raise V3RecoveryError("termination record authorization digest mismatch")
    if authorization.get("source_started_at") != termination.get("source_started_at"):
        raise V3RecoveryError("source start timestamp mismatch")
    if authorization.get("original_exit_status") != termination.get("wrapper_exit_status"):
        raise V3RecoveryError("source exit status mismatch")
    evidence, _digest = load_source_evidence(
        root,
        protocol=protocol,
        recovery_id=recovery_id,
        expected_sha256=str(authorization.get("reconstructed_tracking_evidence_sha256", "")),
    )
    return authorization, termination, evidence


def _load_protocol(root: Path) -> tuple[dict[str, Any], str]:
    protocol, _lock, digest = load_and_validate_v3_protocol(
        root / "configs/v3_experiment_protocol.yaml",
        lock_path=root / "experiments/v3/protocol_lock.json",
        repository_root=root,
    )
    return protocol, digest


def estimate_recovery_runtime(root: Path, protocol: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / V3_PROCESSED_MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    rows = sum(
        int(row["model_eligible_rows"])
        for row in manifest.get("monthly_counts", [])
        if "2024-02" <= str(row["month"]) <= "2025-10"
    )
    if rows <= 0:
        raise V3RecoveryError("recovery runtime estimate requires full-refit row counts")
    lightgbm = rows / THROUGHPUT_ROWS_PER_SECOND["lightgbm_cpu"]
    catboost = rows / THROUGHPUT_ROWS_PER_SECOND["catboost_cpu"]
    return {
        "full_refit_rows_per_base": rows,
        "lightgbm_refit_seconds": round(lightgbm, 1),
        "catboost_refit_seconds": round(catboost, 1),
        "two_base_fit_seconds": round(lightgbm + catboost, 1),
        "finalists_evaluated_from_scratch": int(protocol["finalists"]["total"]),
        "screening_repeated": False,
        "cpu_confirmation_repeated": False,
        "expected_wall_clock_minutes": {"lower": 60, "upper": 90},
    }


def recovery_preflight(root: Path, *, recovery_id: str) -> dict[str, Any]:
    root = root.resolve()
    directory = recovery_directory(root, recovery_id)
    protocol, protocol_sha = _load_protocol(root)
    base = preflight(root, stage="development")
    return {
        "mode": "dry-run/preflight",
        "execution_mode": "governed_recovery",
        "recovery_id": recovery_id,
        "recovery_directory": str(directory),
        "protocol_sha256": protocol_sha,
        "scientific_protocol_changed": False,
        "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
        "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
        "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
        "authorization_required_for_apply": True,
        "expected_inputs": [
            str(directory / SOURCE_EVIDENCE_NAME),
            str(directory / SOURCE_EVIDENCE_DIGEST_NAME),
            str(directory / TERMINATION_RECORD_NAME),
            str(directory / AUTHORIZATION_NAME),
        ],
        "screening_will_repeat": False,
        "cpu_confirmation_will_repeat": False,
        "authoritative_base_refits": 2,
        "november_finalists_from_scratch": int(protocol["finalists"]["total"]),
        "stops_before_december": True,
        "historical_test_accessed": False,
        "parquet_opened": False,
        "model_fit_started": False,
        "network_contacted": False,
        "production_v0_mutated": False,
        "runtime_estimate": estimate_recovery_runtime(root, protocol),
        "base_preflight": base,
    }


def require_recovery_applied_state(root: Path) -> str:
    code_sha = require_merged_applied_state(root)
    try:
        from flight_delay.modeling.v3.execution import _git

        _git(root, "merge-base", "--is-ancestor", CORRECTIVE_COMMIT_SHA, "HEAD")
    except Exception as error:
        raise V3RecoveryError("corrected selector commit must be an ancestor") from error
    return code_sha


def _applied_recovery_preflight(root: Path, recovery_id: str) -> dict[str, Any]:
    return recovery_preflight(root, recovery_id=recovery_id)


def _verify_rebuilt_lineage(prepared: PreparedV3Data, evidence: dict[str, Any]) -> None:
    source = _as_dict(evidence.get("source_dataset_lineage"), "source dataset lineage")
    for name, value in prepared.lineage.items():
        if source.get(name) != value:
            raise V3RecoveryError(f"rebuilt development lineage mismatch: {name}")
    if source.get("november_state_sha256") != prepared.november_state.sha256:
        raise V3RecoveryError("rebuilt historical-state digest differs from source execution")
    if (
        prepared.lineage.get("december_decoded") is not False
        or prepared.lineage.get("january_may_2026_accessed") is not False
    ):
        raise V3RecoveryError("recovery development data crossed a prohibited period")


def _recovery_tracker() -> WandbTracker:
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not entity or not project:
        raise V3RecoveryError("online recovery requires WANDB_ENTITY and WANDB_PROJECT")
    return WandbTracker(entity=entity, project=project)


def run_recovery_apply(
    root: Path,
    *,
    recovery_id: str,
    tracking: str,
    applied_state_validator: Callable[[Path], str] = require_recovery_applied_state,
    preflight_validator: Callable[[Path, str], dict[str, Any]] = _applied_recovery_preflight,
    version_validator: Callable[[], None] = require_versions,
    prepared_loader: Callable[[Path], PreparedV3Data] = prepare_development_data,
    tracker_factory: Callable[[], Any] = _recovery_tracker,
    workflow_runner: Callable[..., dict[str, Any]] = run_refit_and_november,
    protocol_loader: Callable[[Path], tuple[dict[str, Any], str]] = _load_protocol,
    production_validator: Callable[[Path, dict[str, Any]], dict[str, Any]] = validate_production_v0,
) -> dict[str, Any]:
    """Refit two reconstructed bases and evaluate 15 finalists; never screens or opens December."""

    root = root.resolve()
    directory = recovery_directory(root, recovery_id)
    if tracking != "online":
        raise V3RecoveryError("applied governed recovery requires new online tracking runs")
    code_sha = applied_state_validator(root)
    preflight_validator(root, recovery_id)
    version_validator()
    protocol, protocol_sha = protocol_loader(root)
    if protocol_sha != SOURCE_PROTOCOL_SHA:
        raise V3RecoveryError("recovery protocol differs from the frozen source protocol")
    authorization, termination, evidence = load_authorization(
        root, protocol=protocol, recovery_id=recovery_id
    )
    reserved = (
        RECOVERY_MARKER_NAME,
        DEVELOPMENT_RESULT_NAME,
        RECOVERY_DECISION_NAME,
        RECOVERY_WINNER_LOCK_NAME,
        RECOVERY_WINNER_MODEL_NAME,
        RECOVERY_STATE_NAME,
    )
    if any((directory / name).exists() for name in reserved):
        raise V3RecoveryError("recovery execution output already exists")
    production_before = production_validator(root, protocol)
    marker = directory / RECOVERY_MARKER_NAME
    _atomic_json(
        marker,
        {
            "status": "started",
            "execution_mode": "governed_recovery",
            "recovery_id": recovery_id,
            "recovery_reason": RECOVERY_REASON,
            "started_at": utc_now(),
            "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
            "source_execution_protocol_sha": SOURCE_PROTOCOL_SHA,
            "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
            "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
            "recovery_implementation_sha": code_sha,
            "source_tracking_evidence_sha256": authorization[
                "reconstructed_tracking_evidence_sha256"
            ],
            "original_termination_record_sha256": authorization[
                "original_termination_record_sha256"
            ],
            "december_opened": False,
            "historical_test_accessed": False,
            "production_v0_mutated": False,
        },
        refuse_existing=True,
    )
    stage = "development_lineage"
    try:
        prepared = prepared_loader(root)
        _verify_rebuilt_lineage(prepared, evidence)
        _atomic_bytes(
            directory / RECOVERY_STATE_NAME,
            prepared.november_state.to_bytes(),
            refuse_existing=True,
        )
        metadata = {
            "group": f"v3-recovery-{recovery_id}",
            "execution_mode": "governed_recovery",
            "recovery_id": recovery_id,
            "recovery_reason": RECOVERY_REASON,
            "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
            "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
            "protocol_sha": protocol_sha,
            "protocol_sha256": protocol_sha,
            "source_tracking_evidence_sha256": authorization[
                "reconstructed_tracking_evidence_sha256"
            ],
            "implementation_git_sha": code_sha,
            "hardware_identity": platform.platform(),
            "feature_state_digest": prepared.november_state.sha256,
            "dataset_lineage": prepared.lineage,
        }
        stage = "authoritative_two_base_refit_and_november"
        november = workflow_runner(
            prepared=prepared,
            protocol=protocol,
            advanced=evidence["reconstruction"]["advanced_to_refit"],
            tracker=tracker_factory(),
            metadata=metadata,
            r3_reconstruction_passed=True,
        )
        sanitized = sanitized_workflow_result(november)
        _atomic_json(directory / DEVELOPMENT_RESULT_NAME, sanitized, refuse_existing=True)
        decision = {
            **metadata,
            "decision": november["decision"],
            "execution_mode": "governed_recovery",
            "recovery_reason": RECOVERY_REASON,
            "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
            "source_tracking_evidence_sha256": authorization[
                "reconstructed_tracking_evidence_sha256"
            ],
            "source_termination_record_sha256": authorization["original_termination_record_sha256"],
            "source_started_at": termination["source_started_at"],
            "screening_repeated": False,
            "cpu_confirmation_repeated": False,
            "partial_original_november_finalist_used": False,
            "advanced_to_refit": evidence["reconstruction"]["advanced_to_refit"],
            "november": sanitized,
            "production_remains": "v0",
            "stopped_before_december": True,
            "historical_test_accessed": False,
        }
        _atomic_json(directory / RECOVERY_DECISION_NAME, decision, refuse_existing=True)
        winner = november.get("winner")
        if winner is not None:
            model_path = directory / RECOVERY_WINNER_MODEL_NAME
            _write_winner_model(model_path, winner["model"])
            lock = _winner_lock_payload(
                protocol=protocol,
                protocol_sha=protocol_sha,
                code_sha=code_sha,
                winner=winner,
                state=prepared.november_state,
                model_sha=sha256_file(model_path),
            )
            lock.update(
                {
                    "execution_mode": "governed_recovery",
                    "recovery_id": recovery_id,
                    "recovery_reason": RECOVERY_REASON,
                    "source_execution_implementation_sha": SOURCE_IMPLEMENTATION_SHA,
                    "corrective_commit_sha": CORRECTIVE_COMMIT_SHA,
                    "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
                    "source_tracking_evidence_sha256": authorization[
                        "reconstructed_tracking_evidence_sha256"
                    ],
                }
            )
            _atomic_json(directory / RECOVERY_WINNER_LOCK_NAME, lock, refuse_existing=True)
        if production_validator(root, protocol) != production_before:
            raise V3RecoveryError("production v0 changed during governed recovery")
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


def adoption_preflight(root: Path, *, recovery_id: str) -> dict[str, Any]:
    directory = recovery_directory(root, recovery_id)
    return {
        "mode": "dry-run/preflight",
        "operation": "governed_recovery_adoption",
        "recovery_id": recovery_id,
        "source_recovery_decision": str(directory / RECOVERY_DECISION_NAME),
        "canonical_decision": str(root.resolve() / DECISION_PATH),
        "original_marker_will_be_rewritten": False,
        "fails_if_canonical_decision_exists": True,
        "production_v0_mutation_permitted": False,
        "december_opened": False,
        "historical_test_accessed": False,
    }


def adopt_recovery(root: Path, *, recovery_id: str) -> dict[str, Any]:
    """Adopt completed recovery outputs without rewriting the original historical marker."""

    root = root.resolve()
    directory = recovery_directory(root, recovery_id)
    marker = _read_recovery_json(directory / RECOVERY_MARKER_NAME)
    decision = _read_recovery_json(directory / RECOVERY_DECISION_NAME)
    if marker.get("status") != "complete" or marker.get("decision") != decision.get("decision"):
        raise V3RecoveryError("only a completed recovery decision can be adopted")
    if (
        decision.get("execution_mode") != "governed_recovery"
        or decision.get("recovery_id") != recovery_id
    ):
        raise V3RecoveryError("recovery decision provenance is missing")
    source_marker = root / DEVELOPMENT_MARKER
    if sha256_file(source_marker) != SOURCE_MARKER_SHA256:
        raise V3RecoveryError("canonical source marker digest mismatch")
    source_marker_payload = _read_recovery_json(source_marker)
    if source_marker_payload.get("status") != "started":
        raise V3RecoveryError("adoption requires the preserved original started marker")
    if (root / DECISION_PATH).exists():
        raise V3RecoveryError("canonical decision already exists")
    if (root / RECOVERY_ADOPTION).exists():
        raise V3RecoveryError("recovery adoption already exists")

    decision_bytes = (directory / RECOVERY_DECISION_NAME).read_bytes()
    decision_name = str(decision["decision"])
    sources: list[tuple[Path, Path]] = [(directory / RECOVERY_DECISION_NAME, root / DECISION_PATH)]
    if decision_name == "winner":
        sources.extend(
            [
                (directory / RECOVERY_STATE_NAME, root / STATE_PATH),
                (directory / RECOVERY_WINNER_LOCK_NAME, root / WINNER_LOCK),
                (directory / RECOVERY_WINNER_MODEL_NAME, root / WINNER_MODEL),
            ]
        )
    elif decision_name != "governed_stop":
        raise V3RecoveryError("recovery decision must be winner or governed_stop")
    if any(not source.is_file() for source, _target in sources):
        raise V3RecoveryError("recovery adoption source artifact is missing")
    if any(target.exists() for _source, target in sources):
        raise V3RecoveryError("canonical recovery target already exists")
    for source, target in sources:
        _atomic_bytes(target, source.read_bytes(), refuse_existing=True)

    adoption = {
        "schema_version": 1,
        "status": "adopted",
        "execution_mode": "governed_recovery",
        "recovery_id": recovery_id,
        "adopted_at": utc_now(),
        "decision": decision_name,
        "source_execution_marker_sha256": SOURCE_MARKER_SHA256,
        "recovery_marker_sha256": sha256_file(directory / RECOVERY_MARKER_NAME),
        "recovery_decision_sha256": _sha256_bytes(decision_bytes),
        "canonical_decision_sha256": sha256_file(root / DECISION_PATH),
        "original_marker_rewritten": False,
        "production_remains": "v0",
        "december_opened": False,
        "historical_test_accessed": False,
    }
    if decision_name == "winner":
        adoption.update(
            {
                "canonical_state_sha256": sha256_file(root / STATE_PATH),
                "canonical_winner_lock_sha256": sha256_file(root / WINNER_LOCK),
                "canonical_winner_model_sha256": sha256_file(root / WINNER_MODEL),
            }
        )
    return _freeze_self_hashed_json(
        root / RECOVERY_ADOPTION, adoption, digest_field="adoption_payload_sha256"
    )


def validate_recovery_adoption_for_december(root: Path, marker: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicit adoption while preserving the source marker's original bytes."""

    adoption = _read_recovery_json(root / RECOVERY_ADOPTION)
    _require_self_hash(adoption, "adoption_payload_sha256", "recovery adoption")
    if marker.get("status") != "started" or sha256_file(root / DEVELOPMENT_MARKER) != adoption.get(
        "source_execution_marker_sha256"
    ):
        raise V3RecoveryError("recovery adoption source marker mismatch")
    if adoption.get("status") != "adopted" or adoption.get("decision") != "winner":
        raise V3RecoveryError("December requires an adopted recovery winner")
    checks = {
        DECISION_PATH: "canonical_decision_sha256",
        STATE_PATH: "canonical_state_sha256",
        WINNER_LOCK: "canonical_winner_lock_sha256",
        WINNER_MODEL: "canonical_winner_model_sha256",
    }
    for relative, field in checks.items():
        if not (root / relative).is_file() or sha256_file(root / relative) != adoption.get(field):
            raise V3RecoveryError(f"adopted recovery artifact mismatch: {relative}")
    decision = _read_recovery_json(root / DECISION_PATH)
    if decision.get("execution_mode") != "governed_recovery" or decision.get(
        "recovery_id"
    ) != adoption.get("recovery_id"):
        raise V3RecoveryError("adopted decision lacks recovery provenance")
    return adoption
