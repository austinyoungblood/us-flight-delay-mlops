"""Governed v3 recovery tests using only synthetic evidence and synthetic workflow results."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flight_delay.data.manifest import canonical_json_bytes, write_manifest
from flight_delay.modeling.v1_selection import GateEvidence, select_v1_threshold
from flight_delay.modeling.v3 import execution as v3_execution
from flight_delay.modeling.v3 import recovery
from flight_delay.modeling.v3.execution import DECISION_PATH, DEVELOPMENT_MARKER
from flight_delay.modeling.v3.models import candidate_specs
from flight_delay.modeling.v3.protocol import FOLD_IDS, sha256_file
from flight_delay.modeling.v3.recovery import (
    AUTHORIZATION_NAME,
    CORRECTIVE_COMMIT_SHA,
    DEVELOPMENT_RESULT_NAME,
    RECOVERY_ADOPTION,
    RECOVERY_DECISION_NAME,
    RECOVERY_MARKER_NAME,
    RECOVERY_REASON,
    RECOVERY_STATE_NAME,
    RECOVERY_WINNER_LOCK_NAME,
    RECOVERY_WINNER_MODEL_NAME,
    SOURCE_GROUP,
    SOURCE_IMPLEMENTATION_SHA,
    SOURCE_PROTOCOL_SHA,
    TERMINATION_RECORD_NAME,
    V3RecoveryError,
    adopt_recovery,
    adoption_preflight,
    build_source_evidence,
    create_authorization,
    create_termination_record,
    estimate_recovery_runtime,
    freeze_source_evidence,
    load_authorization,
    load_source_evidence,
    load_termination_record,
    recovery_directory,
    recovery_preflight,
    require_recovery_applied_state,
    run_recovery_apply,
    validate_recovery_adoption_for_december,
    wandb_source_runs,
)

RECOVERY_ID = "threshold-fix-001"
ROOT = Path(__file__).resolve().parents[2]


def _lineage(state_sha: str = "9" * 64) -> dict[str, Any]:
    return {
        "protocol_sha256": SOURCE_PROTOCOL_SHA,
        "v3_dataset_manifest_digest": "dataset-digest",
        "november_state_sha256": state_sha,
        "december_decoded": False,
        "january_may_2026_accessed": False,
    }


def _fold_summary(primary: float) -> dict[str, float | int]:
    summary: dict[str, float | int] = {}
    for index, fold_id in enumerate(FOLD_IDS):
        value = primary - index * 0.001
        summary.update(
            {
                f"{fold_id}/max_precision_at_operating_recall": value,
                f"{fold_id}/average_precision": value + 0.05,
                f"{fold_id}/roc_auc": value + 0.15,
                f"{fold_id}/log_loss": 1.0 - value,
                f"{fold_id}/brier_score": 0.5 - value / 2,
                f"{fold_id}/fit_rows": 1000 + index,
                f"{fold_id}/evaluation_rows": 200 + index,
                f"{fold_id}/stage_runtime_seconds": 1.0 + index,
            }
        )
    return summary


def _raw_run(
    protocol: dict[str, Any],
    *,
    candidate_id: str,
    family: str,
    backend: str,
    stage: str,
    primary: float,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    spec = next(
        spec
        for spec in candidate_specs(protocol, family=family, backend=backend)  # type: ignore[arg-type]
        if spec.candidate_id == candidate_id
    )
    run_id = f"{stage}-{candidate_id}"
    return {
        "run_id": run_id,
        "run_url": f"https://wandb.example/runs/{run_id}",
        "name": f"v3-{candidate_id}-{backend.lower()}",
        "state": "finished",
        "created_at": "2026-08-17T01:00:00+00:00",
        "updated_at": "2026-08-17T01:05:00+00:00",
        "group": SOURCE_GROUP,
        "config": {
            "group": SOURCE_GROUP,
            "stage": stage,
            "protocol_sha256": SOURCE_PROTOCOL_SHA,
            "implementation_git_sha": SOURCE_IMPLEMENTATION_SHA,
            "candidate_id": candidate_id,
            "family": family,
            "backend": backend,
            "base_configuration": spec.base_configuration,
            "weight_policy": spec.weight_policy,
            "candidate_identity": spec.identity_parameters,
            "feature_state_digest": lineage["november_state_sha256"],
            "dataset_lineage": lineage,
        },
        "summary": _fold_summary(primary),
    }


def raw_tracking_runs(
    protocol: dict[str, Any], *, lineage: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    lineage = lineage or _lineage()
    screening = (
        ("LGBM12-UNIFORM", "lightgbm", "CPU", 0.50),
        ("LGBM12-EXP120", "lightgbm", "CPU", 0.60),
        ("LGBM10-UNIFORM", "lightgbm", "CPU", 0.40),
        ("LGBM10-EXP120", "lightgbm", "CPU", 0.30),
        ("CB07-UNIFORM", "catboost", "GPU", 0.30),
        ("CB07-EXP120", "catboost", "GPU", 0.40),
        ("CB04-UNIFORM", "catboost", "GPU", 0.50),
        ("CB04-EXP120", "catboost", "GPU", 0.60),
    )
    confirmation = (
        ("LGBM12-UNIFORM", "lightgbm", "CPU", 0.65),
        ("LGBM12-EXP120", "lightgbm", "CPU", 0.64),
        ("CB04-UNIFORM", "catboost", "CPU", 0.66),
        ("CB04-EXP120", "catboost", "CPU", 0.67),
    )
    return [
        *[
            _raw_run(
                protocol,
                candidate_id=candidate,
                family=family,
                backend=backend,
                stage="screening",
                primary=score,
                lineage=lineage,
            )
            for candidate, family, backend, score in screening
        ],
        *[
            _raw_run(
                protocol,
                candidate_id=candidate,
                family=family,
                backend=backend,
                stage="cpu_confirmation",
                primary=score,
                lineage=lineage,
            )
            for candidate, family, backend, score in confirmation
        ],
    ]


def test_valid_evidence_reconstructs_screening_and_cpu_advancement(v3_protocol: dict) -> None:
    evidence = build_source_evidence(
        protocol=v3_protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(v3_protocol),
        exported_at="2026-08-18T00:00:00+00:00",
    )
    reconstruction = evidence["reconstruction"]
    assert reconstruction["advanced_to_cpu_confirmation"] == [
        "CB04-EXP120",
        "CB04-UNIFORM",
        "LGBM12-EXP120",
        "LGBM12-UNIFORM",
    ]
    assert reconstruction["advanced_to_refit"] == [
        {"candidate_id": "LGBM12-UNIFORM", "family": "lightgbm"},
        {"candidate_id": "CB04-EXP120", "family": "catboost"},
    ]
    assert reconstruction["screening_repeated"] is False
    assert reconstruction["cpu_confirmation_repeated"] is False
    assert reconstruction["partial_november_finalist_used"] is False
    assert len(reconstruction["screening_cpu_differences"]) == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runs: runs.pop(), "eight screening and four"),
        (
            lambda runs: runs.append(copy.deepcopy(runs[0])),
            "duplicate tracking run or candidate",
        ),
        (
            lambda runs: runs[0]["config"].__setitem__("candidate_id", "LGBM99-UNIFORM"),
            "wrong or unknown candidate",
        ),
        (
            lambda runs: runs[0]["config"].__setitem__("family", "catboost"),
            "wrong backend|wrong or unknown candidate",
        ),
        (
            lambda runs: runs[0]["config"].__setitem__("backend", "GPU"),
            "wrong backend",
        ),
        (
            lambda runs: runs[0]["config"].__setitem__("protocol_sha256", "0" * 64),
            "protocol_sha256",
        ),
        (
            lambda runs: runs[0]["config"].__setitem__("implementation_git_sha", "0" * 40),
            "implementation_git_sha",
        ),
        (
            lambda runs: runs[0]["summary"].pop("FOLD_4/roc_auc"),
            "incomplete fold evidence",
        ),
    ],
)
def test_evidence_reconstruction_fails_closed(
    v3_protocol: dict, mutation: Any, message: str
) -> None:
    runs = raw_tracking_runs(v3_protocol)
    mutation(runs)
    with pytest.raises(V3RecoveryError, match=message):
        build_source_evidence(protocol=v3_protocol, recovery_id=RECOVERY_ID, tracking_runs=runs)


def test_wrong_cpu_confirmation_candidate_is_refused(v3_protocol: dict) -> None:
    runs = raw_tracking_runs(v3_protocol)
    replacement = _raw_run(
        v3_protocol,
        candidate_id="LGBM10-UNIFORM",
        family="lightgbm",
        backend="CPU",
        stage="cpu_confirmation",
        primary=0.9,
        lineage=_lineage(),
    )
    runs[8] = replacement
    with pytest.raises(V3RecoveryError, match="do not match reconstructed screening"):
        build_source_evidence(protocol=v3_protocol, recovery_id=RECOVERY_ID, tracking_runs=runs)


def test_source_group_and_lineage_must_be_exact(v3_protocol: dict) -> None:
    with pytest.raises(V3RecoveryError, match="source group differs"):
        build_source_evidence(
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            tracking_runs=[],
            source_group="wrong",
        )
    runs = raw_tracking_runs(v3_protocol)
    runs[0]["config"]["dataset_lineage"] = {**_lineage(), "extra": True}
    with pytest.raises(V3RecoveryError, match="one exact dataset lineage"):
        build_source_evidence(protocol=v3_protocol, recovery_id=RECOVERY_ID, tracking_runs=runs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run.__setitem__("run_url", ""), "run ID and run URL"),
        (lambda run: run.__setitem__("state", "running"), "not finished"),
        (lambda run: run.__setitem__("created_at", ""), "timestamps"),
        (lambda run: run.__setitem__("group", "wrong"), "group provenance"),
        (lambda run: run["config"].__setitem__("stage", "november_finalist"), "unexpected"),
        (
            lambda run: run["config"].__setitem__("feature_state_digest", "0" * 64),
            "feature-state lineage",
        ),
        (
            lambda run: run["summary"].__setitem__("FOLD_1/roc_auc", float("nan")),
            "finite number",
        ),
    ],
)
def test_tracking_run_identity_and_completion_are_required(
    v3_protocol: dict, mutation: Any, message: str
) -> None:
    runs = raw_tracking_runs(v3_protocol)
    mutation(runs[0])
    with pytest.raises(V3RecoveryError, match=message):
        build_source_evidence(protocol=v3_protocol, recovery_id=RECOVERY_ID, tracking_runs=runs)


def test_source_evidence_refuses_prohibited_period_lineage(v3_protocol: dict) -> None:
    runs = raw_tracking_runs(v3_protocol, lineage={**_lineage(), "december_decoded": True})
    with pytest.raises(V3RecoveryError, match="prohibited period"):
        build_source_evidence(protocol=v3_protocol, recovery_id=RECOVERY_ID, tracking_runs=runs)


class FakeSummary:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._json_dict = payload


class FakeRun:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.id = payload["run_id"]
        self.url = payload["run_url"]
        self.name = payload["name"]
        self.state = payload["state"]
        self.created_at = payload["created_at"]
        self.updated_at = payload["updated_at"]
        self.group = payload["group"]
        self.config = payload["config"]
        self.summary = FakeSummary(payload["summary"])


def test_wandb_export_is_read_only_and_ignores_partial_finalist(v3_protocol: dict) -> None:
    source = raw_tracking_runs(v3_protocol)
    partial = copy.deepcopy(source[0])
    partial["run_id"] = "partial-finalist"
    partial["config"]["stage"] = "november_finalist"

    class FakeApi:
        def runs(self, path: str, *, filters: dict[str, Any]) -> list[FakeRun]:
            assert path == "entity/project"
            assert filters == {"group": SOURCE_GROUP}
            return [FakeRun(row) for row in [*source, partial]]

    exported = wandb_source_runs(entity="entity", project="project", api_factory=lambda: FakeApi())
    assert len(exported) == 12
    assert all(row["config"]["stage"] != "november_finalist" for row in exported)
    with pytest.raises(V3RecoveryError, match="WANDB_ENTITY"):
        wandb_source_runs(entity="", project="project", api_factory=lambda: FakeApi())
    with pytest.raises(V3RecoveryError, match="original source group"):
        wandb_source_runs(
            entity="entity", project="project", source_group="wrong", api_factory=lambda: FakeApi()
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _source_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "source"
    marker = source / DEVELOPMENT_MARKER
    _write_json(
        marker,
        {
            "status": "started",
            "implementation_git_sha": SOURCE_IMPLEMENTATION_SHA,
            "protocol_sha": SOURCE_PROTOCOL_SHA,
            "started_at": "2026-08-17T00:00:00+00:00",
            "december_opened": False,
            "historical_test_accessed": False,
        },
    )
    source_log = source / "governed-v3.log"
    source_log.write_text("synthetic captured log\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "SOURCE_MARKER_SHA256", sha256_file(marker))
    monkeypatch.setattr(recovery, "SOURCE_LOG_SHA256", sha256_file(source_log))
    return source, source_log


def _freeze_inputs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: dict[str, Any],
    *,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source, source_log = _source_handoff(root, monkeypatch)
    freeze_source_evidence(
        root,
        protocol=protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(protocol, lineage=lineage),
        exported_at="2026-08-18T00:00:00+00:00",
    )
    create_termination_record(
        root,
        recovery_id=RECOVERY_ID,
        source_root=source,
        source_log=source_log,
        original_pid=1234,
        wrapper_exit_status=143,
        termination_mechanism="operator SIGTERM after external confirmation",
        termination_reason=RECOVERY_REASON,
        original_execution_terminated=True,
        created_at="2026-08-18T00:01:00+00:00",
    )
    return create_authorization(
        root,
        protocol=protocol,
        recovery_id=RECOVERY_ID,
        corrected_selector_test_evidence={"result": "132 passed"},
        corrected_selector_benchmark_evidence={"rows": 555295, "seconds": 1.8817},
        authorized_at="2026-08-18T00:02:00+00:00",
    )


def test_evidence_files_are_immutable_and_independently_reconstructed(
    tmp_path: Path, v3_protocol: dict
) -> None:
    _payload, digest = freeze_source_evidence(
        tmp_path,
        protocol=v3_protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(v3_protocol),
        exported_at="2026-08-18T00:00:00+00:00",
    )
    loaded, observed = load_source_evidence(
        tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID, expected_sha256=digest
    )
    assert loaded["reconstruction"]["advanced_to_refit"][0]["family"] == "lightgbm"
    assert observed == digest
    with pytest.raises(V3RecoveryError, match="immutable output"):
        freeze_source_evidence(
            tmp_path,
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            tracking_runs=raw_tracking_runs(v3_protocol),
        )


def test_source_evidence_detects_sidecar_authorization_and_payload_tampering(
    tmp_path: Path, v3_protocol: dict
) -> None:
    _payload, digest = freeze_source_evidence(
        tmp_path,
        protocol=v3_protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(v3_protocol),
        exported_at="2026-08-18T00:00:00+00:00",
    )
    directory = recovery_directory(tmp_path, RECOVERY_ID)
    sidecar = directory / "source_evidence.sha256"
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(V3RecoveryError, match="sidecar mismatch"):
        load_source_evidence(tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    with pytest.raises(V3RecoveryError, match="authorization digest mismatch"):
        load_source_evidence(
            tmp_path,
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            expected_sha256="0" * 64,
        )

    evidence_path = directory / "source_evidence.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    encoded = canonical_json_bytes(payload) + b"\n"
    evidence_path.write_bytes(encoded)
    sidecar.write_text(recovery._sha256_bytes(encoded) + "\n", encoding="utf-8")
    with pytest.raises(V3RecoveryError, match="independently reconstructed"):
        load_source_evidence(tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID)


def test_termination_record_requires_handoff_and_exact_source_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_log = _source_handoff(tmp_path, monkeypatch)
    with pytest.raises(V3RecoveryError, match="terminated attestation"):
        create_termination_record(
            tmp_path,
            recovery_id=RECOVERY_ID,
            source_root=source,
            source_log=source_log,
            original_pid=None,
            wrapper_exit_status=1,
            termination_mechanism="operator",
            termination_reason=RECOVERY_REASON,
            original_execution_terminated=False,
        )
    monkeypatch.setattr(recovery, "SOURCE_MARKER_SHA256", "0" * 64)
    with pytest.raises(V3RecoveryError, match="source marker digest mismatch"):
        create_termination_record(
            tmp_path,
            recovery_id=RECOVERY_ID,
            source_root=source,
            source_log=source_log,
            original_pid=None,
            wrapper_exit_status=1,
            termination_mechanism="operator",
            termination_reason=RECOVERY_REASON,
            original_execution_terminated=True,
        )


def test_source_log_digest_mismatch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol)
    authorization_path = recovery_directory(tmp_path, RECOVERY_ID) / AUTHORIZATION_NAME
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["source_execution_log_sha256"] = "0" * 64
    authorization["authorization_payload_sha256"] = recovery._payload_digest(
        authorization, "authorization_payload_sha256"
    )
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(V3RecoveryError, match="governed recovery incident"):
        load_authorization(tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID)


def test_authorization_missing_tampered_and_missing_termination_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    with pytest.raises(V3RecoveryError, match="cannot read governed state"):
        load_authorization(tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID)

    root = tmp_path / "complete"
    _freeze_inputs(root, monkeypatch, v3_protocol)
    authorization_path = recovery_directory(root, RECOVERY_ID) / AUTHORIZATION_NAME
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    payload["reason"] = "changed"
    authorization_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(V3RecoveryError, match="tampered"):
        load_authorization(root, protocol=v3_protocol, recovery_id=RECOVERY_ID)

    missing = tmp_path / "missing-termination"
    _source_handoff(missing, monkeypatch)
    freeze_source_evidence(
        missing,
        protocol=v3_protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(v3_protocol),
    )
    with pytest.raises(V3RecoveryError, match="cannot read governed state"):
        create_authorization(
            missing,
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            corrected_selector_test_evidence={"passed": True},
            corrected_selector_benchmark_evidence={"passed": True},
        )


def _rewrite_authorization(path: Path, **updates: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload["authorization_payload_sha256"] = recovery._payload_digest(
        payload, "authorization_payload_sha256"
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"corrected_selector_test_evidence": {}}, "lacks corrected selector"),
        ({"original_termination_record_sha256": "0" * 64}, "termination record"),
        ({"source_started_at": "wrong"}, "start timestamp"),
        ({"original_exit_status": 0}, "exit status"),
    ],
)
def test_authorization_cross_checks_every_handoff_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v3_protocol: dict,
    updates: dict[str, Any],
    message: str,
) -> None:
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol)
    path = recovery_directory(tmp_path, RECOVERY_ID) / AUTHORIZATION_NAME
    _rewrite_authorization(path, **updates)
    with pytest.raises(V3RecoveryError, match=message):
        load_authorization(tmp_path, protocol=v3_protocol, recovery_id=RECOVERY_ID)


def test_authorization_requires_selector_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    source, source_log = _source_handoff(tmp_path, monkeypatch)
    freeze_source_evidence(
        tmp_path,
        protocol=v3_protocol,
        recovery_id=RECOVERY_ID,
        tracking_runs=raw_tracking_runs(v3_protocol),
    )
    create_termination_record(
        tmp_path,
        recovery_id=RECOVERY_ID,
        source_root=source,
        source_log=source_log,
        original_pid=None,
        wrapper_exit_status=143,
        termination_mechanism="operator",
        termination_reason=RECOVERY_REASON,
        original_execution_terminated=True,
    )
    with pytest.raises(V3RecoveryError, match="selector test and benchmark"):
        create_authorization(
            tmp_path,
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            corrected_selector_test_evidence={},
            corrected_selector_benchmark_evidence={"passed": True},
        )


@dataclass
class FakeState:
    sha256: str
    schema_sha256: str = "8" * 64
    as_of: date = date(2025, 10, 31)

    def to_bytes(self) -> bytes:
        return json.dumps(
            {"sha256": self.sha256, "schema_sha256": self.schema_sha256}, sort_keys=True
        ).encode()


class FakeWinnerModel:
    def predict_proba(self, features: Any) -> np.ndarray:
        scores = np.full(len(features), 0.5)
        return np.column_stack((1 - scores, scores))


def _prepared(lineage: dict[str, Any]) -> Any:
    return SimpleNamespace(
        november_state=FakeState(lineage["november_state_sha256"]), lineage=lineage
    )


def _workflow_result(*, winner: bool) -> dict[str, Any]:
    finalist = {
        "model": FakeWinnerModel(),
        "bundle": {"serialized_bundle_bytes": 1024},
        "wandb_run_id": "new-recovery-run",
        "wandb_run_url": "https://wandb.example/new-recovery-run",
        "kind": "base",
        "family": "lightgbm",
        "base_candidate_id": "LGBM12-UNIFORM",
        "calibration_method": "none",
        "candidate_identity": {"weight_policy": "UNIFORM"},
        "finalist_id": "LGBM12-UNIFORM-none",
        "status": "completed",
        "threshold_selection": {},
        "metrics": {"threshold": 0.42},
        "gate_evidence": (GateEvidence("synthetic", "is true", True, True),),
        "passed": winner,
    }
    return {
        "decision": "winner" if winner else "governed_stop",
        "winner": finalist if winner else None,
        "finalists": [finalist],
        "base_refits": {
            "lightgbm": {"candidate_id": "LGBM12-UNIFORM"},
            "catboost": {"candidate_id": "CB04-EXP120"},
        },
        "production_remains": "v0",
        "stopped_before_december": True,
    }


def _run_synthetic_recovery(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: dict[str, Any],
    *,
    winner: bool,
    lineage: dict[str, Any] | None = None,
    production_validator: Any | None = None,
) -> dict[str, Any]:
    lineage = lineage or _lineage()
    _freeze_inputs(root, monkeypatch, protocol, lineage=lineage)
    observed: dict[str, Any] = {}

    def workflow(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return _workflow_result(winner=winner)

    result = run_recovery_apply(
        root,
        recovery_id=RECOVERY_ID,
        tracking="online",
        applied_state_validator=lambda _root: "7" * 40,
        preflight_validator=lambda _root, _recovery_id: {},
        version_validator=lambda: None,
        prepared_loader=lambda _root: _prepared(lineage),
        tracker_factory=lambda: "synthetic-tracker",
        workflow_runner=workflow,
        protocol_loader=lambda _root: (protocol, SOURCE_PROTOCOL_SHA),
        production_validator=production_validator or (lambda _root, _protocol: {"unchanged": True}),
    )
    assert observed["advanced"] == [
        {"candidate_id": "LGBM12-UNIFORM", "family": "lightgbm"},
        {"candidate_id": "CB04-EXP120", "family": "catboost"},
    ]
    assert observed["metadata"]["group"] == f"v3-recovery-{RECOVERY_ID}"
    assert observed["metadata"]["execution_mode"] == "governed_recovery"
    return result


@pytest.mark.parametrize("winner", [True, False])
def test_recovery_winner_and_governed_stop_outputs_have_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v3_protocol: dict,
    winner: bool,
) -> None:
    result = _run_synthetic_recovery(tmp_path, monkeypatch, v3_protocol, winner=winner)
    directory = recovery_directory(tmp_path, RECOVERY_ID)
    marker = json.loads((directory / RECOVERY_MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["decision"] == ("winner" if winner else "governed_stop")
    assert marker["december_opened"] is False
    assert marker["historical_test_accessed"] is False
    assert result["execution_mode"] == "governed_recovery"
    assert result["recovery_reason"] == RECOVERY_REASON
    assert result["source_execution_implementation_sha"] == SOURCE_IMPLEMENTATION_SHA
    assert result["corrective_commit_sha"] == CORRECTIVE_COMMIT_SHA
    assert result["partial_original_november_finalist_used"] is False
    assert (directory / DEVELOPMENT_RESULT_NAME).is_file()
    assert (directory / RECOVERY_DECISION_NAME).is_file()
    assert (directory / RECOVERY_STATE_NAME).is_file()
    assert (directory / RECOVERY_WINNER_LOCK_NAME).is_file() is winner
    assert (directory / RECOVERY_WINNER_MODEL_NAME).is_file() is winner
    if winner:
        lock = json.loads((directory / RECOVERY_WINNER_LOCK_NAME).read_text(encoding="utf-8"))
        assert lock["execution_mode"] == "governed_recovery"
        assert lock["source_tracking_evidence_sha256"]


def test_recovery_requires_online_tracking_and_valid_authorization(tmp_path: Path) -> None:
    with pytest.raises(V3RecoveryError, match="online tracking"):
        run_recovery_apply(tmp_path, recovery_id=RECOVERY_ID, tracking="disabled")


def test_recovery_refuses_protocol_drift_and_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol)
    common = {
        "recovery_id": RECOVERY_ID,
        "tracking": "online",
        "applied_state_validator": lambda _root: "7" * 40,
        "preflight_validator": lambda _root, _recovery_id: {},
        "version_validator": lambda: None,
        "prepared_loader": lambda _root: _prepared(_lineage()),
        "tracker_factory": lambda: None,
        "workflow_runner": lambda **_kwargs: _workflow_result(winner=False),
        "production_validator": lambda _root, _protocol: {"unchanged": True},
    }
    with pytest.raises(V3RecoveryError, match="protocol differs"):
        run_recovery_apply(
            tmp_path,
            protocol_loader=lambda _root: (v3_protocol, "0" * 64),
            **common,
        )
    (recovery_directory(tmp_path, RECOVERY_ID) / RECOVERY_MARKER_NAME).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(V3RecoveryError, match="output already exists"):
        run_recovery_apply(
            tmp_path,
            protocol_loader=lambda _root: (v3_protocol, SOURCE_PROTOCOL_SHA),
            **common,
        )


@pytest.mark.parametrize(
    ("lineage_update", "message"),
    [
        ({"december_decoded": True}, "rebuilt development lineage mismatch|prohibited period"),
        (
            {"january_may_2026_accessed": True},
            "rebuilt development lineage mismatch|prohibited period",
        ),
    ],
)
def test_recovery_refuses_december_and_historical_test_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v3_protocol: dict,
    lineage_update: dict[str, Any],
    message: str,
) -> None:
    source_lineage = _lineage()
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol, lineage=source_lineage)
    changed = {**source_lineage, **lineage_update}
    with pytest.raises(V3RecoveryError, match=message):
        run_recovery_apply(
            tmp_path,
            recovery_id=RECOVERY_ID,
            tracking="online",
            applied_state_validator=lambda _root: "7" * 40,
            preflight_validator=lambda _root, _recovery_id: {},
            version_validator=lambda: None,
            prepared_loader=lambda _root: _prepared(changed),
            tracker_factory=lambda: None,
            workflow_runner=lambda **_kwargs: _workflow_result(winner=False),
            protocol_loader=lambda _root: (v3_protocol, SOURCE_PROTOCOL_SHA),
            production_validator=lambda _root, _protocol: {"unchanged": True},
        )
    marker = json.loads(
        (recovery_directory(tmp_path, RECOVERY_ID) / RECOVERY_MARKER_NAME).read_text()
    )
    assert marker["status"] == "failed"
    assert marker["failed_stage"] == "development_lineage"


def test_recovery_refuses_production_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    calls = 0

    def production(_root: Path, _protocol: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unchanged": calls == 1}

    with pytest.raises(V3RecoveryError, match="production v0 changed"):
        _run_synthetic_recovery(
            tmp_path,
            monkeypatch,
            v3_protocol,
            winner=False,
            production_validator=production,
        )


def test_recovery_uses_the_corrected_shared_threshold_selector() -> None:
    from flight_delay.modeling.v3 import selection

    assert selection.select_v1_threshold is select_v1_threshold
    result = selection.finalist_evidence(
        finalist_id="synthetic",
        labels=[0, 1, 0, 1],
        probabilities=[0.1, 0.9, 0.2, 0.8],
        audit_metrics={
            "equal_frequency_ece_15": 0.0,
            "serialized_bundle_bytes": 1,
            "single_row_inference_p95_ms": 1.0,
        },
        governance={},
        protocol={
            "november_selection": {
                "threshold_objective": {
                    "eligibility": {
                        "recall_min": 0.0,
                        "precision_min": 0.0,
                        "predicted_positive_rate_max": 1.0,
                    }
                },
                "acceptance_gates": {
                    "operating_point": {
                        "recall_min": 0.0,
                        "precision_min": 0.0,
                        "f1_min": 0.0,
                        "predicted_positive_rate_max": 1.0,
                    },
                    "probability": {
                        "absolute_probability_prevalence_gap_max": 1.0,
                        "equal_frequency_ece_15_max": 1.0,
                    },
                    "discrimination": {
                        "average_precision_absolute_min": 0.0,
                        "roc_auc_absolute_min": 0.0,
                    },
                    "operational": {
                        "single_row_inference_p95_ms_strict_max": 100.0,
                        "serialized_bundle_bytes_strict_max": 100,
                    },
                    "governance": {},
                },
            }
        },
    )
    assert result["threshold_selection"]["threshold_table"]


def test_adoption_preserves_marker_and_refuses_existing_canonical_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _run_synthetic_recovery(tmp_path, monkeypatch, v3_protocol, winner=True)
    source_marker = tmp_path / "source" / DEVELOPMENT_MARKER
    canonical_marker = tmp_path / DEVELOPMENT_MARKER
    canonical_marker.parent.mkdir(parents=True, exist_ok=True)
    canonical_marker.write_bytes(source_marker.read_bytes())
    marker_before = canonical_marker.read_bytes()
    preview = adoption_preflight(tmp_path, recovery_id=RECOVERY_ID)
    assert preview["original_marker_will_be_rewritten"] is False

    adoption = adopt_recovery(tmp_path, recovery_id=RECOVERY_ID)
    assert adoption["status"] == "adopted"
    assert canonical_marker.read_bytes() == marker_before
    assert (tmp_path / DECISION_PATH).is_file()
    validate_recovery_adoption_for_december(
        tmp_path, json.loads(canonical_marker.read_text(encoding="utf-8"))
    )
    monkeypatch.setattr(
        v3_execution.V3HistoricalState,
        "from_bytes",
        classmethod(lambda _cls, _payload: FakeState(_lineage()["november_state_sha256"])),
    )
    winner, restored = v3_execution.validate_december_handoff(tmp_path)
    assert winner["execution_mode"] == "governed_recovery"
    assert restored.sha256 == _lineage()["november_state_sha256"]
    with pytest.raises(V3RecoveryError, match="canonical decision already exists|adoption already"):
        adopt_recovery(tmp_path, recovery_id=RECOVERY_ID)


def test_adoption_fails_if_canonical_decision_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _run_synthetic_recovery(tmp_path, monkeypatch, v3_protocol, winner=False)
    source_marker = tmp_path / "source" / DEVELOPMENT_MARKER
    canonical_marker = tmp_path / DEVELOPMENT_MARKER
    canonical_marker.parent.mkdir(parents=True, exist_ok=True)
    canonical_marker.write_bytes(source_marker.read_bytes())
    _write_json(tmp_path / DECISION_PATH, {"decision": "existing"})
    with pytest.raises(V3RecoveryError, match="canonical decision already exists"):
        adopt_recovery(tmp_path, recovery_id=RECOVERY_ID)


def test_safe_recovery_identifiers_only(tmp_path: Path) -> None:
    assert recovery_directory(tmp_path, "safe-id.1").is_relative_to(tmp_path)
    with pytest.raises(V3RecoveryError, match="safe"):
        recovery_directory(tmp_path, "../escape")


def test_recovery_preflight_and_runtime_estimate_are_static(v3_protocol: dict) -> None:
    report = recovery_preflight(ROOT, recovery_id=RECOVERY_ID)
    assert report["parquet_opened"] is False
    assert report["model_fit_started"] is False
    assert report["network_contacted"] is False
    assert report["screening_will_repeat"] is False
    assert report["cpu_confirmation_will_repeat"] is False
    assert report["november_finalists_from_scratch"] == 15
    assert report["runtime_estimate"]["expected_wall_clock_minutes"] == {
        "lower": 60,
        "upper": 90,
    }
    assert estimate_recovery_runtime(Path("/tmp/no-v3-manifest-here"), v3_protocol) is None


def test_runtime_estimate_requires_positive_rows(tmp_path: Path, v3_protocol: dict) -> None:
    write_manifest(
        tmp_path / "data/manifests/v3_processed_manifest.json",
        {"schema_version": 1, "monthly_counts": []},
    )
    with pytest.raises(V3RecoveryError, match="full-refit row counts"):
        estimate_recovery_runtime(tmp_path, v3_protocol)


def test_recovery_applied_state_requires_corrective_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "require_merged_applied_state", lambda _root: "a" * 40)
    monkeypatch.setattr(v3_execution, "_git", lambda *_args: "")
    assert require_recovery_applied_state(tmp_path) == "a" * 40

    def fail(*_args: Any) -> str:
        raise RuntimeError("not ancestor")

    monkeypatch.setattr(v3_execution, "_git", fail)
    with pytest.raises(V3RecoveryError, match="corrected selector commit"):
        require_recovery_applied_state(tmp_path)


def test_recovery_tracker_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    with pytest.raises(V3RecoveryError, match="WANDB_ENTITY"):
        recovery._recovery_tracker()
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_PROJECT", "project")
    tracker = recovery._recovery_tracker()
    assert tracker.entity == "entity"
    assert tracker.project == "project"


def test_termination_record_load_checks_payload_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol)
    record, digest = load_termination_record(tmp_path, recovery_id=RECOVERY_ID)
    assert record["process_inspected_or_signaled_by_recorder"] is False
    assert len(digest) == 64
    path = recovery_directory(tmp_path, RECOVERY_ID) / TERMINATION_RECORD_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decision_absent"] = False
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(V3RecoveryError, match="tampered"):
        load_termination_record(tmp_path, recovery_id=RECOVERY_ID)


def test_authorization_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_protocol: dict
) -> None:
    _freeze_inputs(tmp_path, monkeypatch, v3_protocol)
    with pytest.raises(V3RecoveryError, match="immutable output"):
        create_authorization(
            tmp_path,
            protocol=v3_protocol,
            recovery_id=RECOVERY_ID,
            corrected_selector_test_evidence={"passed": True},
            corrected_selector_benchmark_evidence={"passed": True},
        )
    assert (recovery_directory(tmp_path, RECOVERY_ID) / AUTHORIZATION_NAME).is_file()
    assert (recovery_directory(tmp_path, RECOVERY_ID) / TERMINATION_RECORD_NAME).is_file()
    assert (tmp_path / RECOVERY_ADOPTION).exists() is False
