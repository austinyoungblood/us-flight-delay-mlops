from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from flight_delay.modeling import v1_execution
from flight_delay.modeling.v1_data import (
    V1_CATEGORICAL_FEATURES,
    V1_FEATURES,
    DevelopmentData,
)
from flight_delay.modeling.v1_execution import (
    DEVELOPMENT_MARKER,
    QUALIFICATION_MARKER,
    RELEASE_CANDIDATE_LOCK,
    WINNER_LOCK,
    WINNER_MODEL,
    V1ExecutionError,
    run_december_apply,
    run_development_apply,
    validate_december_handoff,
)
from flight_delay.modeling.v1_protocol import sha256_file
from flight_delay.modeling.v1_selection import (
    GateEvidence,
    evaluate_november_gates,
)
from flight_delay.modeling.v1_tracking import NullTracker

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA = "a6b1de9de550d1bd94eae0e56f8d88d65801ec488b6c539fc64afbafa4ccfffb"
CODE_SHA = "b" * 40


class _FrozenProbabilityModel:
    """Serializable deterministic stand-in that refuses any retraining."""

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("the frozen model must never be retrained")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = np.where(features["Distance"].to_numpy(dtype=float) > 150.0, 0.99, 0.01)
        return np.column_stack((1.0 - scores, scores))


def _protocol() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())


def _protocol_lock() -> dict[str, Any]:
    return json.loads((ROOT / "experiments/v1/protocol_lock.json").read_text())


def _canonical_frame(start: str, rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows)
    payload: dict[str, object] = {"flight_date": dates}
    for index, column in enumerate(V1_FEATURES):
        if column == "Reporting_Airline":
            payload[column] = ["UA"] * rows
        elif column == "Origin":
            payload[column] = ["DEN"] * rows
        elif column == "Dest":
            payload[column] = ["SFO"] * rows
        elif column == "route":
            payload[column] = ["DEN-SFO"] * rows
        elif column == "Distance":
            payload[column] = [100.0, 200.0] * (rows // 2)
        else:
            payload[column] = np.arange(rows, dtype=float) + index + 1
    payload["target"] = [0, 1] * (rows // 2)
    return pd.DataFrame(payload, columns=("flight_date", *V1_FEATURES, "target"))


def _manifest() -> dict[str, Any]:
    return {
        "manifest_digest": "manifest-digest",
        "parquet_files": {
            "train": {"sha256": "train-sha"},
            "validation": {"sha256": "validation-sha"},
        },
    }


def _development_data() -> DevelopmentData:
    protocol = _protocol()
    return DevelopmentData(
        train=_canonical_frame("2025-01-01"),
        november=_canonical_frame("2025-11-01"),
        manifest=_manifest(),
        protocol=protocol,
        protocol_sha256=PROTOCOL_SHA,
    )


def _patch_static_applied_contracts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracker: NullTracker | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _protocol()
    lock = _protocol_lock()
    monkeypatch.setattr(v1_execution, "require_applied_git_state", lambda _root: CODE_SHA)
    monkeypatch.setattr(
        v1_execution,
        "preflight",
        lambda _root, *, stage: {"stage": stage, "catboost_version_ready": True},
    )
    monkeypatch.setattr(
        v1_execution,
        "load_and_validate_v1_protocol",
        lambda *_args, **_kwargs: (protocol, lock, PROTOCOL_SHA),
    )
    monkeypatch.setattr(v1_execution, "validate_catboost_runtime_contract", lambda _protocol: {})
    if tracker is not None:
        monkeypatch.setattr(v1_execution, "_tracker_from_environment", lambda _tracking: tracker)
    return protocol, lock


def _instrument_marker_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    transitions: list[tuple[str, str]] = []
    create = v1_execution.create_marker
    update = v1_execution.update_marker

    def recording_create(path: Path, payload: dict[str, Any]) -> None:
        transitions.append((path.name, str(payload["status"])))
        create(path, payload)

    def recording_update(path: Path, payload: dict[str, Any]) -> None:
        transitions.append((path.name, str(payload["status"])))
        update(path, payload)

    monkeypatch.setattr(v1_execution, "create_marker", recording_create)
    monkeypatch.setattr(v1_execution, "update_marker", recording_update)
    return transitions


def _rolling_results(tracker: NullTracker, common: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(("CB1", "CB2", "CB3", "CB4")):
        with tracker.start_run(
            name=f"v1-{candidate_id}-rolling",
            group=common["group"],
            metadata={**common, "stage": "rolling", "candidate_id": candidate_id},
        ) as run:
            run.log({"FOLD_1/average_precision": 0.8 - index / 100})
        results.append(
            {
                "candidate_id": candidate_id,
                "status": "completed",
                "mean_average_precision": 0.8 - index / 100,
                "mean_roc_auc": 0.9 - index / 100,
                "mean_log_loss": 0.2 + index / 100,
                "mean_brier_score": 0.05 + index / 100,
                "std_average_precision": 0.01,
                "wandb_run_id": run.id,
                "wandb_run_url": run.url,
            }
        )
    return results


def _passing_metrics(*, bundle_bytes: int, average_precision: float) -> dict[str, Any]:
    return {
        "prevalence": 0.5,
        "average_precision": average_precision,
        "roc_auc": 0.99,
        "brier_score": 0.01,
        "log_loss": 0.02,
        "probability_mean": 0.5,
        "probability_min": 0.01,
        "probability_max": 0.99,
        "probability_median": 0.5,
        "probability_std": 0.49,
        "prior_brier_score": 0.25,
        "prior_log_loss": 0.693,
        "brier_skill_score": 0.96,
        "average_precision_lift_over_prevalence": 2.0,
        "absolute_probability_prevalence_gap": 0.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "predicted_positive_rate": 0.5,
        "equal_frequency_ece_15": 0.01,
        "single_row_inference_p95_ms": 1.0,
        "serialized_bundle_bytes": bundle_bytes,
    }


def _patch_development_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracker: NullTracker,
    winner_enabled: bool,
    accesses: list[str],
    calibration_calls: list[tuple[str, bool, bool]],
    bundle_events: list[tuple[str, str]],
) -> None:
    data = _development_data()

    def load_development(_root: Path) -> DevelopmentData:
        accesses.extend(("train:canonical:2025-01-01/2025-10-31", "validation:canonical:november"))
        return data

    monkeypatch.setattr(v1_execution, "load_development_data", load_development)
    monkeypatch.setattr(
        v1_execution,
        "reconstruct_governed_r3",
        lambda _train, _november: {
            "metrics": {"average_precision": 0.28},
            "reproduction": {"all_metrics_reproduced": True},
        },
    )
    monkeypatch.setattr(
        v1_execution,
        "run_r3_rolling_context",
        lambda _train, _protocol: [
            {"fold_id": f"FOLD_{index}", "average_precision": 0.25} for index in range(1, 5)
        ],
    )
    monkeypatch.setattr(
        v1_execution,
        "run_catboost_rolling",
        lambda *, protocol, train, tracker, common: _rolling_results(tracker, common),
    )
    partitions = SimpleNamespace(
        final_fit=_canonical_frame("2025-01-01"),
        calibration=_canonical_frame("2025-11-01"),
        selection=_canonical_frame("2025-11-16"),
    )
    monkeypatch.setattr(
        v1_execution, "partition_remediation_data", lambda _train, _november: partitions
    )
    monkeypatch.setattr(
        v1_execution,
        "build_catboost_candidate",
        lambda _protocol, _candidate_id: _FrozenProbabilityModel(),
    )
    monkeypatch.setattr(
        v1_execution,
        "fit_catboost_base",
        lambda model, _features, _target, _dates: model,
    )

    def calibration_variant(
        base: _FrozenProbabilityModel,
        *,
        method: str,
        calibration_features: pd.DataFrame | None,
        calibration_target: pd.Series | None,
    ) -> _FrozenProbabilityModel:
        calibration_calls.append(
            (method, calibration_features is not None, calibration_target is not None)
        )
        return base

    monkeypatch.setattr(v1_execution, "build_calibration_variant", calibration_variant)

    def finalist_evidence(
        *,
        model: _FrozenProbabilityModel,
        finalist_id: str,
        spec: Any,
        method: str,
        selection: Any,
        protocol: dict[str, Any],
        protocol_lock: dict[str, Any],
        common: dict[str, Any],
        bundle_directory: Path,
    ) -> dict[str, Any]:
        metrics = _passing_metrics(
            bundle_bytes=1,
            average_precision=0.99 if finalist_id == "CB1-none" else 0.90,
        )
        bundle = v1_execution.write_candidate_bundle(
            directory=bundle_directory,
            model=model,
            candidate_id=finalist_id,
            parameters=spec.parameters,
            calibration_method=method,
            threshold=0.37,
            metrics=metrics,
            metadata=common,
            protocol_lock=protocol_lock,
            verification_features=selection.features,
        )
        bundle_events.append((finalist_id, "written"))
        v1_execution.load_verified_bundle(bundle)
        bundle_events.append((finalist_id, "verified"))
        metrics["serialized_bundle_bytes"] = bundle.byte_size
        governance = {
            "lineage_verified": True,
            "schema_check_passed": True,
            "leakage_check_passed": True,
            "deterministic_reconstruction_check_passed": True,
            "serialization_load_inference_check_passed": True,
            "no_prohibited_test_access": True,
            "no_training_convergence_or_runtime_failure": True,
        }
        gates = evaluate_november_gates(metrics=metrics, protocol=protocol, governance=governance)
        if not (winner_enabled and finalist_id == "CB1-none"):
            gates = (replace(gates[0], passed=False), *gates[1:])
        bundle_events.append((finalist_id, "gated"))
        return {
            "finalist_id": finalist_id,
            "status": "completed",
            "base_candidate_id": spec.candidate_id,
            "calibration_method": method,
            "model": model,
            "parameters": spec.parameters,
            "threshold": 0.37,
            "threshold_selection": {"selected_threshold": 0.37},
            "metrics": metrics,
            "gate_evidence": gates,
            "bundle": bundle,
        }

    monkeypatch.setattr(v1_execution, "_finalist_evidence", finalist_evidence)


def test_development_apply_writes_exact_winner_and_stops_before_december(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = NullTracker()
    protocol, _lock = _patch_static_applied_contracts(monkeypatch, tracker=tracker)
    transitions = _instrument_marker_transitions(monkeypatch)
    accesses: list[str] = []
    calibration_calls: list[tuple[str, bool, bool]] = []
    bundle_events: list[tuple[str, str]] = []
    _patch_development_boundaries(
        monkeypatch,
        tracker=tracker,
        winner_enabled=True,
        accesses=accesses,
        calibration_calls=calibration_calls,
        bundle_events=bundle_events,
    )

    result = run_development_apply(tmp_path, tracking="online")

    assert transitions == [
        ("execution_marker.json", "started"),
        ("execution_marker.json", "complete"),
    ]
    assert result == {
        "decision": "winner",
        "rolling": result["rolling"],
        "r3_rolling_context": result["r3_rolling_context"],
        "finalist_count": 6,
        "stopped_before_december": True,
    }
    marker = json.loads((tmp_path / DEVELOPMENT_MARKER).read_text())
    winner = json.loads((tmp_path / WINNER_LOCK).read_text())
    decision = json.loads((tmp_path / "artifacts/v1/development/decision.json").read_text())
    assert marker["status"] == "complete" and marker["decision"] == "winner"
    assert decision["decision"] == "winner" and decision["production_remains"] == "v0"
    assert winner["protocol_id"] == protocol["protocol_id"]
    assert winner["protocol_sha256"] == PROTOCOL_SHA
    assert winner["implementation_git_sha"] == CODE_SHA
    assert winner["candidate_id"] == "CB1-none"
    assert winner["base_candidate_id"] == "CB1"
    assert winner["calibration_method"] == "none"
    assert winner["threshold"] == 0.37
    assert winner["feature_schema"] == list(V1_FEATURES)
    assert winner["categorical_schema"] == list(V1_CATEGORICAL_FEATURES)
    assert winner["dataset_manifest_identity"] == "manifest-digest"
    assert winner["train_parquet_sha256"] == "train-sha"
    assert winner["validation_parquet_sha256"] == "validation-sha"
    assert winner["model_file_sha256"] == sha256_file(tmp_path / WINNER_MODEL)
    assert winner["candidate_bundle_bytes"] > 0
    assert len(winner["all_gate_evidence"]) == 23
    assert all(gate["passed"] for gate in winner["all_gate_evidence"])
    size_gate = next(
        gate
        for gate in winner["all_gate_evidence"]
        if gate["gate_name"] == "serialized_bundle_bytes"
    )
    assert size_gate["observed"] == winner["candidate_bundle_bytes"]
    assert winner["december_evaluated"] is False
    assert not (tmp_path / QUALIFICATION_MARKER).exists()
    assert accesses == [
        "train:canonical:2025-01-01/2025-10-31",
        "validation:canonical:november",
    ]
    assert "test.parquet" not in " ".join(accesses)
    assert calibration_calls == [
        (method, method != "none", method != "none")
        for _candidate in range(2)
        for method in ("none", "sigmoid", "isotonic")
    ]
    assert [event for event in bundle_events if event[1] == "verified"] == [
        (f"{candidate}-{method}", "verified")
        for candidate in ("CB1", "CB2")
        for method in ("none", "sigmoid", "isotonic")
    ]
    for finalist_id in {name for name, _event in bundle_events}:
        assert [event for name, event in bundle_events if name == finalist_id] == [
            "written",
            "verified",
            "gated",
        ]
    stages = [run.metadata["stage"] for run in tracker.runs]
    assert stages.count("rolling") == 4
    assert stages.count("november_finalist") == 6
    assert stages == [
        "r3_reconstruction_and_rolling_context",
        "rolling",
        "rolling",
        "rolling",
        "rolling",
        "november_finalist",
        "november_finalist",
        "november_finalist",
        "november_finalist",
        "november_finalist",
        "november_finalist",
        "decision",
    ]
    assert all(run.metadata["protocol_sha256"] == PROTOCOL_SHA for run in tracker.runs)
    assert all(run.metadata["implementation_git_sha"] == CODE_SHA for run in tracker.runs)
    assert not hasattr(tracker, "log_artifact") and not hasattr(tracker, "link_artifact")
    assert not (tmp_path / "release").exists() and not (tmp_path / "deploy").exists()


def test_development_apply_records_governed_stop_without_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = NullTracker()
    _patch_static_applied_contracts(monkeypatch, tracker=tracker)
    transitions = _instrument_marker_transitions(monkeypatch)
    accesses: list[str] = []
    calibration_calls: list[tuple[str, bool, bool]] = []
    bundle_events: list[tuple[str, str]] = []
    _patch_development_boundaries(
        monkeypatch,
        tracker=tracker,
        winner_enabled=False,
        accesses=accesses,
        calibration_calls=calibration_calls,
        bundle_events=bundle_events,
    )

    result = run_development_apply(tmp_path, tracking="online")

    assert result["decision"] == "governed_stop" and result["finalist_count"] == 6
    assert transitions == [
        ("execution_marker.json", "started"),
        ("execution_marker.json", "complete"),
    ]
    marker = json.loads((tmp_path / DEVELOPMENT_MARKER).read_text())
    decision = json.loads((tmp_path / "artifacts/v1/development/decision.json").read_text())
    assert marker["decision"] == "governed_stop"
    assert decision["decision"] == "governed_stop"
    assert len(decision["finalists"]) == 6
    assert all(
        not all(gate["passed"] for gate in row["gate_evidence"]) for row in decision["finalists"]
    )
    assert (tmp_path / "artifacts/v1/development/stop-report.md").is_file()
    assert not (tmp_path / WINNER_MODEL).exists()
    assert not (tmp_path / WINNER_LOCK).exists()
    assert accesses == [
        "train:canonical:2025-01-01/2025-10-31",
        "validation:canonical:november",
    ]
    assert len(tracker.runs) == 12
    assert not (tmp_path / "release").exists() and not (tmp_path / "deploy").exists()


def test_development_failure_marker_is_durable_and_retry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = NullTracker()
    _patch_static_applied_contracts(monkeypatch, tracker=tracker)
    transitions = _instrument_marker_transitions(monkeypatch)
    accesses: list[str] = []

    def load_development(_root: Path) -> DevelopmentData:
        accesses.extend(("train:canonical", "validation:canonical:november"))
        return _development_data()

    monkeypatch.setattr(v1_execution, "load_development_data", load_development)
    monkeypatch.setattr(
        v1_execution,
        "reconstruct_governed_r3",
        lambda _train, _november: (_ for _ in ()).throw(RuntimeError("sensitive detail")),
    )

    with pytest.raises(RuntimeError, match="sensitive detail"):
        run_development_apply(tmp_path, tracking="online")

    marker = json.loads((tmp_path / DEVELOPMENT_MARKER).read_text())
    assert transitions == [
        ("execution_marker.json", "started"),
        ("execution_marker.json", "failed"),
    ]
    assert marker["status"] == "failed"
    assert marker["sanitized_error_type"] == "RuntimeError"
    assert marker["failed_stage"] == "r3_reconstruction"
    assert "sensitive detail" not in json.dumps(marker)
    assert not (tmp_path / WINNER_MODEL).exists()
    assert not (tmp_path / WINNER_LOCK).exists()
    assert not (tmp_path / "artifacts/v1/development/decision.json").exists()

    with pytest.raises(V1ExecutionError, match="rerun prohibited"):
        run_development_apply(tmp_path, tracking="online")
    assert accesses == ["train:canonical", "validation:canonical:november"]


def test_development_refuses_non_online_tracking_before_data_or_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_static_applied_contracts(monkeypatch)
    data_accesses: list[str] = []

    def forbidden_data_load(_root: Path) -> DevelopmentData:
        data_accesses.append("unexpected")
        raise AssertionError("data must not be loaded")

    monkeypatch.setattr(
        v1_execution,
        "load_development_data",
        forbidden_data_load,
    )
    monkeypatch.setattr(
        v1_execution,
        "build_catboost_candidate",
        lambda *_args: (_ for _ in ()).throw(AssertionError("model must not be built")),
    )

    with pytest.raises(V1ExecutionError, match="requires --tracking online"):
        run_development_apply(tmp_path, tracking="disabled")
    assert data_accesses == []
    assert not (tmp_path / DEVELOPMENT_MARKER).exists()


def _write_november_winner(
    root: Path,
    *,
    protocol_sha: str = PROTOCOL_SHA,
    protocol_id: str | None = None,
    code_sha: str = CODE_SHA,
) -> dict[str, Any]:
    development = root / "artifacts/v1/development"
    development.mkdir(parents=True, exist_ok=True)
    (root / DEVELOPMENT_MARKER).write_text(
        json.dumps({"status": "complete", "decision": "winner"}), encoding="utf-8"
    )
    model_path = root / WINNER_MODEL
    joblib.dump(_FrozenProbabilityModel(), model_path)
    winner = {
        "protocol_id": protocol_id or _protocol()["protocol_id"],
        "protocol_sha256": protocol_sha,
        "implementation_git_sha": code_sha,
        "candidate_id": "CB1-none",
        "calibration_method": "none",
        "model_file_sha256": sha256_file(model_path),
        "feature_schema": list(V1_FEATURES),
        "categorical_schema": list(V1_CATEGORICAL_FEATURES),
        "threshold": 0.5,
        "candidate_bundle_bytes": 4096,
        "december_evaluated": False,
    }
    (root / WINNER_LOCK).write_text(json.dumps(winner), encoding="utf-8")
    return winner


def _patch_december_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracker: NullTracker,
    accesses: list[str],
) -> None:
    _patch_static_applied_contracts(monkeypatch, tracker=tracker)

    def load_december(_root: Path) -> pd.DataFrame:
        accesses.append("validation:canonical:2025-12-01/2025-12-31")
        return _canonical_frame("2025-12-01")

    monkeypatch.setattr(v1_execution, "load_december_data", load_december)
    monkeypatch.setattr(
        v1_execution,
        "measure_single_row_latency",
        lambda _model, _features: {"p95_ms": 1.0},
    )
    monkeypatch.setattr(v1_execution, "read_manifest", lambda _path: _manifest())


def test_december_apply_passes_once_without_refit_recalibration_or_threshold_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    winner_before = _write_november_winner(tmp_path)
    tracker = NullTracker()
    accesses: list[str] = []
    _patch_december_boundaries(monkeypatch, tracker=tracker, accesses=accesses)
    transitions = _instrument_marker_transitions(monkeypatch)
    evaluated_thresholds: list[float] = []
    threshold_metrics = v1_execution._threshold_metrics

    def record_threshold(labels: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
        evaluated_thresholds.append(threshold)
        return threshold_metrics(labels, scores, threshold)

    monkeypatch.setattr(v1_execution, "_threshold_metrics", record_threshold)

    result = run_december_apply(tmp_path, tracking="online")

    assert result["passed"] is True
    assert result["candidate_id"] == "CB1-none"
    assert result["winner_model_sha256"] == winner_before["model_file_sha256"]
    assert result["production_remains"] == "v0"
    assert len(result["gate_evidence"]) == 13
    assert all(gate["passed"] for gate in result["gate_evidence"])
    assert transitions == [
        ("execution_marker.json", "started"),
        ("execution_marker.json", "complete"),
    ]
    marker = json.loads((tmp_path / QUALIFICATION_MARKER).read_text())
    release = json.loads((tmp_path / RELEASE_CANDIDATE_LOCK).read_text())
    result_file = json.loads(
        (tmp_path / "artifacts/v1/qualification/qualification_result.json").read_text()
    )
    assert marker["decision"] == "release_candidate"
    assert release["same_frozen_november_model"] is True
    assert release["winner_model_sha256"] == winner_before["model_file_sha256"]
    assert release["registry_artifact_created"] is False
    assert release["production_promoted"] is False
    assert result_file["passed"] is True
    assert evaluated_thresholds == [winner_before["threshold"]]
    assert json.loads((tmp_path / WINNER_LOCK).read_text()) == winner_before
    assert accesses == ["validation:canonical:2025-12-01/2025-12-31"]
    assert len(tracker.runs) == 1
    assert tracker.runs[0].metadata["stage"] == "december_qualification"
    assert tracker.runs[0].metadata["protocol_sha256"] == PROTOCOL_SHA
    assert tracker.runs[0].metadata["implementation_git_sha"] == CODE_SHA
    assert not hasattr(tracker, "log_artifact") and not hasattr(tracker, "link_artifact")
    assert not (tmp_path / "release").exists() and not (tmp_path / "deploy").exists()

    with pytest.raises(V1ExecutionError, match="already started or evaluated"):
        run_december_apply(tmp_path, tracking="online")
    assert accesses == ["validation:canonical:2025-12-01/2025-12-31"]


def test_december_apply_records_gate_failure_without_release_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_november_winner(tmp_path)
    tracker = NullTracker()
    accesses: list[str] = []
    _patch_december_boundaries(monkeypatch, tracker=tracker, accesses=accesses)
    evaluate = v1_execution.evaluate_qualification_gates

    def fail_one_gate(**kwargs: Any) -> tuple[GateEvidence, ...]:
        gates = evaluate(**kwargs)
        return (replace(gates[0], passed=False), *gates[1:])

    monkeypatch.setattr(v1_execution, "evaluate_qualification_gates", fail_one_gate)

    result = run_december_apply(tmp_path, tracking="online")

    assert result["passed"] is False
    assert not (tmp_path / RELEASE_CANDIDATE_LOCK).exists()
    marker = json.loads((tmp_path / QUALIFICATION_MARKER).read_text())
    persisted = json.loads(
        (tmp_path / "artifacts/v1/qualification/qualification_result.json").read_text()
    )
    assert marker["status"] == "complete" and marker["decision"] == "qualification_failed"
    assert persisted["passed"] is False
    assert any(not gate["passed"] for gate in persisted["gate_evidence"])
    assert persisted["production_remains"] == "v0"
    assert accesses == ["validation:canonical:2025-12-01/2025-12-31"]
    assert not (tmp_path / "release").exists() and not (tmp_path / "deploy").exists()


def test_december_failure_marker_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_november_winner(tmp_path)
    tracker = NullTracker()
    accesses: list[str] = []
    _patch_december_boundaries(monkeypatch, tracker=tracker, accesses=accesses)

    def failed_december_load(_root: Path) -> pd.DataFrame:
        accesses.append("validation:canonical:2025-12-01/2025-12-31")
        raise RuntimeError("private failure")

    monkeypatch.setattr(
        v1_execution,
        "load_december_data",
        failed_december_load,
    )

    with pytest.raises(RuntimeError, match="private failure"):
        run_december_apply(tmp_path, tracking="online")

    marker = json.loads((tmp_path / QUALIFICATION_MARKER).read_text())
    assert marker["status"] == "failed"
    assert marker["failed_stage"] == "december_data_guard"
    assert marker["sanitized_error_type"] == "RuntimeError"
    assert "private failure" not in json.dumps(marker)
    assert not (tmp_path / RELEASE_CANDIDATE_LOCK).exists()
    assert accesses == ["validation:canonical:2025-12-01/2025-12-31"]


def test_december_handoff_rejects_missing_and_corrupt_winner_state(tmp_path: Path) -> None:
    missing_lock = tmp_path / "missing-lock"
    development = missing_lock / "artifacts/v1/development"
    development.mkdir(parents=True)
    (missing_lock / DEVELOPMENT_MARKER).write_text(
        json.dumps({"status": "complete", "decision": "winner"}), encoding="utf-8"
    )
    with pytest.raises(V1ExecutionError, match="cannot read governed state"):
        validate_december_handoff(missing_lock)

    missing_model = tmp_path / "missing-model"
    _write_november_winner(missing_model)
    (missing_model / WINNER_MODEL).unlink()
    with pytest.raises(V1ExecutionError, match="model hash mismatch"):
        validate_december_handoff(missing_model)

    bad_schema = tmp_path / "bad-schema"
    _write_november_winner(bad_schema)
    lock_path = bad_schema / WINNER_LOCK
    lock = json.loads(lock_path.read_text())
    lock["feature_schema"] = ["wrong"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(V1ExecutionError, match="schema mismatch"):
        validate_december_handoff(bad_schema)

    bad_threshold = tmp_path / "bad-threshold"
    _write_november_winner(bad_threshold)
    lock_path = bad_threshold / WINNER_LOCK
    lock = json.loads(lock_path.read_text())
    lock["threshold"] = "changed"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(V1ExecutionError, match="threshold is invalid"):
        validate_december_handoff(bad_threshold)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_sha256", "wrong", "protocol SHA mismatch"),
        ("protocol_id", "wrong", "protocol ID mismatch"),
        ("implementation_git_sha", "wrong", "implementation lineage differs"),
    ],
)
def test_december_refuses_winner_lineage_drift_before_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    winner = _write_november_winner(tmp_path)
    winner[field] = value
    (tmp_path / WINNER_LOCK).write_text(json.dumps(winner), encoding="utf-8")
    tracker = NullTracker()
    accesses: list[str] = []
    _patch_december_boundaries(monkeypatch, tracker=tracker, accesses=accesses)

    with pytest.raises(V1ExecutionError, match=message):
        run_december_apply(tmp_path, tracking="online")
    assert accesses == []
    assert not (tmp_path / QUALIFICATION_MARKER).exists()
