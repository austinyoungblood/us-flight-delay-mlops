from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from flight_delay.modeling import v1_execution
from flight_delay.modeling.v1_data import V1_FEATURES
from flight_delay.modeling.v1_execution import (
    CandidateBundle,
    V1ExecutionError,
    _atomic_text,
    _finalist_evidence,
    _read_json_object,
    _sanitized_finalist,
    _tracker_from_environment,
    create_marker,
    load_verified_bundle,
    preflight,
    require_r3_reconstruction,
    run_catboost_rolling,
    update_marker,
    validate_december_handoff,
    validate_dependency_isolation,
    validate_production_v0,
    write_candidate_bundle,
)
from flight_delay.modeling.v1_selection import GateEvidence
from flight_delay.modeling.v1_tracking import NullTracker, WandbTracker

ROOT = Path(__file__).resolve().parents[2]


class _SerializableModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = np.asarray(features.iloc[:, 0], dtype=float)
        scores = (scores - scores.min() + 1) / (scores.max() - scores.min() + 2)
        return np.column_stack((1 - scores, scores))


class _DistanceModel:
    def fit(self, *_args: object, **_kwargs: object) -> _DistanceModel:
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = np.where(features["Distance"].to_numpy() > 150, 0.8, 0.2)
        return np.column_stack((1 - scores, scores))


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {column: np.arange(5, dtype=float) + index for index, column in enumerate(V1_FEATURES)}
    )


def _canonical_frame(dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = len(dates)
    payload: dict[str, object] = {"flight_date": dates}
    for column in V1_FEATURES:
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
            payload[column] = np.arange(rows, dtype=float) + 1
    payload["target"] = [0, 1] * (rows // 2)
    return pd.DataFrame(payload, columns=("flight_date", *V1_FEATURES, "target"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dependency_contract_is_optional_and_runtime_isolated() -> None:
    evidence = validate_dependency_isolation(ROOT)
    assert evidence["v1_extra"] == ["catboost==1.2.10"]
    assert evidence["runtime_images_install_base_only"] is True
    assert evidence["v1_constraints"] == [
        "catboost==1.2.10",
        "graphviz==0.21",
        "plotly==6.5.0",
    ]


def test_preflight_is_read_only_and_reports_the_full_locked_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parquet opened")),
    )
    report = preflight(ROOT, stage="development")
    assert report["candidates"] == ["CB1", "CB2", "CB3", "CB4"]
    assert report["rolling_folds"] == ["FOLD_1", "FOLD_2", "FOLD_3", "FOLD_4"]
    assert report["calibration_variants"] == ["none", "sigmoid", "isotonic"]
    assert report["parquet_opened"] is False
    assert report["historical_test_accessed"] is False
    assert report["stops_before_december"] is True
    qualification = preflight(ROOT, stage="qualification")
    assert qualification["requires_completed_november_winner_lock"] is True
    assert qualification["retraining_permitted"] is False
    assert qualification["recalibration_permitted"] is False


def test_dry_run_subprocess_refuses_forbidden_imports_network_and_parquet(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import socket
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'catboost', 'wandb', 'boto3', 'botocore'}:
            raise AssertionError('forbidden import: ' + fullname)
        return None
import sys
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
import pandas as pd
pd.read_parquet = lambda *a, **k: (_ for _ in ()).throw(AssertionError('parquet'))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{tmp_path}:{ROOT / 'src'}"
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_v1_development.py")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["catboost_runtime_imported"] is False
    assert payload["wandb_imported"] is False
    assert payload["network_contacted"] is False
    assert payload["parquet_opened"] is False


def test_marker_is_durable_and_duplicate_creation_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "execution_marker.json"
    create_marker(marker, {"status": "started", "historical_test_accessed": False})
    with pytest.raises(V1ExecutionError, match="already exists"):
        create_marker(marker, {"status": "started"})
    update_marker(marker, {"status": "failed", "failed_stage": "fit"})
    assert json.loads(marker.read_text()) == {
        "failed_stage": "fit",
        "historical_test_accessed": False,
        "status": "failed",
    }


def test_atomic_text_and_state_readers_fail_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _atomic_text(report, "safe\n", refuse_existing=True)
    assert report.read_text() == "safe\n"
    with pytest.raises(V1ExecutionError, match="already exists"):
        _atomic_text(report, "replacement\n", refuse_existing=True)
    with pytest.raises(V1ExecutionError, match="does not exist"):
        update_marker(tmp_path / "missing.json", {"status": "failed"})
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(V1ExecutionError, match="unreadable"):
        update_marker(malformed, {"status": "failed"})
    with pytest.raises(V1ExecutionError, match="cannot read governed state"):
        _read_json_object(tmp_path / "absent.json")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(V1ExecutionError, match="must be an object"):
        _read_json_object(scalar)


def test_candidate_bundle_round_trip_hashes_size_and_corruption(tmp_path: Path) -> None:
    protocol_lock = json.loads((ROOT / "experiments/v1/protocol_lock.json").read_text())
    bundle = write_candidate_bundle(
        directory=tmp_path / "candidate",
        model=_SerializableModel(),
        candidate_id="CB1-none",
        parameters={"iterations": 300},
        calibration_method="none",
        threshold=0.5,
        metrics={"average_precision": 0.5},
        metadata={"protocol_id": "test", "implementation_git_sha": "a" * 40},
        protocol_lock=protocol_lock,
        verification_features=_features(),
    )
    assert set(bundle.file_hashes) == {
        "model.joblib",
        "feature_schema.json",
        "categorical_features.json",
        "threshold.json",
        "metrics_development.json",
        "metadata.json",
        "protocol_lock.json",
    }
    assert bundle.byte_size == sum(path.stat().st_size for path in bundle.directory.iterdir())
    assert isinstance(load_verified_bundle(bundle), _SerializableModel)
    (bundle.directory / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(V1ExecutionError, match="hash mismatch"):
        load_verified_bundle(bundle)


def test_rolling_execution_runs_exactly_four_candidates_by_four_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = [pd.Timestamp(2025, month, day) for month in range(1, 11) for day in (1, 2)]
    train = _canonical_frame(dates)
    protocol = yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())
    built: list[str] = []

    def builder(_protocol: dict[str, object], candidate_id: str) -> _DistanceModel:
        built.append(candidate_id)
        return _DistanceModel()

    monkeypatch.setattr(v1_execution, "build_catboost_candidate", builder)
    tracker = NullTracker()
    results = run_catboost_rolling(
        protocol=protocol,
        train=train,
        tracker=tracker,
        common={"group": "test", "protocol_id": "test"},
    )
    assert len(results) == 4 and len(tracker.runs) == 4
    assert built == [candidate for candidate in ("CB1", "CB2", "CB3", "CB4") for _ in range(4)]
    assert all(len(result["folds"]) == 4 for result in results)
    assert all(
        set(fold)
        == {
            "fold_id",
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "prevalence",
            "probability_mean",
        }
        for result in results
        for fold in result["folds"]
    )


def test_finalist_is_evaluated_bundled_and_gated_once(tmp_path: Path) -> None:
    protocol = yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())
    lock = json.loads((ROOT / "experiments/v1/protocol_lock.json").read_text())
    selection = v1_execution.adapt_v1_frame(
        _canonical_frame(list(pd.date_range("2025-11-16", periods=4)))
    )
    finalist = _finalist_evidence(
        model=_DistanceModel(),
        finalist_id="CB1-none",
        spec=SimpleNamespace(candidate_id="CB1", parameters={"iterations": 300}),
        method="none",
        selection=selection,
        protocol=protocol,
        protocol_lock=lock,
        common={"protocol_id": protocol["protocol_id"]},
        bundle_directory=tmp_path / "candidate",
    )
    assert finalist["status"] == "completed"
    assert finalist["threshold"] == 0.8
    assert finalist["bundle"].byte_size > 0
    assert len(finalist["gate_evidence"]) == 23

    class ConstantModel(_DistanceModel):
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            scores = np.full(len(features), 0.5)
            return np.column_stack((1 - scores, scores))

    stopped = _finalist_evidence(
        model=ConstantModel(),
        finalist_id="CB1-sigmoid",
        spec=SimpleNamespace(candidate_id="CB1", parameters={"iterations": 300}),
        method="sigmoid",
        selection=selection,
        protocol=protocol,
        protocol_lock=lock,
        common={"protocol_id": protocol["protocol_id"]},
        bundle_directory=tmp_path / "not-created",
    )
    assert stopped["status"] == "no_eligible_threshold"
    assert stopped["gate_evidence"] == ()


def test_bundle_refuses_overwrite(tmp_path: Path) -> None:
    directory = tmp_path / "candidate"
    directory.mkdir()
    with pytest.raises(V1ExecutionError, match="already exists"):
        write_candidate_bundle(
            directory=directory,
            model=_SerializableModel(),
            candidate_id="CB1-none",
            parameters={},
            calibration_method="none",
            threshold=0.5,
            metrics={},
            metadata={},
            protocol_lock={},
            verification_features=_features(),
        )


def test_r3_reconstruction_failure_blocks_challenger_trust() -> None:
    with pytest.raises(V1ExecutionError, match="blocked all CatBoost challengers"):
        require_r3_reconstruction({"reproduction": {"all_metrics_reproduced": False}})
    result = {"reproduction": {"all_metrics_reproduced": True}}
    assert require_r3_reconstruction(result) is result


def test_sanitized_finalist_removes_models_and_serializes_gate_evidence() -> None:
    sanitized = _sanitized_finalist(
        {
            "finalist_id": "CB1-none",
            "model": object(),
            "bundle": CandidateBundle(Path("x"), {}, "digest", 1, "sha"),
            "gate_evidence": (GateEvidence("gate", "is true", True, True),),
        }
    )
    assert "model" not in sanitized and "bundle" not in sanitized
    assert sanitized["gate_evidence"] == [
        {"gate_name": "gate", "requirement": "is true", "observed": True, "passed": True}
    ]


def test_production_v0_guard_rejects_release_and_deployment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = yaml.safe_load((ROOT / "configs/v1_experiment_protocol.yaml").read_text())
    release = json.loads((ROOT / "release/release_decision.json").read_text())
    deployment = json.loads((ROOT / "deploy/deployment_manifest.json").read_text())
    changed_release = dict(release)
    changed_release["registry_version"] = "v1"

    def release_drift(path: Path) -> dict[str, object]:
        return changed_release if path.name == "release_decision.json" else deployment

    monkeypatch.setattr(v1_execution, "_read_json_object", release_drift)
    with pytest.raises(V1ExecutionError, match="release no longer preserves"):
        validate_production_v0(ROOT, protocol)

    def deployment_drift(path: Path) -> dict[str, object]:
        if path.name == "release_decision.json":
            return release
        changed = json.loads(json.dumps(deployment))
        changed["model"]["registry_version"] = "v1"
        return changed

    monkeypatch.setattr(v1_execution, "_read_json_object", deployment_drift)
    with pytest.raises(V1ExecutionError, match="deployment manifest"):
        validate_production_v0(ROOT, protocol)


def test_null_tracker_records_metadata_without_external_side_effects() -> None:
    tracker = NullTracker()
    with tracker.start_run(name="test", group="group", metadata={"stage": "rolling"}) as run:
        run.log({"metric": 1.0})
    assert tracker.runs[0].metadata == {"stage": "rolling", "group": "group"}
    assert tracker.runs[0].logged == [{"metric": 1.0}]


def test_real_tracker_is_lazy_and_contains_no_registry_methods() -> None:
    before = "wandb" in sys.modules
    tracker = WandbTracker(entity="entity", project="project")
    assert ("wandb" in sys.modules) is before
    assert not hasattr(tracker, "log_artifact")
    assert not hasattr(tracker, "link_artifact")


def test_applied_tracking_requires_online_mode_and_complete_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(V1ExecutionError, match="requires --tracking online"):
        _tracker_from_environment("disabled")
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(V1ExecutionError, match="environment is incomplete"):
        _tracker_from_environment("online")
    monkeypatch.setenv("WANDB_API_KEY", "secret-not-printed")
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_PROJECT", "project")
    tracker = _tracker_from_environment("online")
    assert isinstance(tracker, WandbTracker)


def _write_december_state(root: Path, *, december_evaluated: bool = False) -> None:
    development = root / "artifacts/v1/development"
    development.mkdir(parents=True)
    (development / "execution_marker.json").write_text(
        json.dumps({"status": "complete", "decision": "winner"}), encoding="utf-8"
    )
    model_path = development / "november_winner.joblib"
    joblib.dump(_SerializableModel(), model_path)
    (development / "november_winner_lock.json").write_text(
        json.dumps(
            {
                "december_evaluated": december_evaluated,
                "model_file_sha256": _sha(model_path),
                "feature_schema": list(V1_FEATURES),
                "categorical_schema": ["Reporting_Airline", "Origin", "Dest", "route"],
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )


def test_december_refuses_without_winner_after_stop_or_prior_evaluation(tmp_path: Path) -> None:
    with pytest.raises(V1ExecutionError, match="cannot read governed state"):
        validate_december_handoff(tmp_path)
    marker = tmp_path / "artifacts/v1/development/execution_marker.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"status": "complete", "decision": "governed_stop"}))
    with pytest.raises(V1ExecutionError, match="completed November winner"):
        validate_december_handoff(tmp_path)
    other = tmp_path / "prior"
    _write_december_state(other, december_evaluated=True)
    with pytest.raises(V1ExecutionError, match="not eligible"):
        validate_december_handoff(other)


def test_december_handoff_rejects_corruption_and_accepts_exact_lock(tmp_path: Path) -> None:
    _write_december_state(tmp_path)
    marker, lock = validate_december_handoff(tmp_path)
    assert marker["decision"] == "winner" and lock["threshold"] == 0.5
    model_path = tmp_path / "artifacts/v1/development/november_winner.joblib"
    model_path.write_bytes(b"changed")
    with pytest.raises(V1ExecutionError, match="hash mismatch"):
        validate_december_handoff(tmp_path)


def test_no_registry_or_model_artifact_mutation_exists_in_v1_execution_path() -> None:
    source = (ROOT / "src/flight_delay/modeling/v1_execution.py").read_text()
    for prohibited in ("wandb.Artifact", "link_artifact", "use_artifact", "production:v1"):
        assert prohibited not in source
    assert "test.parquet" not in source
