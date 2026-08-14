from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from flight_delay.modeling.v1_protocol import (
    CANONICAL_PROTOCOL_SHA256,
    PROTOCOL_ID,
    V1ProtocolError,
    load_and_validate_v1_protocol,
    validate_v1_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/v1_experiment_protocol.yaml"
LOCK_PATH = ROOT / "experiments/v1/protocol_lock.json"


def canonical_protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def assert_invalid(protocol: dict[str, Any]) -> None:
    with pytest.raises(V1ProtocolError):
        validate_v1_protocol(protocol, repository_root=ROOT)


def test_canonical_protocol_and_lock_are_valid() -> None:
    protocol, lock, protocol_sha256 = load_and_validate_v1_protocol(
        PROTOCOL_PATH, lock_path=LOCK_PATH, repository_root=ROOT
    )
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol_sha256 == CANONICAL_PROTOCOL_SHA256
    assert lock["training_started"] is False
    assert lock["wandb_runs_created"] is False
    assert lock["fresh_final_accessed"] is False


def test_altered_catboost_hyperparameter_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["catboost_search"]["candidates"][0]["depth"] = 7
    assert_invalid(protocol)


def test_fifth_candidate_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["catboost_search"]["candidates"].append(
        {"id": "CB5", "depth": 10, "iterations": 600, "learning_rate": 0.02, "l2_leaf_reg": 9}
    )
    protocol["catboost_search"]["candidate_count"] = 5
    assert_invalid(protocol)


def test_missing_candidate_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["catboost_search"]["candidates"].pop()
    protocol["catboost_search"]["candidate_count"] = 3
    assert_invalid(protocol)


def test_altered_catboost_version_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["model_family"]["primary"]["version"] = "1.2.11"
    assert_invalid(protocol)


def test_unsafe_feature_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["feature_contract"]["model_features"].append("ArrDelay")
    assert_invalid(protocol)


@pytest.mark.parametrize("field", ["flight_date", "target"])
def test_ordering_and_label_fields_cannot_be_model_features(field: str) -> None:
    protocol = canonical_protocol()
    protocol["feature_contract"]["model_features"].append(field)
    assert_invalid(protocol)


def test_invalid_categorical_feature_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["feature_contract"]["categorical_features"].append("Month")
    assert_invalid(protocol)


def test_fold_overlap_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["rolling_origin"]["folds"][0]["validation_start"] = "2025-06-30"
    assert_invalid(protocol)


def test_fold_extending_into_2026_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["rolling_origin"]["folds"][3]["validation_end_exclusive"] = "2026-01-02"
    assert_invalid(protocol)


def test_calibration_overlap_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["refit_calibration"]["calibration_period"]["start"] = "2025-10-31"
    assert_invalid(protocol)


def test_selection_overlap_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["november_selection"]["period"]["start"] = "2025-11-15"
    assert_invalid(protocol)


def test_december_misuse_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["december_qualification"]["period"]["start"] = "2025-11-30"
    assert_invalid(protocol)


def test_historical_test_reuse_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["historical_test"]["access_prohibited"] = False
    assert_invalid(protocol)


def test_changed_threshold_constraints_are_rejected() -> None:
    protocol = canonical_protocol()
    constraints = protocol["november_selection"]["threshold_objective"]["constraints"]
    constraints["precision_min"] = 0.29
    assert_invalid(protocol)


def test_changed_november_acceptance_gate_is_rejected() -> None:
    protocol = canonical_protocol()
    gates = protocol["november_selection"]["acceptance_gates"]["operating_point"]
    gates["f1_min"] = 0.37
    assert_invalid(protocol)


def test_invalid_fresh_holdout_rule_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["fresh_final"]["selection_rule"]["flight_date_strictly_after"] = "2026-04-30"
    assert_invalid(protocol)


def test_bootstrap_configuration_drift_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["uncertainty"]["replicates"] = 499
    assert_invalid(protocol)


def test_incumbent_identity_drift_is_rejected() -> None:
    protocol = canonical_protocol()
    protocol["incumbent"]["registry_version"] = "v1"
    assert_invalid(protocol)


def test_protocol_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["protocol_sha256"] = "0" * 64
    lock_path = tmp_path / "protocol_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(V1ProtocolError, match="protocol lock"):
        load_and_validate_v1_protocol(PROTOCOL_PATH, lock_path=lock_path, repository_root=ROOT)


def test_protocol_validator_has_no_restricted_import_network_or_parquet_access(
    tmp_path: Path,
) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import builtins
import pathlib
import socket

forbidden_roots = {"catboost", "boto3", "wandb"}
original_import = builtins.__import__
original_open = builtins.open
original_path_open = pathlib.Path.open

def guarded_import(name, *args, **kwargs):
    if name.partition(".")[0] in forbidden_roots:
        raise AssertionError(f"forbidden import: {name}")
    return original_import(name, *args, **kwargs)

def guarded_open(file, *args, **kwargs):
    if str(file).endswith(".parquet"):
        raise AssertionError(f"forbidden parquet access: {file}")
    return original_open(file, *args, **kwargs)

def guarded_path_open(self, *args, **kwargs):
    if self.suffix == ".parquet":
        raise AssertionError(f"forbidden parquet access: {self}")
    return original_path_open(self, *args, **kwargs)

def blocked_network(*args, **kwargs):
    raise AssertionError("network access is forbidden")

builtins.__import__ = guarded_import
builtins.open = guarded_open
pathlib.Path.open = guarded_path_open
socket.socket.connect = blocked_network
socket.create_connection = blocked_network
socket.getaddrinfo = blocked_network
""".lstrip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_v1_protocol.py")],
        cwd=ROOT,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": f"{tmp_path}:{ROOT / 'src'}",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"


def test_mutated_protocol_bytes_fail_the_canonical_sha_gate(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        PROTOCOL_PATH.read_text(encoding="utf-8").replace(
            "candidate_count: 4", "candidate_count: 5"
        ),
        encoding="utf-8",
    )
    with pytest.raises(V1ProtocolError, match="protocol SHA256"):
        load_and_validate_v1_protocol(protocol_path, lock_path=LOCK_PATH, repository_root=ROOT)
