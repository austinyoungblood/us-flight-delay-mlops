from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from flight_delay.features.leakage import FeatureLeakageError, validate_model_features
from flight_delay.modeling.v2.protocol import (
    CATBOOST_DOMAINS,
    HISTORICAL_FEATURES,
    LIGHTGBM_DOMAINS,
    PROTOCOL_SHA256,
    V2_FEATURES,
    V2ProtocolError,
    _validate_dependency_tree,
    canonical_sha256,
    load_and_validate_v2_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_protocol_validates_exact_contract() -> None:
    protocol, lock, digest = load_and_validate_v2_protocol(
        ROOT / "configs/v2_experiment_protocol.yaml",
        lock_path=ROOT / "experiments/v2/protocol_lock.json",
        repository_root=ROOT,
    )
    assert digest == PROTOCOL_SHA256
    assert protocol["feature_contract"]["base_features"] == list(V2_FEATURES[:20])
    assert protocol["feature_contract"]["historical_features"] == list(HISTORICAL_FEATURES)
    assert len(protocol["lightgbm_search"]["candidates"]) == 16
    assert len(protocol["catboost_search"]["candidates"]) == 12
    assert lock["lightgbm_candidates"] == protocol["lightgbm_search"]["candidates"]
    assert lock["catboost_candidates"] == protocol["catboost_search"]["candidates"]
    assert canonical_sha256(lock["lightgbm_candidates"]) == lock["lightgbm_candidates_sha256"]
    assert canonical_sha256(lock["catboost_candidates"]) == lock["catboost_candidates_sha256"]


@pytest.mark.parametrize(
    ("family", "domains", "prefix"),
    [("lightgbm", LIGHTGBM_DOMAINS, "LGBM"), ("catboost", CATBOOST_DOMAINS, "CB")],
)
def test_candidate_matrices_are_unique_in_domain_and_backend_free(
    v2_protocol: dict[str, object], family: str, domains: dict[str, tuple[object, ...]], prefix: str
) -> None:
    rows = v2_protocol[f"{family}_search"]["candidates"]  # type: ignore[index]
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(str(row["id"]).startswith(prefix) for row in rows)
    for row in rows:
        assert set(row) == {"id", *domains}
        assert all(row[name] in values for name, values in domains.items())
        assert not {"task_type", "devices", "n_jobs"} & set(row)


def test_protocol_and_lock_byte_drift_fail_closed(tmp_path: Path) -> None:
    protocol = ROOT / "configs/v2_experiment_protocol.yaml"
    lock = ROOT / "experiments/v2/protocol_lock.json"
    drifted_protocol = tmp_path / "protocol.yaml"
    drifted_protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(V2ProtocolError, match="canonical protocol"):
        load_and_validate_v2_protocol(drifted_protocol, lock_path=lock, repository_root=ROOT)

    drifted_lock = tmp_path / "lock.json"
    payload = json.loads(lock.read_text())
    payload["training_started"] = True
    drifted_lock.write_text(json.dumps(payload))
    with pytest.raises(V2ProtocolError, match="protocol-lock SHA256"):
        load_and_validate_v2_protocol(protocol, lock_path=drifted_lock, repository_root=ROOT)


def test_dependency_hash_and_manifest_digest_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"manifest_digest": "manifest-ok"}))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _validate_dependency_tree(
        {
            "artifact": {
                "path": "artifact.json",
                "sha256": digest,
                "manifest_digest": "manifest-ok",
            }
        },
        tmp_path,
    )
    with pytest.raises(V2ProtocolError, match="artifact hash mismatch"):
        _validate_dependency_tree(
            {"artifact": {"path": "artifact.json", "sha256": "0" * 64}}, tmp_path
        )
    with pytest.raises(V2ProtocolError, match="drifted"):
        _validate_dependency_tree(
            {
                "artifact": {
                    "path": "artifact.json",
                    "sha256": digest,
                    "manifest_digest": "wrong",
                }
            },
            tmp_path,
        )


def test_central_leakage_guard_accepts_only_the_frozen_historical_contract() -> None:
    assert validate_model_features(V2_FEATURES) == frozenset(V2_FEATURES)
    with pytest.raises(FeatureLeakageError):
        validate_model_features((*V2_FEATURES, "DepDelay"))
    with pytest.raises(FeatureLeakageError):
        validate_model_features((*V2_FEATURES, "current_month_delay_rate"))


def test_protocol_yaml_declares_no_training_result_or_december_access() -> None:
    protocol = yaml.safe_load((ROOT / "configs/v2_experiment_protocol.yaml").read_text())
    assert protocol["state"] == {
        "phase": "pretraining_protocol_locked",
        "protocol_precedes_training": True,
        "training_started": False,
        "results_exist": False,
        "wandb_runs_created": False,
        "december_opened": False,
        "historical_test_accessed": False,
    }
    assert protocol["release"]["production_version_after_v2_development"] == "v0"
    assert protocol["december_qualification"]["development_may_open"] is False


def test_v2_protocol_documentation_is_public_and_complete() -> None:
    text = (ROOT / "docs/v2-model-experiment-protocol.md").read_text()
    for phrase in (
        "Seventeen numeric features",
        "October-31 state",
        "Exact LightGBM matrix",
        "Exact CatBoost matrix",
        "GPU screening",
        "authoritative CPU confirmation",
        "production `v0`",
    ):
        assert phrase in text
    assert "coding agent" not in text.casefold()
    assert "brief 0" not in text.casefold()
