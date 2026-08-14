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
ORIGINAL_LIGHTGBM_CANDIDATES_SHA256 = (
    "71849b773eabc506b6eda2763bbd1be693a9167aee4a73e1433e43d5749de0d0"
)
ORIGINAL_CATBOOST_SEARCH_SHA256 = "1715e6385513f736c87887c7562ae3de5c7adaf93e41e9f6fb526f2db6533886"
IMMUTABLE_V1_SHA256 = {
    "configs/v1_experiment_protocol.yaml": (
        "a6b1de9de550d1bd94eae0e56f8d88d65801ec488b6c539fc64afbafa4ccfffb"
    ),
    "experiments/v1/protocol_lock.json": (
        "e5373df1c8d11e75132769283f4c784420637f73f9cb11a87d61895462d29b38"
    ),
    "experiments/v1/development_result.json": (
        "de8dd8221e72187dad82bcd1f81f633eae7d908d9a09e82da4480c184536ce5c"
    ),
    "docs/v1-model-experiment-result.md": (
        "3b6f24e0613cc6693e344b0e455a30406182c1372876a43954f018622f9657a2"
    ),
}


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


def test_pretraining_subsampling_correction_preserves_candidate_identity_and_v1() -> None:
    protocol = yaml.safe_load((ROOT / "configs/v2_experiment_protocol.yaml").read_text())
    lock = json.loads((ROOT / "experiments/v2/protocol_lock.json").read_text())
    lightgbm = protocol["lightgbm_search"]

    assert lightgbm["common_parameters"]["subsample_freq"] == 1
    assert canonical_sha256(lightgbm["candidates"]) == ORIGINAL_LIGHTGBM_CANDIDATES_SHA256
    assert [row["id"] for row in lightgbm["candidates"]] == [
        f"LGBM{index:02d}" for index in range(1, 17)
    ]
    assert canonical_sha256(protocol["catboost_search"]) == ORIGINAL_CATBOOST_SEARCH_SHA256
    assert lock["immutable_v1_sha256"] == IMMUTABLE_V1_SHA256
    for path, expected in IMMUTABLE_V1_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected

    correction = protocol["pretraining_correction"]
    assert correction["applied_before_any_real_fit"] is True
    assert correction["observed_model_results_available"] is False
    assert correction["observed_model_results_influenced_correction"] is False
    assert correction["candidate_identity_rows_changed"] is False
    assert correction["catboost_protocol_changed"] is False


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
        "subsample_freq=1",
        "before any real v2 fit",
        "No observed model result influenced this correction",
        "production `v0`",
    ):
        assert phrase in text
    assert "coding agent" not in text.casefold()
    assert "brief 0" not in text.casefold()
