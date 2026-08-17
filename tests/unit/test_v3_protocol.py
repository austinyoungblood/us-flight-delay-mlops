"""Synthetic validation of the frozen v3 protocol and its lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from flight_delay.features.leakage import FORBIDDEN_FEATURES, validate_model_features
from flight_delay.modeling.v2.protocol import HISTORICAL_FEATURES as V2_HISTORICAL_FEATURES
from flight_delay.modeling.v3.protocol import (
    BASE_CONFIGURATION_IDS,
    CANDIDATE_IDENTITY_IDS,
    CATEGORICAL_FEATURES,
    DETERMINISTIC_SEASONAL_FEATURES,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    SEASONAL_HISTORICAL_FEATURES,
    V3_FEATURES,
    V3ProtocolError,
    canonical_sha256,
    load_and_validate_v3_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/v3_experiment_protocol.yaml"
LOCK_PATH = ROOT / "experiments/v3/protocol_lock.json"


def load() -> tuple[dict, dict, str]:
    return load_and_validate_v3_protocol(PROTOCOL_PATH, lock_path=LOCK_PATH, repository_root=ROOT)


def test_frozen_protocol_validates_and_matches_its_digest() -> None:
    protocol, lock, protocol_sha = load()
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol_sha == PROTOCOL_SHA256
    assert lock["protocol_sha256"] == protocol_sha
    assert lock["base_git_sha"] == protocol["dependencies"]["base_git_sha"]


def test_protocol_declares_no_training_and_sealed_future_periods() -> None:
    protocol, lock, _sha = load()
    for marker in (
        "training_started",
        "results_exist",
        "wandb_runs_created",
        "december_opened",
        "historical_test_accessed",
    ):
        assert protocol["state"][marker] is False
        assert lock[marker] is False
    assert lock["production_registry_version"] == "v0"
    assert protocol["release"]["production_version_after_v3_development"] == "v0"
    assert protocol["development_data"]["december_2025_decoded_during_development"] is False
    assert protocol["prohibited_periods"]["historical_test_2026"]["access_prohibited"] is True
    assert protocol["fresh_final"]["access_count_during_protocol_and_implementation"] == 0


def test_feature_contract_retains_all_thirty_seven_v2_features() -> None:
    protocol, _lock, _sha = load()
    contract = protocol["feature_contract"]
    assert contract["total_feature_count"] == 48
    assert len(V3_FEATURES) == 48
    assert len(set(V3_FEATURES)) == 48
    retained = set(contract["base_features"]) | set(V2_HISTORICAL_FEATURES)
    assert len(retained) == 37
    assert retained <= set(V3_FEATURES)
    assert set(DETERMINISTIC_SEASONAL_FEATURES) <= set(V3_FEATURES)
    assert set(SEASONAL_HISTORICAL_FEATURES) <= set(V3_FEATURES)
    assert validate_model_features(V3_FEATURES) == frozenset(V3_FEATURES)
    assert not set(V3_FEATURES) & FORBIDDEN_FEATURES


def test_expanded_native_categoricals_are_all_model_features() -> None:
    protocol, _lock, _sha = load()
    assert tuple(protocol["feature_contract"]["categorical_features"]) == CATEGORICAL_FEATURES
    assert len(CATEGORICAL_FEATURES) == 8
    assert set(CATEGORICAL_FEATURES) <= set(V3_FEATURES)


def test_exactly_eight_identities_cross_four_configurations_with_two_policies() -> None:
    protocol, lock, _sha = load()
    identities = protocol["candidate_identities"]["identities"]
    assert [row["id"] for row in identities] == list(CANDIDATE_IDENTITY_IDS)
    assert len(identities) == 8
    assert {row["base_configuration"] for row in identities} == set(BASE_CONFIGURATION_IDS)
    assert {row["weight_policy"] for row in identities} == {"UNIFORM", "EXPONENTIAL_120D"}
    assert lock["candidate_identities_sha256"] == canonical_sha256(identities)


def test_carried_forward_hyperparameters_match_the_frozen_v2_protocol() -> None:
    protocol, _lock, _sha = load()
    v2 = yaml.safe_load((ROOT / "configs/v2_experiment_protocol.yaml").read_bytes())
    v2_by_id = {
        row["id"]: row
        for row in (*v2["lightgbm_search"]["candidates"], *v2["catboost_search"]["candidates"])
    }
    for base in protocol["carried_forward_configurations"]["base_configurations"]:
        source = v2_by_id[base["id"]]
        carried = {name: value for name, value in base.items() if name != "family"}
        assert carried == source, f"{base['id']} drifted from its v2 hyperparameters"


def test_ranking_prioritizes_worst_fold_then_november() -> None:
    protocol, _lock, _sha = load()
    ranking = protocol["search_metric"]["candidate_ranking"]
    assert ranking[0] == "worst_fold_operating_precision_desc"
    assert ranking[1] == "fold_4_november_operating_precision_desc"
    assert ranking[-1] == "candidate_id_lexical_asc"
    assert len(ranking) == 9


def test_november_gates_are_not_relaxed_from_v2() -> None:
    protocol, _lock, _sha = load()
    v2 = yaml.safe_load((ROOT / "configs/v2_experiment_protocol.yaml").read_bytes())
    v3_gates = protocol["november_selection"]["acceptance_gates"]
    v2_gates = v2["november_selection"]["acceptance_gates"]
    assert v3_gates["operating_point"] == v2_gates["operating_point"]
    assert v3_gates["probability"] == v2_gates["probability"]
    assert v3_gates["discrimination"] == v2_gates["discrimination"]
    assert v3_gates["operational"] == v2_gates["operational"]
    assert set(v2_gates["governance"]) <= set(v3_gates["governance"])


def test_advancement_is_eight_then_four_then_two() -> None:
    protocol, _lock, _sha = load()
    advancement = protocol["advancement"]
    assert advancement["screening"]["total_identities"] == 8
    assert advancement["screening"]["top_per_family_to_cpu_confirmation"] == 2
    assert advancement["cpu_confirmation"]["total_identities"] == 4
    assert advancement["cpu_confirmation"]["top_per_family_to_full_refit"] == 1
    assert advancement["full_refit"]["total_bases"] == 2
    assert protocol["finalists"]["total"] == 15


def test_folds_are_contiguous_and_end_on_november() -> None:
    protocol, _lock, _sha = load()
    folds = protocol["rolling_origin"]["folds"]
    assert len(folds) == 4
    for fold in folds:
        assert fold["fit_start"] == "2024-02-01"
        assert fold["fit_end_exclusive"] == fold["evaluation_start"]
    assert folds[-1]["evaluation_start"] == "2025-11-01"
    assert folds[-1]["evaluation_end_exclusive"] == "2025-12-01"


def test_v3_source_manifest_excludes_every_2026_archive() -> None:
    manifest = json.loads((ROOT / "data/manifests/v3_source_manifest.json").read_text())
    years = {int(record["year"]) for record in manifest["files"]}
    assert years == {2024, 2025}
    assert len(manifest["files"]) == 24


def test_v0_lineage_manifests_are_byte_identical_to_the_v2_declaration() -> None:
    protocol, _lock, _sha = load()
    v2 = yaml.safe_load((ROOT / "configs/v2_experiment_protocol.yaml").read_bytes())
    v3_source = protocol["dependencies"]["immutable_v0_v1_v2_source_manifest"]
    v3_processed = protocol["dependencies"]["immutable_v0_v1_v2_processed_manifest"]
    assert v3_source["sha256"] == v2["dependencies"]["source_dataset_manifest"]["sha256"]
    assert v3_processed["sha256"] == v2["dependencies"]["processed_dataset_manifest"]["sha256"]


def test_reused_2025_archives_are_byte_identical_between_v0_and_v3() -> None:
    v0 = json.loads((ROOT / "data/manifests/source_manifest.json").read_text())
    v3 = json.loads((ROOT / "data/manifests/v3_source_manifest.json").read_text())
    v0_2025 = {(r["year"], r["month"]): r for r in v0["files"] if r["year"] == 2025}
    v3_2025 = {(r["year"], r["month"]): r for r in v3["files"] if r["year"] == 2025}
    assert v0_2025 == v3_2025
    assert len(v3_2025) == 12


@pytest.mark.parametrize(
    "mutation",
    [
        {"protocol_id": "tampered"},
        {"schema_version": 2},
    ],
)
def test_tampered_protocol_is_rejected(tmp_path: Path, mutation: dict) -> None:
    payload = yaml.safe_load(PROTOCOL_PATH.read_bytes())
    payload.update(mutation)
    tampered = tmp_path / "protocol.yaml"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(V3ProtocolError):
        load_and_validate_v3_protocol(tampered, lock_path=LOCK_PATH, repository_root=ROOT)


def test_tampered_lock_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(LOCK_PATH.read_text())
    payload["training_started"] = True
    tampered = tmp_path / "lock.json"
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(V3ProtocolError):
        load_and_validate_v3_protocol(PROTOCOL_PATH, lock_path=tampered, repository_root=ROOT)


def test_missing_protocol_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(V3ProtocolError):
        load_and_validate_v3_protocol(
            tmp_path / "absent.yaml", lock_path=LOCK_PATH, repository_root=ROOT
        )
