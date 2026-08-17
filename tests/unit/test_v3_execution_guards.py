"""Durable-execution guards: merge state, December handoff, and frozen winner lineage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.manifest import write_manifest
from flight_delay.modeling.v3.data import V3DataGuardError, load_december_features
from flight_delay.modeling.v3.execution import (
    DECISION_PATH,
    DEVELOPMENT_MARKER,
    STATE_PATH,
    WINNER_LOCK,
    WINNER_MODEL,
    V3ExecutionError,
    _common_metadata,
    _fixed_threshold_metrics,
    _winner_lock_payload,
    implementation_git_sha,
    require_merged_applied_state,
    validate_december_handoff,
)
from flight_delay.modeling.v3.features import build_v3_historical_state
from flight_delay.modeling.v3.protocol import V3_FEATURES
from tests.conftest import make_v3_frame

ROOT = Path(__file__).resolve().parents[2]


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "feature/work")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "README.md").write_text("v3 guard fixture\n", encoding="utf-8")
    git(root, "add", "README.md")
    git(root, "commit", "-q", "-m", "initial")
    return root


def test_applied_execution_is_refused_off_main(temp_repo: Path) -> None:
    with pytest.raises(V3ExecutionError, match="reviewed PR is on main"):
        require_merged_applied_state(temp_repo)


def test_applied_execution_is_refused_on_a_dirty_worktree(temp_repo: Path) -> None:
    (temp_repo / "README.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(V3ExecutionError, match="clean Git worktree"):
        require_merged_applied_state(temp_repo)


def test_applied_execution_requires_the_frozen_protocol_commit(temp_repo: Path) -> None:
    git(temp_repo, "checkout", "-q", "-b", "main")
    with pytest.raises(V3ExecutionError, match="frozen v3 protocol commit"):
        require_merged_applied_state(temp_repo)


def test_implementation_sha_is_the_head_commit(temp_repo: Path) -> None:
    assert len(implementation_git_sha(temp_repo)) == 40


def _state():
    history = make_v3_frame(start="2024-01-01", end="2025-10-31")
    return build_v3_historical_state(history, as_of="2025-10-31")


def _seed_winner(root: Path, state, *, decision: str = "winner", **overrides) -> dict:
    (root / DEVELOPMENT_MARKER).parent.mkdir(parents=True, exist_ok=True)
    (root / DEVELOPMENT_MARKER).write_text(
        json.dumps({"status": "complete", "decision": decision}), encoding="utf-8"
    )
    (root / STATE_PATH).write_bytes(state.to_bytes())
    (root / WINNER_MODEL).write_bytes(b"frozen-model-bytes")
    import hashlib

    payload = {
        "finalist_id": "CB04-EXP120-sigmoid",
        "december_evaluated": False,
        "historical_state_as_of": "2025-10-31",
        "historical_state_sha256": state.sha256,
        "model_sha256": hashlib.sha256(b"frozen-model-bytes").hexdigest(),
        **overrides,
    }
    (root / WINNER_LOCK).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_december_handoff_accepts_a_complete_frozen_winner(tmp_path: Path) -> None:
    state = _state()
    _seed_winner(tmp_path, state)
    winner, restored = validate_december_handoff(tmp_path)
    assert winner["finalist_id"] == "CB04-EXP120-sigmoid"
    assert restored.sha256 == state.sha256


def test_december_handoff_requires_a_winner_decision(tmp_path: Path) -> None:
    _seed_winner(tmp_path, _state(), decision="governed_stop")
    with pytest.raises(V3ExecutionError, match="frozen November winner"):
        validate_december_handoff(tmp_path)


def test_december_cannot_be_evaluated_twice(tmp_path: Path) -> None:
    _seed_winner(tmp_path, _state(), december_evaluated=True)
    with pytest.raises(V3ExecutionError, match="already been evaluated"):
        validate_december_handoff(tmp_path)


def test_december_handoff_rejects_a_shifted_state_cutoff(tmp_path: Path) -> None:
    _seed_winner(tmp_path, _state(), historical_state_as_of="2025-11-30")
    with pytest.raises(V3ExecutionError, match="October-31 state"):
        validate_december_handoff(tmp_path)


def test_december_handoff_rejects_a_tampered_model(tmp_path: Path) -> None:
    state = _state()
    _seed_winner(tmp_path, state)
    (tmp_path / WINNER_MODEL).write_bytes(b"swapped-model")
    with pytest.raises(V3ExecutionError, match="model hash mismatch"):
        validate_december_handoff(tmp_path)


def test_december_handoff_rejects_a_tampered_state(tmp_path: Path) -> None:
    state = _state()
    _seed_winner(tmp_path, state, historical_state_sha256="0" * 64)
    with pytest.raises(V3ExecutionError, match="historical-state hash mismatch"):
        validate_december_handoff(tmp_path)


def test_winner_lock_records_ensemble_weights(v3_protocol: dict) -> None:
    state = _state()
    winner = {
        "finalist_id": "ENS25-sigmoid",
        "kind": "ensemble",
        "family": "ensemble",
        "calibration_method": "sigmoid",
        "ensemble_id": "ENS25",
        "lightgbm_weight": 0.25,
        "catboost_weight": 0.75,
        "lightgbm_base_candidate_id": "LGBM12-EXP120",
        "catboost_base_candidate_id": "CB04-EXP120",
        "metrics": {"threshold": 0.42},
        "bundle": {"serialized_bundle_bytes": 4096},
        "gate_evidence": [],
    }
    payload = _winner_lock_payload(
        protocol=v3_protocol,
        protocol_sha="a" * 64,
        code_sha="b" * 40,
        winner=winner,
        state=state,
        model_sha="c" * 64,
    )
    assert payload["lightgbm_weight"] == 0.25
    assert payload["catboost_weight"] == 0.75
    assert payload["feature_schema"] == list(V3_FEATURES)
    assert payload["december_evaluated"] is False
    assert payload["production_remains"] == "v0"


def test_winner_lock_records_a_base_identity(v3_protocol: dict) -> None:
    winner = {
        "finalist_id": "CB04-EXP120-none",
        "kind": "base",
        "family": "catboost",
        "calibration_method": "none",
        "base_candidate_id": "CB04-EXP120",
        "candidate_identity": {"depth": 6, "weight_policy": "EXPONENTIAL_120D"},
        "metrics": {"threshold": 0.3},
        "bundle": {"serialized_bundle_bytes": 2048},
        "gate_evidence": [],
    }
    payload = _winner_lock_payload(
        protocol=v3_protocol,
        protocol_sha="a" * 64,
        code_sha="b" * 40,
        winner=winner,
        state=_state(),
        model_sha="c" * 64,
    )
    assert payload["base_candidate_id"] == "CB04-EXP120"
    assert payload["candidate_identity"]["weight_policy"] == "EXPONENTIAL_120D"
    assert "lightgbm_weight" not in payload


def test_fixed_threshold_metrics_apply_the_frozen_threshold() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.7, 0.2, 0.6, 0.3])
    metrics = _fixed_threshold_metrics(labels, scores, 0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["predicted_positive_rate"] == pytest.approx(0.5)


def test_common_metadata_carries_the_governed_lineage() -> None:
    metadata = _common_metadata(
        protocol_sha="a" * 64,
        code_sha="b" * 40,
        lineage={"november_state_sha256": "c" * 64},
    )
    assert metadata["group"] == f"v3-{'a' * 64}-{'b' * 40}"
    assert metadata["cpu_confirmation_backend"] == "CPU"
    assert metadata["screening_backends"] == {"lightgbm": "CPU", "catboost": "GPU:0"}
    assert metadata["feature_state_digest"] == "c" * 64


def test_december_features_require_the_october_31_state(tmp_path: Path) -> None:
    history = make_v3_frame(start="2024-01-01", end="2025-09-30")
    early = build_v3_historical_state(history, as_of="2025-09-30")
    with pytest.raises(V3DataGuardError, match="frozen October-31 feature state"):
        load_december_features(tmp_path, state=early)


def test_december_features_require_an_authorized_decode(tmp_path: Path) -> None:
    from flight_delay.data.prepare_v3 import V3_PROCESSED_MANIFEST

    write_manifest(
        tmp_path / V3_PROCESSED_MANIFEST,
        {
            "schema_version": 1,
            "december_2025_decoded": False,
            "january_may_2026_decoded": False,
            "parquet_files": {},
        },
    )
    with pytest.raises(V3DataGuardError, match="authorized December decode"):
        load_december_features(tmp_path, state=_state())


def test_decision_and_state_paths_are_v3_scoped() -> None:
    for path in (DECISION_PATH, STATE_PATH, WINNER_LOCK, WINNER_MODEL):
        assert str(path).startswith("artifacts/v3/")


def test_december_features_read_only_december(tmp_path: Path) -> None:
    """A full authorized read still refuses any row outside December 2025."""

    from flight_delay.data.prepare import OUTPUT_COLUMNS
    from flight_delay.data.prepare_v3 import V3_PROCESSED_DIRECTORY, V3_PROCESSED_MANIFEST

    state = _state()
    december = make_v3_frame(start="2025-12-01", end="2025-12-31")
    processed = tmp_path / V3_PROCESSED_DIRECTORY
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "v3_december.parquet"
    december.loc[:, OUTPUT_COLUMNS].to_parquet(path, engine="pyarrow", index=False)
    write_manifest(
        tmp_path / V3_PROCESSED_MANIFEST,
        {
            "schema_version": 1,
            "december_2025_decoded": True,
            "january_may_2026_decoded": False,
            "parquet_files": {
                "v3_december": {
                    "filename": path.name,
                    "byte_size": path.stat().st_size,
                    "row_count": len(december),
                    "sha256": "0" * 64,
                }
            },
        },
    )
    features, target, dates = load_december_features(
        tmp_path, state=state, verify_source_hash=False
    )
    assert tuple(features.columns) == V3_FEATURES
    assert len(features) == len(december)
    assert pd.to_datetime(dates).min() >= pd.Timestamp("2025-12-01")
    assert pd.to_datetime(dates).max() < pd.Timestamp("2026-01-01")
    assert set(target.unique()) <= {0, 1}


def test_winner_model_is_written_once_and_never_overwritten(tmp_path: Path) -> None:
    from flight_delay.modeling.v3.execution import _write_winner_model

    path = tmp_path / WINNER_MODEL
    _write_winner_model(path, {"frozen": "winner"})
    assert path.is_file()
    with pytest.raises(V3ExecutionError, match="already exists"):
        _write_winner_model(path, {"frozen": "other"})
    # No temporary file is left behind next to the immutable artifact.
    assert [child.name for child in path.parent.iterdir()] == [path.name]


def test_real_model_construction_uses_the_lazy_runtime_import(v3_protocol: dict) -> None:
    """Building with the real runtime must accept every frozen identity unchanged."""

    from flight_delay.modeling.v3.models import V3ModelError, build_candidate, candidate_specs

    spec = candidate_specs(v3_protocol, family="lightgbm", backend="CPU")[0]
    broken = type(spec)(
        family=spec.family,
        candidate_id=spec.candidate_id,
        base_configuration=spec.base_configuration,
        weight_policy=spec.weight_policy,
        identity_parameters=spec.identity_parameters,
        constructor_parameters={"not_a_real_parameter": object()},
        backend=spec.backend,
    )

    class Strict:
        def __init__(self, **_: object) -> None:
            raise TypeError("unexpected parameter")

    with pytest.raises(V3ModelError, match="rejected frozen parameters"):
        build_candidate(broken, classifier_type=Strict)
