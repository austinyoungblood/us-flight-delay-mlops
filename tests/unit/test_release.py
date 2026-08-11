import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.release import (
    DATASET_ARTIFACT,
    DATASET_DIGEST,
    LOCKED_THRESHOLD,
    REFERENCE_DEVELOPMENT_METRICS,
    ReconstructionResult,
    ReleaseGuardError,
    aggregate_digest,
    compare_development_metrics,
    create_one_time_marker,
    final_test_metrics,
    load_release_policy,
    reconstruct_r3,
    validate_release_policy,
    verify_locked_files,
    write_release_bundle,
    write_selection_lock,
)


class _Model:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_Model":
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        column = "probability" if "probability" in features else "Distance"
        probabilities = np.asarray(features[column], dtype=float)
        return np.column_stack((1 - probabilities, probabilities))


def _policy() -> dict:
    return {
        "policy_version": "brief05-v1",
        "dataset": {"artifact": DATASET_ARTIFACT, "digest": DATASET_DIGEST},
        "candidate": {
            "configuration_id": "R3",
            "calibrator": "sigmoid",
            "re_rank_candidates": False,
            "retune_model": False,
            "retune_threshold": False,
        },
    }


def test_policy_locks_candidate_and_forbids_search() -> None:
    validate_release_policy(_policy())
    changed = _policy()
    changed["candidate"]["retune_threshold"] = True
    with pytest.raises(ReleaseGuardError, match="unauthorized search"):
        validate_release_policy(changed)
    assert LOCKED_THRESHOLD == 0.1840285229739868


def test_locked_hash_verification_detects_tampering(tmp_path: Path) -> None:
    item = tmp_path / "model_bundle" / "metadata.json"
    item.parent.mkdir()
    item.write_text("original", encoding="utf-8")
    digest = hashlib.sha256(item.read_bytes()).hexdigest()
    lock = {
        "file_hashes": {"model_bundle/metadata.json": digest},
        "aggregate_bundle_digest": aggregate_digest({"model_bundle/metadata.json": digest}),
    }
    verify_locked_files(tmp_path, lock)
    item.write_text("changed", encoding="utf-8")
    with pytest.raises(ReleaseGuardError, match="hash mismatch"):
        verify_locked_files(tmp_path, lock)


def test_one_time_marker_refuses_second_creation(tmp_path: Path) -> None:
    marker = tmp_path / "marker.json"
    create_one_time_marker(marker, {"status": "started"})
    assert json.loads(marker.read_text()) == {"status": "started"}
    with pytest.raises(ReleaseGuardError, match="already exists"):
        create_one_time_marker(marker, {"status": "started"})


def test_final_test_gate_uses_lift_and_proper_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flight_delay.modeling.release.measure_single_row_latency",
        lambda *_args, **_kwargs: {"sample_count": 4, "p95_ms": 1.0},
    )
    features = pd.DataFrame({"probability": [0.01, 0.01, 0.99, 0.99]})
    target = pd.Series([0, 0, 1, 1])
    metrics, gates = final_test_metrics(
        model=_Model(), features=features, target=target, threshold=0.5, bundle_size=1024
    )
    assert metrics["average_precision_lift_over_prevalence"] == pytest.approx(2.0)
    assert metrics["brier_skill_score"] > 0
    assert all(gates.values())


def test_bundle_size_is_a_strict_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flight_delay.modeling.release.measure_single_row_latency",
        lambda *_args, **_kwargs: {"sample_count": 4, "p95_ms": 1.0},
    )
    features = pd.DataFrame({"probability": [0.05, 0.15, 0.80, 0.90]})
    _, gates = final_test_metrics(
        model=_Model(),
        features=features,
        target=pd.Series([0, 0, 1, 1]),
        threshold=0.5,
        bundle_size=10 * 1024 * 1024,
    )
    assert gates["bundle_size"] is False


def test_policy_loader_and_reproduction_comparison(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    import yaml

    policy_path.write_text(yaml.safe_dump(_policy()), encoding="utf-8")
    assert load_release_policy(policy_path)["candidate"]["configuration_id"] == "R3"
    exact = compare_development_metrics(dict(REFERENCE_DEVELOPMENT_METRICS))
    assert exact["all_metrics_reproduced"] is True
    changed = dict(REFERENCE_DEVELOPMENT_METRICS)
    changed["roc_auc"] = float(changed["roc_auc"]) + 1e-6
    assert compare_development_metrics(changed)["all_metrics_reproduced"] is False


def test_writes_complete_hash_verifiable_release_bundle(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    bundle_root = release_root / "model_bundle"
    route_path = release_root / "route_stats.parquet"
    route_path.parent.mkdir(parents=True)
    route_path.write_bytes(b"display-only-route-evidence")
    route = {
        "display_only": True,
        "source_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(),
        "byte_size": route_path.stat().st_size,
    }
    policy_path = tmp_path / "release_policy.yaml"
    import yaml

    policy_path.write_text(yaml.safe_dump(_policy()), encoding="utf-8")
    final_fit = pd.DataFrame(
        {
            "target": [0, 1],
            "Distance": [100.0, 200.0],
            "CRSElapsedTime": [60.0, 90.0],
            "scheduled_departure_hour": [8, 9],
            "Reporting_Airline": ["AA", "UA"],
            "Origin": ["DEN", "SFO"],
            "Dest": ["SFO", "DEN"],
            "Month": [1, 2],
        }
    )
    selection = pd.DataFrame({"probability": [0.1, 0.9]})
    result = ReconstructionResult(
        model=_Model(),
        feature_schema=("probability",),
        final_fit=final_fit,
        selection_features=selection,
        metrics=dict(REFERENCE_DEVELOPMENT_METRICS),
        reproduction={"all_metrics_reproduced": True},
    )
    bundle = write_release_bundle(
        result=result,
        bundle_directory=bundle_root,
        policy_path=policy_path,
        route_metadata=route,
        reconstruction_git_sha="a" * 40,
    )
    lock = write_selection_lock(
        path=tmp_path / "selection_lock.json",
        reconstruction_git_sha="a" * 40,
        policy_path=policy_path,
        bundle=bundle,
        route_metadata=route,
        development_metrics=result.metrics,
    )
    verify_locked_files(release_root, lock)
    assert set(path.name for path in bundle_root.iterdir()) == {
        "model.joblib",
        "feature_schema.json",
        "threshold.json",
        "training_baseline.json",
        "metrics_development.json",
        "metadata.json",
        "MODEL_CARD.md",
        "release_policy.yaml",
    }
    assert lock["candidate_id"] == "R3-sigmoid"
    assert lock["final_test_evaluated"] is False


def test_reconstructs_only_locked_candidate_without_test_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame({"Distance": [0.1, 0.9], "target": [0, 1]})
    partitions = SimpleNamespace(final_fit=frame, calibration=frame, selection=frame)
    opened: list[str] = []

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        opened.append(path.name)
        return frame

    monkeypatch.setattr(
        "flight_delay.modeling.release.read_manifest",
        lambda _path: {
            "parquet_files": {
                "train": {"sha256": "valid"},
                "validation": {"sha256": "valid"},
            }
        },
    )
    monkeypatch.setattr("flight_delay.modeling.release.sha256_file", lambda _path: "valid")
    monkeypatch.setattr("flight_delay.modeling.release.pd.read_parquet", fake_read_parquet)
    monkeypatch.setattr(
        "flight_delay.modeling.release.partition_remediation_data",
        lambda _train, _validation: partitions,
    )
    monkeypatch.setattr(
        "flight_delay.modeling.release.build_remediation_model",
        lambda config_id, _parameters: (_Model(), ("Distance",))
        if config_id == "R3"
        else pytest.fail("unexpected candidate"),
    )
    monkeypatch.setattr(
        "flight_delay.modeling.release.fit_calibrator",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        "flight_delay.modeling.release._development_metrics",
        lambda *_args, **_kwargs: dict(REFERENCE_DEVELOPMENT_METRICS),
    )
    result = reconstruct_r3(tmp_path)
    assert result.reproduction["all_metrics_reproduced"] is True
    assert opened == ["train.parquet", "validation.parquet"]
    assert "test.parquet" not in opened
