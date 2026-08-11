"""Execute the governed Brief 05 bundle, Registry, and one-time test workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv

import wandb
from flight_delay.data.download import sha256_file
from flight_delay.data.manifest import read_manifest
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.release import (
    CANDIDATE_ID,
    DATASET_ARTIFACT,
    DATASET_DIGEST,
    LOCKED_THRESHOLD,
    REGISTRY_PATH,
    ReleaseGuardError,
    build_route_asset,
    create_one_time_marker,
    final_test_metrics,
    git_sha,
    load_release_policy,
    read_json,
    reconstruct_r3,
    require_clean_worktree,
    require_git_ancestor,
    verify_locked_files,
    write_json,
    write_release_bundle,
    write_selection_lock,
)

DATASET_ROOT = Path("artifacts/brief04/dataset-v0")
RELEASE_ROOT = Path("artifacts/brief05/release")
BUNDLE_ROOT = RELEASE_ROOT / "model_bundle"
LOCK_PATH = Path("release/selection_lock.json")
MARKER_PATH = Path("release/final_test_marker.json")
RESULT_PATH = Path("release/final_test_result.json")
DECISION_PATH = Path("release/release_decision.json")
POLICY_PATH = Path("configs/release_policy.yaml")
ARTIFACT_NAME = "flight-delay-r3-sigmoid-release"


def _tracking() -> tuple[str, str]:
    load_dotenv()
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise ReleaseGuardError("WANDB_ENTITY and WANDB_PROJECT are required")
    return entity, project


def _registry_artifact(alias: str) -> Any:
    api = wandb.Api()
    return api.artifact(f"{REGISTRY_PATH}:{alias}")


def _registry_evidence(artifact: Any) -> dict[str, Any]:
    source = artifact.source_artifact if getattr(artifact, "is_link", False) else artifact
    return {
        "registry_path": REGISTRY_PATH,
        "registry_name": artifact.name,
        "registry_version": artifact.version,
        "registry_digest": artifact.digest,
        "aliases": sorted(artifact.aliases),
        "source_artifact_name": source.name,
        "source_artifact_version": source.version,
        "source_artifact_digest": source.digest,
    }


def prepare() -> None:
    require_clean_worktree()
    load_release_policy(POLICY_PATH)
    reconstruction_sha = git_sha()
    if RELEASE_ROOT.exists():
        raise ReleaseGuardError(f"release output already exists: {RELEASE_ROOT}")
    result = reconstruct_r3(DATASET_ROOT)
    route_metadata = build_route_asset(
        source_manifest_path=Path("data/manifests/source_manifest.json"),
        raw_directory=Path("data/raw/bts_reporting_carrier"),
        output_path=RELEASE_ROOT / "route_stats.parquet",
    )
    bundle = write_release_bundle(
        result=result,
        bundle_directory=BUNDLE_ROOT,
        policy_path=POLICY_PATH,
        route_metadata=route_metadata,
        reconstruction_git_sha=reconstruction_sha,
    )
    lock = write_selection_lock(
        path=LOCK_PATH,
        reconstruction_git_sha=reconstruction_sha,
        policy_path=POLICY_PATH,
        bundle=bundle,
        route_metadata=route_metadata,
        development_metrics=result.metrics,
    )
    verify_locked_files(RELEASE_ROOT, lock)
    print(
        json.dumps(
            {
                "candidate": CANDIDATE_ID,
                "reconstruction_git_sha": reconstruction_sha,
                "reproduced": result.reproduction["all_metrics_reproduced"],
                "aggregate_bundle_digest": lock["aggregate_bundle_digest"],
                "bundle_size_bytes": lock["bundle_size_bytes"],
                "selection_lock": str(LOCK_PATH),
            },
            sort_keys=True,
        )
    )


def stage() -> None:
    require_clean_worktree()
    lock = read_json(LOCK_PATH)
    require_git_ancestor(lock["reconstruction_git_sha"])
    if sha256_file(POLICY_PATH) != lock["policy_sha256"]:
        raise ReleaseGuardError("release policy hash changed")
    verify_locked_files(RELEASE_ROOT, lock)
    entity, project = _tracking()
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="brief05-release-artifact",
        name="brief05-r3-sigmoid-release",
        tags=["brief05", "release", "staging"],
        config={
            "candidate_id": CANDIDATE_ID,
            "dataset_artifact": DATASET_ARTIFACT,
            "dataset_digest": DATASET_DIGEST,
            "reconstruction_git_sha": lock["reconstruction_git_sha"],
            "selection_lock_git_sha": git_sha(),
            "bundle_digest": lock["aggregate_bundle_digest"],
            "final_test_evaluated": False,
        },
        settings=wandb.Settings(code_dir="."),
    )
    try:
        artifact = wandb.Artifact(
            ARTIFACT_NAME,
            type="model",
            description="Immutable Brief 05 R3 sigmoid release candidate",
            metadata={
                "candidate_id": CANDIDATE_ID,
                "dataset_artifact": DATASET_ARTIFACT,
                "dataset_digest": DATASET_DIGEST,
                "bundle_digest": lock["aggregate_bundle_digest"],
                "final_test_evaluated": False,
            },
        )
        artifact.add_dir(str(BUNDLE_ROOT), name="model_bundle")
        artifact.add_file(str(RELEASE_ROOT / "route_stats.parquet"), name="route_stats.parquet")
        artifact.add_file(str(LOCK_PATH), name="selection_lock.json")
        logged = run.log_artifact(artifact, aliases=["brief05-staging-source"])
        logged.wait()
        linked = logged.link(REGISTRY_PATH, aliases=["staging"])
        linked.wait()
        run.summary.update(
            {
                "source_artifact": logged.name,
                "source_artifact_digest": logged.digest,
                "registry_version": linked.version,
                "registry_digest": linked.digest,
                "staging_verified": False,
            }
        )
        run_id = run.id
        run_url = run.url
    finally:
        run.finish()
    staging = _registry_artifact("staging")
    clean_root = Path("artifacts/brief05/registry-download-staging")
    if clean_root.exists():
        shutil.rmtree(clean_root)
    download_root = Path(staging.download(root=str(clean_root)))
    downloaded_lock = download_root / "selection_lock.json"
    if sha256_file(downloaded_lock) != sha256_file(LOCK_PATH):
        raise ReleaseGuardError("downloaded selection lock differs from committed lock")
    verify_locked_files(download_root, lock)
    evidence = {
        **_registry_evidence(staging),
        "release_run_id": run_id,
        "release_run_url": run_url,
        "selection_lock_sha256": sha256_file(LOCK_PATH),
        "aggregate_bundle_digest": lock["aggregate_bundle_digest"],
        "clean_download_verified": True,
        "production_created": False,
    }
    write_json(RELEASE_ROOT / "registry_staging_evidence.json", evidence)
    print(json.dumps(evidence, sort_keys=True))


def _guard_staging(lock: dict[str, Any]) -> tuple[Any, Path, dict[str, Any]]:
    staging = _registry_artifact("staging")
    clean_root = Path("artifacts/brief05/final-test-staging-download")
    if clean_root.exists():
        shutil.rmtree(clean_root)
    download_root = Path(staging.download(root=str(clean_root)))
    if sha256_file(download_root / "selection_lock.json") != sha256_file(LOCK_PATH):
        raise ReleaseGuardError("staging does not resolve to the committed selection lock")
    verify_locked_files(download_root, lock)
    return staging, download_root, _registry_evidence(staging)


def evaluate_once() -> None:
    require_clean_worktree()
    lock = read_json(LOCK_PATH)
    if lock.get("final_test_evaluated") is not False:
        raise ReleaseGuardError("selection lock does not declare a sealed final test")
    require_git_ancestor(lock["reconstruction_git_sha"])
    if (
        lock.get("dataset_artifact") != DATASET_ARTIFACT
        or lock.get("dataset_digest") != DATASET_DIGEST
    ):
        raise ReleaseGuardError("selection-lock dataset lineage mismatch")
    if sha256_file(POLICY_PATH) != lock["policy_sha256"]:
        raise ReleaseGuardError("release policy is missing or altered")
    staging, downloaded_root, registry = _guard_staging(lock)
    create_one_time_marker(
        MARKER_PATH,
        {
            "status": "started",
            "git_sha": git_sha(),
            "selection_lock_sha256": sha256_file(LOCK_PATH),
            "bundle_digest": lock["aggregate_bundle_digest"],
            "registry_version": registry["registry_version"],
            "registry_digest": registry["registry_digest"],
        },
    )

    # This is the sole code path that opens the sealed final-test parquet.
    manifest = read_manifest(DATASET_ROOT / "data/manifests/processed_manifest.json")
    test_path = DATASET_ROOT / "data/processed/test.parquet"
    if sha256_file(test_path) != manifest["parquet_files"]["test"]["sha256"]:
        raise ReleaseGuardError("final-test split hash mismatch")
    test = pd.read_parquet(test_path)
    schema = read_json(downloaded_root / "model_bundle/feature_schema.json")["features"]
    validate_model_features(schema)
    model = joblib.load(downloaded_root / "model_bundle/model.joblib")
    threshold = read_json(downloaded_root / "model_bundle/threshold.json")["threshold"]
    if threshold != LOCKED_THRESHOLD or threshold != lock["threshold"]:
        raise ReleaseGuardError("locked threshold mismatch")
    metrics, gates = final_test_metrics(
        model=model,
        features=test.loc[:, schema],
        target=test["target"],
        threshold=threshold,
        bundle_size=lock["bundle_size_bytes"],
    )
    passed = all(gates.values())
    entity, project = _tracking()
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="brief05-final-test",
        name="brief05-one-time-final-test",
        tags=["brief05", "final-test", "one-time"],
        config={
            "candidate_id": CANDIDATE_ID,
            "dataset_artifact": DATASET_ARTIFACT,
            "dataset_digest": DATASET_DIGEST,
            "git_sha": git_sha(),
            "bundle_digest": lock["aggregate_bundle_digest"],
            "registry_version": registry["registry_version"],
            "registry_digest": registry["registry_digest"],
            "threshold": threshold,
            "final_test_evaluated": True,
        },
        settings=wandb.Settings(code_dir="."),
    )
    try:
        run.log({f"final_test/{key}": value for key, value in metrics.items()})
        run.log({f"final_test_gate/{key}": value for key, value in gates.items()})
        run.summary["all_production_gates_passed"] = passed
        run_id, run_url = run.id, run.url
    finally:
        run.finish()
    result = {
        "candidate_id": CANDIDATE_ID,
        "dataset_artifact": DATASET_ARTIFACT,
        "dataset_digest": DATASET_DIGEST,
        "selection_lock_sha256": sha256_file(LOCK_PATH),
        "bundle_digest": lock["aggregate_bundle_digest"],
        "registry": registry,
        "threshold": threshold,
        "metrics": metrics,
        "gates": gates,
        "all_production_gates_passed": passed,
        "final_test_run_id": run_id,
        "final_test_run_url": run_url,
    }
    write_json(RESULT_PATH, result)
    write_json(
        MARKER_PATH,
        {
            "status": "complete",
            "git_sha": git_sha(),
            "selection_lock_sha256": sha256_file(LOCK_PATH),
            "bundle_digest": lock["aggregate_bundle_digest"],
            "registry_version": registry["registry_version"],
            "registry_digest": registry["registry_digest"],
            "final_test_result_sha256": sha256_file(RESULT_PATH),
            "final_test_run_id": run_id,
            "evaluation_count": 1,
        },
    )
    serving_alias = "staging"
    if passed:
        production = staging.link(REGISTRY_PATH, aliases=["production"])
        production.wait()
        verified = _registry_artifact("production")
        production_root = Path("artifacts/brief05/registry-download-production")
        if production_root.exists():
            shutil.rmtree(production_root)
        verified_root = Path(verified.download(root=str(production_root)))
        if sha256_file(verified_root / "selection_lock.json") != sha256_file(LOCK_PATH):
            raise ReleaseGuardError("production clean-download lock mismatch")
        verify_locked_files(verified_root, lock)
        registry = _registry_evidence(verified)
        registry["clean_download_verified"] = True
        serving_alias = "production"
    decision = {
        "serving_alias": serving_alias,
        "registry_path": REGISTRY_PATH,
        "registry_version": registry["registry_version"],
        "registry_digest": registry["registry_digest"],
        "source_artifact_name": registry["source_artifact_name"],
        "source_artifact_version": registry["source_artifact_version"],
        "source_artifact_digest": registry["source_artifact_digest"],
        "bundle_digest": lock["aggregate_bundle_digest"],
        "final_test_passed": passed,
        "failed_gates": sorted(name for name, value in gates.items() if not value),
        "final_test_result_sha256": sha256_file(RESULT_PATH),
    }
    write_json(DECISION_PATH, decision)
    print(json.dumps({"result": result, "decision": decision}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("prepare", "stage", "evaluate-once"))
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        {"prepare": prepare, "stage": stage, "evaluate-once": evaluate_once}[args.command]()
    except (OSError, ValueError, ReleaseGuardError, wandb.Error) as error:
        print(f"Brief 05 release failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
