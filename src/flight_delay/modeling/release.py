"""Governed reconstruction, immutable evidence, and release gates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from typing import Any
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss

from flight_delay.data.download import sha256_file
from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.data.preprocessing import filter_eligible_flights
from flight_delay.data.reliability import compute_route_reliability
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.artifacts import build_training_baseline
from flight_delay.modeling.calibration import calibration_audit, fit_calibrator
from flight_delay.modeling.evaluation import evaluate_binary, measure_single_row_latency
from flight_delay.modeling.remediation import (
    EXPECTED_MATRIX,
    build_remediation_model,
    partition_remediation_data,
)

DATASET_ARTIFACT = (
    "austin-youngblood-university-of-denver/us-flight-delay-mlops/flight-delay-bts-sampled:v0"
)
DATASET_DIGEST = "2ecdb5a6a60b23ed1ee1d603fb976516"
CANDIDATE_ID = "R3-sigmoid"
CALIBRATION_METHOD = "sigmoid"
LOCKED_THRESHOLD = 0.1840285229739868
REGISTRY_PATH = "wandb-registry-Model/us-flight-arrival-delay-15m"
# The original run used Python 3.11.15; the reproducible environment available for this release
# uses Python 3.11.14. A 1e-9 absolute tolerance covers sub-nanounit solver/calibration drift while
# remaining orders of magnitude below any reported metric or release-gate precision.
REPRODUCTION_ABSOLUTE_TOLERANCE = 1e-9
RELEASE_BUNDLE_FILES = frozenset(
    {
        "model.joblib",
        "feature_schema.json",
        "threshold.json",
        "training_baseline.json",
        "metrics_development.json",
        "metadata.json",
        "MODEL_CARD.md",
        "release_policy.yaml",
    }
)
REFERENCE_DEVELOPMENT_METRICS: dict[str, float | int | bool] = {
    "accuracy": 0.5213801697951553,
    "average_precision": 0.2823880567429311,
    "brier_score": 0.15368595658236373,
    "f1": 0.37015272130923504,
    "false_negative": 2199,
    "false_positive": 16236,
    "log_loss": 0.48140966513208466,
    "mean_probability_gap": 0.008805018760396077,
    "precision": 0.2501731861635801,
    "predicted_positive_rate": 0.5621673546745592,
    "prevalence": 0.1977308720824571,
    "recall": 0.711265756302521,
    "roc_auc": 0.6281178113133866,
    "true_negative": 14665,
    "true_positive": 5417,
    "equal_frequency_ece_15": 0.011190970971699217,
}


class ReleaseGuardError(RuntimeError):
    """Raised when immutable release or one-time evaluation checks fail."""


@dataclass(frozen=True)
class ReconstructionResult:
    model: Any
    feature_schema: tuple[str, ...]
    final_fit: pd.DataFrame
    selection_features: pd.DataFrame
    metrics: dict[str, float | int | bool]
    reproduction: dict[str, Any]


def write_json(path: Path, payload: Any) -> None:
    """Write canonical, newline-terminated JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseGuardError(f"expected an object in {path}")
    return payload


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ReleaseGuardError("release action requires a clean Git worktree")


def require_git_ancestor(ancestor: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], capture_output=True
    )
    if result.returncode:
        raise ReleaseGuardError("locked reconstruction Git SHA is not an ancestor of HEAD")


def validate_release_policy(policy: dict[str, Any]) -> None:
    if policy.get("policy_version") != "brief05-v1":
        raise ReleaseGuardError("unexpected release policy version")
    if policy.get("dataset") != {"artifact": DATASET_ARTIFACT, "digest": DATASET_DIGEST}:
        raise ReleaseGuardError("release policy dataset lineage changed")
    candidate = policy.get("candidate", {})
    if candidate.get("configuration_id") != "R3" or candidate.get("calibrator") != "sigmoid":
        raise ReleaseGuardError("release policy candidate changed")
    if (
        candidate.get("re_rank_candidates")
        or candidate.get("retune_model")
        or candidate.get("retune_threshold")
    ):
        raise ReleaseGuardError("release policy permits unauthorized search")


def load_release_policy(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseGuardError("release policy must be a mapping")
    validate_release_policy(payload)
    return payload


def _development_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    evaluation = evaluate_binary(target, probabilities, threshold=LOCKED_THRESHOLD)
    for figure in evaluation.figures.values():
        import matplotlib.pyplot as plt

        plt.close(figure)
    audit = calibration_audit(target, probabilities)
    return {
        **evaluation.metrics,
        "prevalence": float(target.mean()),
        "mean_probability_gap": audit.mean_probability_gap,
        "equal_width_ece_10": audit.equal_width_ece_10,
        "equal_frequency_ece_15": audit.equal_frequency_ece_15,
        "equal_frequency_mce_15": audit.equal_frequency_mce_15,
        "lineage_verified": True,
        "schema_check_passed": True,
        "leakage_check_passed": True,
        "convergence_check_passed": True,
    }


def compare_development_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    passed = True
    for name, expected in REFERENCE_DEVELOPMENT_METRICS.items():
        observed = metrics[name]
        if isinstance(expected, bool | int) and not isinstance(expected, float):
            difference = 0 if observed == expected else 1
            within = observed == expected
        else:
            difference = abs(float(observed) - float(expected))
            within = difference <= REPRODUCTION_ABSOLUTE_TOLERANCE
        comparisons[name] = {
            "expected": expected,
            "observed": observed,
            "absolute_difference": difference,
            "within_tolerance": within,
        }
        passed = passed and within
    return {
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "all_metrics_reproduced": passed,
        "comparisons": comparisons,
    }


def _verify_serialization_round_trip(
    model: Any, features: pd.DataFrame, expected_probabilities: np.ndarray
) -> None:
    """Fail closed unless an in-memory joblib round trip preserves predictions."""

    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    restored_probabilities = restored.predict_proba(features)[:, 1]
    if restored_probabilities.shape != expected_probabilities.shape or not np.allclose(
        expected_probabilities, restored_probabilities
    ):
        raise ReleaseGuardError("R3 in-memory serialization check failed")


def reconstruct_r3(dataset_root: Path) -> ReconstructionResult:
    """Rebuild only locked R3 sigmoid without reading the final-test file."""

    train_path = dataset_root / "data/processed/train.parquet"
    validation_path = dataset_root / "data/processed/validation.parquet"
    manifest = read_manifest(dataset_root / "data/manifests/processed_manifest.json")
    if sha256_file(train_path) != manifest["parquet_files"]["train"]["sha256"]:
        raise ReleaseGuardError("training split hash mismatch")
    if sha256_file(validation_path) != manifest["parquet_files"]["validation"]["sha256"]:
        raise ReleaseGuardError("development validation split hash mismatch")
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    partitions = partition_remediation_data(train, validation)
    parameters = dict(EXPECTED_MATRIX["R3"])
    base, schema = build_remediation_model("R3", parameters)
    validate_model_features(schema)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base.fit(partitions.final_fit.loc[:, schema], partitions.final_fit["target"])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ReleaseGuardError("R3 reconstruction emitted a convergence warning")
    calibrated = fit_calibrator(
        base,
        partitions.calibration.loc[:, schema],
        partitions.calibration["target"],
        method=CALIBRATION_METHOD,
    )
    selection_features = partitions.selection.loc[:, schema]
    probabilities = calibrated.predict_proba(selection_features)[:, 1]
    metrics = _development_metrics(partitions.selection["target"], probabilities)
    _verify_serialization_round_trip(calibrated, selection_features, probabilities)
    metrics["serialization_check_passed"] = True
    reproduction = compare_development_metrics(metrics)
    if not reproduction["all_metrics_reproduced"]:
        failures = {
            name: evidence
            for name, evidence in reproduction["comparisons"].items()
            if not evidence["within_tolerance"]
        }
        raise ReleaseGuardError(
            "R3 development metrics did not reproduce within tolerance: "
            f"{json.dumps(failures, sort_keys=True)}"
        )
    return ReconstructionResult(
        calibrated,
        tuple(schema),
        partitions.final_fit,
        selection_features,
        metrics,
        reproduction,
    )


def build_route_asset(
    *, source_manifest_path: Path, raw_directory: Path, output_path: Path, min_support: int = 30
) -> dict[str, Any]:
    """Build display-only reliability from every eligible completed 2025 source flight."""

    manifest = read_manifest(source_manifest_path)
    monthly_frames: list[pd.DataFrame] = []
    source_rows = eligible_rows = 0
    usecols = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Cancelled",
        "Diverted",
        "ArrDel15",
        "ArrDelay",
    ]
    records = [record for record in manifest["files"] if int(record["year"]) == 2025]
    if len(records) != 12:
        raise ReleaseGuardError("route asset requires exactly 12 source months for 2025")
    for record in records:
        archive_path = raw_directory / record["archive_filename"]
        if sha256_file(archive_path) != record["sha256"]:
            raise ReleaseGuardError(f"route source hash mismatch: {archive_path.name}")
        with (
            ZipFile(archive_path) as archive,
            archive.open(record["selected_csv_member"]) as source,
        ):
            frame = pd.read_csv(
                source,
                usecols=usecols,
                dtype={name: "string" for name in ("Reporting_Airline", "Origin", "Dest")},
            )
        eligibility = filter_eligible_flights(frame)
        source_rows += len(frame)
        eligible_rows += len(eligibility.frame)
        compact = eligibility.frame.loc[
            :, ["Reporting_Airline", "Origin", "Dest", "ArrDel15", "ArrDelay"]
        ].copy()
        for column in ("Reporting_Airline", "Origin", "Dest"):
            compact[column] = compact[column].astype("category")
        compact["ArrDel15"] = pd.to_numeric(compact["ArrDel15"], errors="raise").astype("int8")
        compact["ArrDelay"] = pd.to_numeric(compact["ArrDelay"], errors="coerce").astype("float32")
        monthly_frames.append(compact)
    combined = pd.concat(monthly_frames, ignore_index=True)
    for column in ("Reporting_Airline", "Origin", "Dest"):
        combined[column] = combined[column].astype("category")
    route_stats = compute_route_reliability(combined, min_support=min_support)
    if set(route_stats.columns) & set(EXPECTED_MATRIX["R3"]):
        raise ReleaseGuardError("route asset leaked into model configuration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_stats.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
    return {
        "display_only": True,
        "source_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "source_manifest_digest": manifest["manifest_digest"],
        "source_month_count": len(records),
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "route_stat_rows": len(route_stats),
        "minimum_support": min_support,
        "sha256": sha256_file(output_path),
        "byte_size": output_path.stat().st_size,
    }


def _bundle_hashes(directory: Path) -> dict[str, str]:
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != RELEASE_BUNDLE_FILES:
        raise ReleaseGuardError(f"release bundle contract mismatch: {sorted(observed)}")
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file()}


def aggregate_digest(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(file_hashes)).hexdigest()


def write_release_bundle(
    *,
    result: ReconstructionResult,
    bundle_directory: Path,
    policy_path: Path,
    route_metadata: dict[str, Any],
    reconstruction_git_sha: str,
) -> dict[str, Any]:
    bundle_directory.mkdir(parents=True, exist_ok=True)
    if any(bundle_directory.iterdir()):
        raise ReleaseGuardError("release bundle directory must be empty")
    joblib.dump(result.model, bundle_directory / "model.joblib")
    write_json(bundle_directory / "feature_schema.json", {"features": list(result.feature_schema)})
    write_json(
        bundle_directory / "threshold.json",
        {"threshold": LOCKED_THRESHOLD, "source": "R3 sigmoid locked remediation selection"},
    )
    write_json(
        bundle_directory / "training_baseline.json",
        build_training_baseline(
            result.final_fit, dataset_artifact=DATASET_ARTIFACT, dataset_digest=DATASET_DIGEST
        ),
    )
    write_json(
        bundle_directory / "metrics_development.json",
        {
            "split": "2025-11-16/2025-11-30",
            "metrics": result.metrics,
            "reproduction": result.reproduction,
        },
    )
    write_json(
        bundle_directory / "metadata.json",
        {
            "candidate_id": CANDIDATE_ID,
            "configuration": dict(EXPECTED_MATRIX["R3"]),
            "calibration_method": CALIBRATION_METHOD,
            "partitions": {
                "base_fit": "2025-01-01/2025-10-31",
                "calibration": "2025-11-01/2025-11-15",
                "development_selection": "2025-11-16/2025-11-30",
            },
            "dataset_artifact": DATASET_ARTIFACT,
            "dataset_digest": DATASET_DIGEST,
            "reconstruction_git_sha": reconstruction_git_sha,
            "route_asset": route_metadata,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "final_test_evaluated": False,
        },
    )
    copyfile(policy_path, bundle_directory / "release_policy.yaml")
    (bundle_directory / "MODEL_CARD.md").write_text(
        "# R3 sigmoid release candidate\n\n"
        "This immutable candidate uses only pre-departure schedule features. It was fitted on "
        "January-October 2025 and calibrated on November 1-15, 2025. The threshold was locked "
        "from the November 16-30 remediation selection evidence. `route_stats.parquet` is "
        "display-only "
        "and is never a model input. At bundle creation, the January-May 2026 final test remained "
        "sealed. See `release_policy.yaml` for the one-time release gate.\n",
        encoding="utf-8",
    )
    restored = joblib.load(bundle_directory / "model.joblib")
    before = result.model.predict_proba(result.selection_features.iloc[:100])[:, 1]
    after = restored.predict_proba(result.selection_features.iloc[:100])[:, 1]
    if not np.allclose(before, after, rtol=0, atol=1e-12):
        raise ReleaseGuardError("release bundle serialization check failed")
    hashes = _bundle_hashes(bundle_directory)
    return {
        "file_hashes": hashes,
        "byte_size": sum(path.stat().st_size for path in bundle_directory.iterdir()),
        "aggregate_digest": aggregate_digest(hashes),
        "model_load_and_inference_check_passed": True,
    }


def write_selection_lock(
    *,
    path: Path,
    reconstruction_git_sha: str,
    policy_path: Path,
    bundle: dict[str, Any],
    route_metadata: dict[str, Any],
    development_metrics: dict[str, Any],
) -> dict[str, Any]:
    locked_hashes = {
        **{f"model_bundle/{name}": digest for name, digest in bundle["file_hashes"].items()},
        "route_stats.parquet": route_metadata["sha256"],
    }
    payload = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "reconstruction_git_sha": reconstruction_git_sha,
        "dataset_artifact": DATASET_ARTIFACT,
        "dataset_digest": DATASET_DIGEST,
        "configuration": dict(EXPECTED_MATRIX["R3"]),
        "calibration_method": CALIBRATION_METHOD,
        "calibration_period": "2025-11-01/2025-11-15",
        "development_selection_period": "2025-11-16/2025-11-30",
        "threshold": LOCKED_THRESHOLD,
        "development_metrics": development_metrics,
        "policy_sha256": sha256_file(policy_path),
        "file_hashes": locked_hashes,
        "aggregate_bundle_digest": aggregate_digest(locked_hashes),
        "bundle_size_bytes": bundle["byte_size"] + route_metadata["byte_size"],
        "route_asset": route_metadata,
        "final_test_evaluated": False,
    }
    write_json(path, payload)
    return payload


def verify_locked_files(root: Path, lock: dict[str, Any]) -> None:
    for relative, expected in lock["file_hashes"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReleaseGuardError(f"locked file hash mismatch: {relative}")
    if aggregate_digest(lock["file_hashes"]) != lock["aggregate_bundle_digest"]:
        raise ReleaseGuardError("aggregate bundle digest mismatch")


def create_one_time_marker(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create the durable marker; an existing marker always refuses evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ReleaseGuardError("final-test marker already exists; rerun refused") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")


def final_test_metrics(
    *, model: Any, features: pd.DataFrame, target: pd.Series, threshold: float, bundle_size: int
) -> tuple[dict[str, Any], dict[str, bool]]:
    probabilities = model.predict_proba(features)[:, 1]
    evaluation = evaluate_binary(target, probabilities, threshold=threshold)
    for figure in evaluation.figures.values():
        import matplotlib.pyplot as plt

        plt.close(figure)
    audit = calibration_audit(target, probabilities)
    prevalence = float(target.mean())
    prior_probabilities = np.full(len(target), prevalence)
    prior_brier = float(brier_score_loss(target, prior_probabilities))
    prior_log_loss = float(log_loss(target, prior_probabilities, labels=[0, 1]))
    latency = measure_single_row_latency(model, features, sample_count=200)
    metrics = {
        **evaluation.metrics,
        "prevalence": prevalence,
        "average_precision_lift_over_prevalence": float(
            evaluation.metrics["average_precision"] / prevalence
        ),
        "constant_prior_brier_score": prior_brier,
        "brier_skill_score": float(1 - evaluation.metrics["brier_score"] / prior_brier),
        "constant_prior_log_loss": prior_log_loss,
        "mean_probability_gap": audit.mean_probability_gap,
        "equal_frequency_ece_15": audit.equal_frequency_ece_15,
        "equal_frequency_mce_15": audit.equal_frequency_mce_15,
        "inference_latency_p95_ms": latency["p95_ms"],
        "inference_latency_sample_count": latency["sample_count"],
        "bundle_size_bytes": bundle_size,
    }
    gates = {
        "roc_auc": metrics["roc_auc"] >= 0.58,
        "average_precision_lift": metrics["average_precision_lift_over_prevalence"] >= 1.20,
        "brier_skill_score": metrics["brier_skill_score"] > 0,
        "log_loss_beats_prior": metrics["log_loss"] < prior_log_loss,
        "mean_probability_gap": metrics["mean_probability_gap"] <= 0.05,
        "equal_frequency_ece_15": metrics["equal_frequency_ece_15"] <= 0.05,
        "inference_latency_p95": metrics["inference_latency_p95_ms"] < 25,
        "bundle_size": bundle_size < 10 * 1024 * 1024,
        "exact_lineage": True,
        "feature_schema": True,
        "leakage_guard": True,
        "serialization": True,
        "model_load": True,
        "inference_contract": bool(
            len(probabilities) == len(target)
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all()
        ),
    }
    return metrics, gates
