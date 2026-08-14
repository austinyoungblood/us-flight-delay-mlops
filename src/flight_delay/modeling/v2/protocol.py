"""Strict offline validation for the immutable governed v2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from flight_delay.data.prepare import PROCESSED_FEATURES
from flight_delay.features.leakage import FeatureLeakageError, validate_model_features

PROTOCOL_ID = "us-flight-delay-v2-historical-propensity-dual-boost-v1"
PROTOCOL_SHA256 = "8e57b0f63656003c9981b3b5e44623e0b7c556f6e0c7222352ac38dd5119420a"
PROTOCOL_LOCK_SHA256 = "fdf1318e0222e6a9effca0e7bc381be11dc800ea45546782d8d20d0385be5073"
PROTOCOL_COMMIT_SHA = "226ddd2c279cc4dd087ce3d2daab64c7aad1682c"
BASE_GIT_SHA = "d80f837007836576766cb92e9b11e17f9b6c1ee3"
LIGHTGBM_CANDIDATES_SHA256 = "71849b773eabc506b6eda2763bbd1be693a9167aee4a73e1433e43d5749de0d0"
CATBOOST_CANDIDATES_SHA256 = "d932ca00f40b60775e13dcff5d6b1a3cadd614802afee3cf4c26481bb1b8b4d8"
CATBOOST_SEARCH_SHA256 = "1715e6385513f736c87887c7562ae3de5c7adaf93e41e9f6fb526f2db6533886"

LIGHTGBM_COMMON_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "random_state": 42,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "subsample_freq": 1,
}
CATBOOST_COMMON_PARAMETERS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": 42,
    "has_time": True,
    "allow_writing_files": False,
    "verbose": False,
}
PRETRAINING_CORRECTION: dict[str, Any] = {
    "id": "activate_lightgbm_row_subsampling_v1",
    "applied_before_any_real_fit": True,
    "observed_model_results_available": False,
    "observed_model_results_influenced_correction": False,
    "original_declared_dimension": "subsample: [0.8, 1.0]",
    "inert_default": "subsample_freq: 0",
    "governed_activation": "subsample_freq: 1",
    "candidate_identity_rows_changed": False,
    "catboost_protocol_changed": False,
}

HISTORICAL_FEATURES: tuple[str, ...] = (
    "prior_global_delay_rate",
    "prior_carrier_delay_rate",
    "prior_origin_delay_rate",
    "prior_destination_delay_rate",
    "prior_route_delay_rate",
    "prior_carrier_route_delay_rate",
    "prior_carrier_origin_delay_rate",
    "prior_carrier_destination_delay_rate",
    "prior_origin_departure_hour_delay_rate",
    "prior_destination_arrival_hour_delay_rate",
    "log_route_support",
    "log_carrier_route_support",
    "recent_global_delay_rate_3m",
    "recent_carrier_delay_rate_3m",
    "recent_origin_delay_rate_3m",
    "recent_destination_delay_rate_3m",
    "recent_route_delay_rate_3m",
)
V2_FEATURES: tuple[str, ...] = (*PROCESSED_FEATURES, *HISTORICAL_FEATURES)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "route",
)

LIGHTGBM_DOMAINS: dict[str, tuple[Any, ...]] = {
    "num_leaves": (31, 63, 127),
    "max_depth": (-1, 8, 12),
    "learning_rate": (0.02, 0.03, 0.05),
    "n_estimators": (500, 800, 1200),
    "min_child_samples": (50, 100, 200),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
    "reg_lambda": (1.0, 5.0, 10.0),
    "reg_alpha": (0.0, 0.5, 1.0),
    "cat_smooth": (10, 20, 50),
    "cat_l2": (10, 20),
    "scale_pos_weight": (1.0, 1.25, 1.5),
}
CATBOOST_DOMAINS: dict[str, tuple[Any, ...]] = {
    "depth": (6, 8, 10),
    "iterations": (500, 800, 1200),
    "learning_rate": (0.02, 0.03, 0.05),
    "l2_leaf_reg": (3, 7, 12),
    "random_strength": (0.5, 1.0, 2.0),
    "bagging_temperature": (0.0, 1.0, 3.0),
    "border_count": (128, 254),
    "max_ctr_complexity": (1, 2),
    "positive_class_weight": (1.0, 1.25, 1.5),
}


class V2ProtocolError(ValueError):
    """Raised when the governed v2 protocol or lock has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V2ProtocolError(f"{label} must be a mapping")
    return value


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise V2ProtocolError(f"{label} drifted from the frozen protocol")


def _validate_dependency_tree(value: object, root: Path, label: str = "dependencies") -> None:
    mapping = _mapping(value, label)
    if "path" in mapping or "sha256" in mapping:
        if set(mapping) < {"path", "sha256"}:
            raise V2ProtocolError(f"{label} must contain both path and sha256")
        path = root / str(mapping["path"])
        if not path.is_file() or sha256_file(path) != mapping["sha256"]:
            raise V2ProtocolError(f"{label} artifact hash mismatch")
        if "manifest_digest" in mapping:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise V2ProtocolError(f"{label} manifest is unreadable") from error
            _expect(payload.get("manifest_digest"), mapping["manifest_digest"], label)
        return
    for key, child in mapping.items():
        if key == "base_git_sha":
            continue
        if isinstance(child, dict):
            _validate_dependency_tree(child, root, f"{label}.{key}")


def _validate_candidates(
    protocol: dict[str, Any], lock: dict[str, Any], family: str, count: int
) -> None:
    key = f"{family}_search"
    candidates = _mapping(protocol.get(key), key).get("candidates")
    if not isinstance(candidates, list):
        raise V2ProtocolError(f"{family} candidates must be a list")
    _expect(len(candidates), count, f"{family} candidate count")
    _expect(candidates, lock.get(f"{family}_candidates"), f"{family} locked candidates")
    _expect(
        canonical_sha256(candidates),
        lock.get(f"{family}_candidates_sha256"),
        f"{family} candidate digest",
    )
    expected_digest = (
        LIGHTGBM_CANDIDATES_SHA256 if family == "lightgbm" else CATBOOST_CANDIDATES_SHA256
    )
    _expect(canonical_sha256(candidates), expected_digest, f"{family} original candidate matrix")
    expected_ids = (
        [f"LGBM{index:02d}" for index in range(1, 17)]
        if family == "lightgbm"
        else [f"CB{index:02d}" for index in range(1, 13)]
    )
    _expect([row.get("id") for row in candidates], expected_ids, f"{family} candidate IDs")
    domains = LIGHTGBM_DOMAINS if family == "lightgbm" else CATBOOST_DOMAINS
    for row in candidates:
        _expect(set(row), {"id", *domains}, f"{family} candidate schema")
        for name, values in domains.items():
            if row[name] not in values:
                raise V2ProtocolError(f"{family} candidate {row['id']} has invalid {name}")
        if "task_type" in row or "devices" in row or "n_jobs" in row:
            raise V2ProtocolError("execution backend cannot enter candidate identity")


def _validate_semantics(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    _expect(protocol.get("schema_version"), 1, "schema version")
    _expect(protocol.get("protocol_id"), PROTOCOL_ID, "protocol ID")
    _expect(protocol.get("pretraining_correction"), PRETRAINING_CORRECTION, "correction record")
    _expect(lock.get("pretraining_correction"), PRETRAINING_CORRECTION, "locked correction record")
    state = _mapping(protocol.get("state"), "state")
    _expect(state.get("training_started"), False, "training marker")
    _expect(state.get("results_exist"), False, "results marker")
    _expect(state.get("wandb_runs_created"), False, "W&B marker")
    _expect(state.get("december_opened"), False, "December marker")
    _expect(state.get("historical_test_accessed"), False, "historical-test marker")
    _expect(
        _mapping(protocol.get("dependencies"), "dependencies").get("base_git_sha"),
        BASE_GIT_SHA,
        "base Git SHA",
    )

    contract = _mapping(protocol.get("feature_contract"), "feature contract")
    _expect(contract.get("base_features"), list(PROCESSED_FEATURES), "base features")
    _expect(contract.get("historical_features"), list(HISTORICAL_FEATURES), "historical features")
    _expect(contract.get("total_feature_count"), 37, "feature count")
    _expect(contract.get("categorical_features"), list(CATEGORICAL_FEATURES), "categoricals")
    try:
        validate_model_features(V2_FEATURES)
    except FeatureLeakageError as error:
        raise V2ProtocolError("v2 feature contract violates the central leakage guard") from error

    historical = _mapping(protocol.get("historical_feature_state"), "historical state")
    _expect(historical.get("prior_strength"), 50, "prior strength")
    _expect(historical.get("same_month_labels_permitted"), False, "same-month rule")
    _expect(historical.get("future_labels_permitted"), False, "future-label rule")
    _expect(
        _mapping(historical.get("november_and_december_state"), "frozen state").get(
            "as_of_inclusive"
        ),
        "2025-10-31",
        "November/December state cutoff",
    )

    development = _mapping(protocol.get("development_data"), "development data")
    _expect(development.get("forbidden_sources"), ["data/processed/test.parquet"], "test ban")
    _expect(development.get("december_read_during_development"), False, "December read rule")
    _expect(
        _mapping(development.get("burn_in"), "burn-in").get("contributes_model_rows"),
        False,
        "January burn-in rule",
    )

    folds = _mapping(protocol.get("rolling_origin"), "rolling origin").get("folds")
    expected_cutoffs = ["2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30"]
    if (
        not isinstance(folds, list)
        or [row.get("evaluation_state_as_of") for row in folds] != expected_cutoffs
    ):
        raise V2ProtocolError("rolling feature-state cutoffs drifted")
    _expect(len(folds), 4, "fold count")

    policy = _mapping(protocol.get("execution_policy"), "execution policy")
    _expect(policy.get("sequential_candidate_execution"), True, "sequential execution")
    _expect(
        _mapping(policy.get("catboost_screening"), "CatBoost GPU").get("task_type"),
        "GPU",
        "CatBoost screening backend",
    )
    _expect(
        _mapping(policy.get("lightgbm_screening"), "LightGBM CPU").get("backend"),
        "CPU",
        "LightGBM screening backend",
    )
    _expect(
        _mapping(policy.get("cpu_confirmation"), "CPU confirmation").get(
            "authoritative_for_advancement"
        ),
        True,
        "CPU authority",
    )
    _expect(policy.get("backend_excluded_from_candidate_identity"), True, "backend identity rule")

    lightgbm_search = _mapping(protocol.get("lightgbm_search"), "LightGBM search")
    lightgbm_common = _mapping(
        lightgbm_search.get("common_parameters"), "LightGBM common parameters"
    )
    _expect(lightgbm_common, LIGHTGBM_COMMON_PARAMETERS, "LightGBM common parameters")
    _expect(
        lock.get("lightgbm_common_parameters"),
        LIGHTGBM_COMMON_PARAMETERS,
        "locked LightGBM common parameters",
    )
    _expect(
        lock.get("lightgbm_common_parameters_sha256"),
        canonical_sha256(LIGHTGBM_COMMON_PARAMETERS),
        "LightGBM common-parameter digest",
    )

    catboost_search = _mapping(protocol.get("catboost_search"), "CatBoost search")
    catboost_common = _mapping(
        catboost_search.get("common_parameters"), "CatBoost common parameters"
    )
    _expect(catboost_common, CATBOOST_COMMON_PARAMETERS, "CatBoost common parameters")
    _expect(
        lock.get("catboost_common_parameters"),
        CATBOOST_COMMON_PARAMETERS,
        "locked CatBoost common parameters",
    )
    _expect(
        lock.get("catboost_common_parameters_sha256"),
        canonical_sha256(CATBOOST_COMMON_PARAMETERS),
        "CatBoost common-parameter digest",
    )
    _expect(
        lock.get("catboost_search_sha256"),
        CATBOOST_SEARCH_SHA256,
        "unchanged CatBoost search digest",
    )
    _expect(canonical_sha256(catboost_search), CATBOOST_SEARCH_SHA256, "CatBoost search")

    _validate_candidates(protocol, lock, "lightgbm", 16)
    _validate_candidates(protocol, lock, "catboost", 12)

    advancement = _mapping(protocol.get("advancement"), "advancement")
    _expect(
        _mapping(advancement.get("screening"), "screening").get(
            "top_per_family_to_cpu_confirmation"
        ),
        4,
        "screening advancement",
    )
    _expect(
        _mapping(advancement.get("cpu_confirmation"), "confirmation").get(
            "top_per_family_to_full_refit"
        ),
        2,
        "confirmation advancement",
    )
    _expect(
        _mapping(protocol.get("calibration"), "calibration").get("finalist_count"),
        12,
        "finalist count",
    )

    november = _mapping(protocol.get("november_selection"), "November selection")
    eligibility = _mapping(
        _mapping(november.get("threshold_objective"), "threshold objective").get("eligibility"),
        "threshold eligibility",
    )
    _expect(
        eligibility,
        {"recall_min": 0.60, "precision_min": 0.30, "predicted_positive_rate_max": 0.50},
        "November eligibility",
    )
    _expect(
        _mapping(november.get("acceptance_gates"), "November gates").get("operating_point"),
        {
            "recall_min": 0.60,
            "precision_min": 0.30,
            "f1_min": 0.38,
            "predicted_positive_rate_max": 0.50,
        },
        "November operating gates",
    )
    _expect(
        _mapping(protocol.get("release"), "release").get("production_version_after_v2_development"),
        "v0",
        "production version",
    )


def load_and_validate_v2_protocol(
    protocol_path: Path,
    *,
    lock_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and validate the byte-locked v2 protocol without model imports or network access."""

    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = yaml.safe_load(protocol_bytes)
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise V2ProtocolError("v2 protocol or lock is unreadable") from error
    protocol = _mapping(protocol, "protocol")
    lock = _mapping(lock, "protocol lock")
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    _expect(protocol_sha, PROTOCOL_SHA256, "canonical protocol SHA256")
    _expect(hashlib.sha256(lock_bytes).hexdigest(), PROTOCOL_LOCK_SHA256, "protocol-lock SHA256")
    _expect(lock.get("protocol_sha256"), protocol_sha, "lock protocol SHA256")
    _expect(lock.get("protocol_id"), PROTOCOL_ID, "lock protocol ID")
    _expect(lock.get("base_git_sha"), BASE_GIT_SHA, "lock base Git SHA")
    _expect(lock.get("training_started"), False, "lock training marker")
    _expect(lock.get("results_exist"), False, "lock results marker")
    _expect(lock.get("wandb_runs_created"), False, "lock W&B marker")
    _expect(lock.get("december_opened"), False, "lock December marker")
    _expect(lock.get("historical_test_accessed"), False, "lock historical-test marker")
    _expect(lock.get("production_registry_version"), "v0", "lock production version")
    _validate_semantics(protocol, lock)
    _validate_dependency_tree(protocol.get("dependencies"), repository_root.resolve())
    return protocol, lock, protocol_sha
