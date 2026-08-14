"""Strict, offline validation for the pre-training v1 experiment protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from flight_delay.data.prepare import PROCESSED_FEATURES
from flight_delay.features.leakage import FeatureLeakageError, validate_model_features

PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_ID = "us-flight-delay-v1-catboost-rolling-origin-v1"
CANONICAL_PROTOCOL_SHA256 = "a6b1de9de550d1bd94eae0e56f8d88d65801ec488b6c539fc64afbafa4ccfffb"
BASE_GIT_SHA = "fdfb2bfe0386d016b1deaa90d946436bdc3ed14f"
HISTORICAL_TEST_PERIOD = "2026-01-01/2026-05-31"

EXPECTED_INCUMBENT = {
    "registry_collection": "wandb-registry-Model/us-flight-arrival-delay-15m",
    "serving_alias": "production",
    "registry_version": "v0",
    "registry_digest": "865ddd18f6debd44f24a79fc71739f2a",
    "bundle_sha256": "2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572",
    "threshold": 0.1840285229739868,
    "internal_production_gate_passed": False,
    "deployment_purpose": "academic_demo",
    "production_mutation_prohibited": True,
}

EXPECTED_DEPENDENCIES = {
    "base_git_sha": BASE_GIT_SHA,
    "incumbent_selection_lock": {
        "path": "release/selection_lock.json",
        "sha256": "a730a25c34a9f259b3ca02eb92c4ad44c1e75f50fd52ce270a940e4a60142340",
    },
    "processed_dataset_manifest": {
        "path": "data/manifests/processed_manifest.json",
        "sha256": "fa328e92048dededdcbdb4b61ce8c74b67b2d0ef64872dfb25949334cf22dee4",
        "manifest_digest": "c8aa583cbfe7ad8ee4bdcedaa8d479e2056541c71296f222ae0e0a410a48cdaf",
    },
    "source_dataset_manifest": {
        "path": "data/manifests/source_manifest.json",
        "sha256": "996b1172a725bbf733f6647d436ebc30eecd1b14b2d6a3099ebeb1cd78ac9e83",
        "manifest_digest": "0a2e8ef929dee6bc1d5a8cdcf8f0161ce0f11bb54c3809dbea1040f09624b561",
    },
}

EXPECTED_CATEGORICAL_FEATURES = ["Reporting_Airline", "Origin", "Dest", "route"]
EXPECTED_CANDIDATES = [
    {"id": "CB1", "depth": 6, "iterations": 300, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"id": "CB2", "depth": 8, "iterations": 300, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"id": "CB3", "depth": 6, "iterations": 500, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"id": "CB4", "depth": 8, "iterations": 500, "learning_rate": 0.03, "l2_leaf_reg": 7},
]
EXPECTED_COMMON_PARAMETERS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "task_type": "CPU",
    "random_seed": 42,
    "has_time": True,
    "allow_writing_files": False,
    "verbose": False,
    "class_weights": None,
    "auto_class_weights": None,
    "early_stopping": "disabled",
}
EXPECTED_FOLDS = [
    {
        "id": "FOLD_1",
        "train_start": "2025-01-01",
        "train_end_exclusive": "2025-07-01",
        "validation_start": "2025-07-01",
        "validation_end_exclusive": "2025-08-01",
    },
    {
        "id": "FOLD_2",
        "train_start": "2025-01-01",
        "train_end_exclusive": "2025-08-01",
        "validation_start": "2025-08-01",
        "validation_end_exclusive": "2025-09-01",
    },
    {
        "id": "FOLD_3",
        "train_start": "2025-01-01",
        "train_end_exclusive": "2025-09-01",
        "validation_start": "2025-09-01",
        "validation_end_exclusive": "2025-10-01",
    },
    {
        "id": "FOLD_4",
        "train_start": "2025-01-01",
        "train_end_exclusive": "2025-10-01",
        "validation_start": "2025-10-01",
        "validation_end_exclusive": "2025-11-01",
    },
]
EXPECTED_THRESHOLD_OBJECTIVE = {
    "constraints": {
        "recall_min": 0.60,
        "precision_min": 0.30,
        "predicted_positive_rate_max": 0.50,
    },
    "objective": "maximize_f1",
    "tie_break_order": [
        "f1_desc",
        "precision_desc",
        "recall_desc",
        "predicted_positive_rate_asc",
        "distance_from_0_50_asc",
        "threshold_desc",
    ],
    "no_eligible_threshold_outcome": "finalist_fails",
}
EXPECTED_NOVEMBER_GATES = {
    "discrimination": {
        "average_precision_incumbent_margin_min": 0.01,
        "roc_auc_incumbent_margin_min": 0.005,
        "average_precision_lift_over_prevalence_min": 1.35,
    },
    "proper_scoring": {
        "brier_skill_score_vs_prior_strictly_positive": True,
        "brier_score_below_prior": True,
        "log_loss_below_prior": True,
        "brier_score_incumbent_max": 0.153685956583994,
        "log_loss_incumbent_max": 0.48140966513585765,
    },
    "calibration": {
        "absolute_probability_prevalence_gap_max": 0.03,
        "equal_frequency_ece_15_max": 0.03,
    },
    "operating_point": {
        "recall_min": 0.60,
        "precision_min": 0.30,
        "f1_min": 0.38,
        "predicted_positive_rate_max": 0.50,
    },
    "operational": {
        "single_row_inference_p95_ms_strict_max": 25,
        "serialized_bundle_bytes_strict_max": 10_485_760,
    },
    "governance": {
        "lineage_verified": True,
        "schema_check_passed": True,
        "leakage_check_passed": True,
        "deterministic_reconstruction_check_passed": True,
        "serialization_load_inference_check_passed": True,
        "no_prohibited_test_access": True,
        "no_training_convergence_or_runtime_failure": True,
    },
}
EXPECTED_QUALIFICATION_GATES = {
    "brier_skill_score_vs_prior_strictly_positive": True,
    "log_loss_below_prior": True,
    "absolute_probability_prevalence_gap_max": 0.05,
    "equal_frequency_ece_15_max": 0.05,
    "average_precision_lift_over_prevalence_min": 1.25,
    "roc_auc_min": 0.60,
    "recall_min": 0.55,
    "precision_min": 0.25,
    "f1_min": 0.36,
    "predicted_positive_rate_max": 0.60,
    "single_row_inference_p95_ms_strict_max": 25,
    "serialized_bundle_bytes_strict_max": 10_485_760,
    "lineage_schema_leakage_serialization_checks_pass": True,
}
EXPECTED_PAIRED_GATES = {
    "average_precision_v1_gte_v0": True,
    "roc_auc_v1_gte_v0": True,
    "brier_score_v1_strictly_below_v0": True,
    "log_loss_v1_strictly_below_v0": True,
}


class V1ProtocolError(ValueError):
    """Raised when the governed v1 protocol or lock drifts."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file without interpreting its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V1ProtocolError(f"{label} must be a mapping")
    return value


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise V1ProtocolError(f"{label} drifted from the precommitted value")


def _parse_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise V1ProtocolError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise V1ProtocolError(f"{label} must be an ISO date string") from error


def _validate_artifact_dependencies(protocol: dict[str, Any], repository_root: Path) -> None:
    dependencies = _mapping(protocol.get("dependencies"), "dependencies")
    _expect(dependencies, EXPECTED_DEPENDENCIES, "artifact dependencies")
    for label, specification in dependencies.items():
        if label == "base_git_sha":
            continue
        spec = _mapping(specification, f"dependencies.{label}")
        artifact_path = repository_root / str(spec["path"])
        if not artifact_path.is_file():
            raise V1ProtocolError(f"dependencies.{label} is missing")
        if sha256_file(artifact_path) != spec["sha256"]:
            raise V1ProtocolError(f"dependencies.{label} SHA256 mismatch")
        if "manifest_digest" in spec:
            try:
                manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise V1ProtocolError(f"dependencies.{label} is not valid JSON") from error
            if manifest.get("manifest_digest") != spec["manifest_digest"]:
                raise V1ProtocolError(f"dependencies.{label} manifest digest mismatch")


def _validate_features(protocol: dict[str, Any]) -> None:
    contract = _mapping(protocol.get("feature_contract"), "feature_contract")
    features = contract.get("model_features")
    _expect(features, list(PROCESSED_FEATURES), "model feature contract")
    if not isinstance(features, list):
        raise V1ProtocolError("model feature contract must be a list")
    try:
        validate_model_features(features)
    except FeatureLeakageError as error:
        raise V1ProtocolError("model feature contract violates the leakage guard") from error
    if "flight_date" in features:
        raise V1ProtocolError("flight_date is ordering-only")
    if "target" in features:
        raise V1ProtocolError("target is label-only")
    categorical = contract.get("categorical_features")
    _expect(categorical, EXPECTED_CATEGORICAL_FEATURES, "categorical feature contract")
    if not set(categorical).issubset(features):
        raise V1ProtocolError("categorical features must be a subset of model features")
    _expect(contract.get("ordering_only"), ["flight_date"], "ordering-only fields")
    _expect(contract.get("label_only"), ["target"], "label-only fields")
    _expect(contract.get("chronological_sort_before_fit"), True, "chronological sort rule")
    _expect(contract.get("postdeparture_features_permitted"), False, "postdeparture rule")


def _validate_temporal_design(protocol: dict[str, Any]) -> None:
    rolling = _mapping(protocol.get("rolling_origin"), "rolling_origin")
    _expect(rolling.get("fold_count"), 4, "rolling-origin fold count")
    folds = rolling.get("folds")
    _expect(folds, EXPECTED_FOLDS, "rolling-origin folds")
    if not isinstance(folds, list):
        raise V1ProtocolError("rolling-origin folds must be a list")
    development_limit = date(2026, 1, 1)
    previous_validation_end: date | None = None
    for index, fold_value in enumerate(folds):
        fold = _mapping(fold_value, f"fold {index + 1}")
        train_start = _parse_date(fold.get("train_start"), "train_start")
        train_end = _parse_date(fold.get("train_end_exclusive"), "train_end_exclusive")
        validation_start = _parse_date(fold.get("validation_start"), "validation_start")
        validation_end = _parse_date(
            fold.get("validation_end_exclusive"), "validation_end_exclusive"
        )
        if not train_start < train_end <= validation_start < validation_end:
            raise V1ProtocolError("rolling-origin training and validation periods overlap")
        if validation_end > development_limit:
            raise V1ProtocolError("rolling-origin development enters 2026")
        if previous_validation_end is not None and validation_start < previous_validation_end:
            raise V1ProtocolError("rolling-origin validation periods overlap")
        previous_validation_end = validation_end

    refit = _mapping(protocol.get("refit_calibration"), "refit_calibration")
    refit_period = _mapping(refit.get("refit_period"), "refit period")
    calibration = _mapping(refit.get("calibration_period"), "calibration period")
    selection = _mapping(
        _mapping(protocol.get("november_selection"), "november_selection").get("period"),
        "November selection period",
    )
    december = _mapping(
        _mapping(protocol.get("december_qualification"), "december_qualification").get("period"),
        "December qualification period",
    )
    ordered_dates = [
        _parse_date(refit_period.get("start"), "refit start"),
        _parse_date(refit_period.get("end_exclusive"), "refit end"),
        _parse_date(calibration.get("start"), "calibration start"),
        _parse_date(calibration.get("end_exclusive"), "calibration end"),
        _parse_date(selection.get("start"), "selection start"),
        _parse_date(selection.get("end_exclusive"), "selection end"),
        _parse_date(december.get("start"), "December start"),
        _parse_date(december.get("end_exclusive"), "December end"),
    ]
    if ordered_dates != [
        date(2025, 1, 1),
        date(2025, 11, 1),
        date(2025, 11, 1),
        date(2025, 11, 16),
        date(2025, 11, 16),
        date(2025, 12, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
    ]:
        raise V1ProtocolError("refit, calibration, selection, or December periods drifted")


def _validate_fixed_design(protocol: dict[str, Any]) -> None:
    state = _mapping(protocol.get("state"), "state")
    _expect(
        state,
        {
            "phase": "pretraining_protocol_locked",
            "protocol_precedes_training": True,
            "training_started": False,
            "results_exist": False,
            "wandb_runs_created": False,
            "fresh_final_accessed": False,
        },
        "protocol state",
    )
    _expect(protocol.get("incumbent"), EXPECTED_INCUMBENT, "incumbent v0 identity")

    historical = _mapping(protocol.get("historical_test"), "historical_test")
    _expect(historical.get("period"), HISTORICAL_TEST_PERIOD, "historical test period")
    _expect(historical.get("start"), "2026-01-01", "historical test start")
    _expect(historical.get("end_exclusive"), "2026-06-01", "historical test end")
    _expect(historical.get("consumed"), True, "historical test consumed marker")
    _expect(historical.get("source_path"), "data/processed/test.parquet", "test path")
    _expect(historical.get("access_prohibited"), True, "historical test access rule")
    _expect(
        historical.get("prohibited_uses"),
        [
            "training",
            "hyperparameter_selection",
            "calibration",
            "threshold_selection",
            "finalist_selection",
            "qualification",
            "feature_engineering_decisions",
            "acceptance_criteria_changes",
        ],
        "historical test prohibited uses",
    )

    family = _mapping(protocol.get("model_family"), "model_family")
    primary = _mapping(family.get("primary"), "primary model family")
    _expect(primary.get("estimator"), "CatBoostClassifier", "primary estimator")
    _expect(primary.get("package"), "catboost", "primary package")
    _expect(primary.get("version"), "1.2.10", "CatBoost version")
    _expect(primary.get("dependency_scope"), "modeling_optional", "dependency scope")
    _expect(primary.get("installed_in_protocol_pr"), False, "CatBoost installation rule")
    prophet = _mapping(
        _mapping(family.get("alternatives"), "model alternatives").get("prophet"), "Prophet"
    )
    _expect(prophet.get("status"), "considered_but_not_selected", "Prophet status")
    _expect(prophet.get("primary_selection_influence"), "none", "Prophet influence")

    development = _mapping(protocol.get("development_data"), "development_data")
    _expect(development.get("allowed_year"), 2025, "development year")
    _expect(
        development.get("allowed_sources"),
        ["data/processed/train.parquet", "data/processed/validation.parquet"],
        "development data sources",
    )
    _expect(
        development.get("forbidden_sources"),
        ["data/processed/test.parquet"],
        "forbidden data sources",
    )
    _expect(development.get("development_end_exclusive"), "2026-01-01", "development end")

    search = _mapping(protocol.get("catboost_search"), "catboost_search")
    _expect(search.get("search_type"), "fixed_grid", "CatBoost search type")
    _expect(search.get("candidate_count"), 4, "CatBoost candidate count")
    _expect(search.get("common_parameters"), EXPECTED_COMMON_PARAMETERS, "common parameters")
    _expect(search.get("candidates"), EXPECTED_CANDIDATES, "CatBoost candidate grid")
    _expect(
        search.get("prohibited_searches"),
        ["bayesian_optimization", "optuna", "random_search", "manual_out_of_grid_tuning"],
        "prohibited search methods",
    )

    rolling = _mapping(protocol.get("rolling_origin"), "rolling_origin")
    _expect(rolling.get("validation_used_for_early_stopping"), False, "early stopping rule")
    _expect(rolling.get("catboost_finalists_advanced"), 2, "CatBoost finalist count")
    _expect(rolling.get("control_consumes_finalist_slot"), False, "control finalist rule")

    refit = _mapping(protocol.get("refit_calibration"), "refit_calibration")
    _expect(refit.get("variants"), ["none", "sigmoid", "isotonic"], "calibration variants")
    _expect(refit.get("expected_finalist_variant_count"), 6, "finalist variant count")
    _expect(refit.get("base_frozen_before_calibration"), True, "base freeze rule")
    _expect(refit.get("calibration_refits_base_estimator"), False, "calibration refit rule")
    _expect(refit.get("none_uses_calibration_labels"), False, "raw variant rule")

    november = _mapping(protocol.get("november_selection"), "november_selection")
    _expect(november.get("threshold_objective"), EXPECTED_THRESHOLD_OBJECTIVE, "threshold rules")
    _expect(november.get("acceptance_gates"), EXPECTED_NOVEMBER_GATES, "November gates")
    _expect(november.get("every_gate_mandatory"), True, "November mandatory-gate rule")

    december = _mapping(protocol.get("december_qualification"), "december_qualification")
    _expect(
        december.get("characterization"),
        "retrospective_temporal_qualification_holdback",
        "December characterization",
    )
    _expect(december.get("evaluation_count"), 1, "December evaluation count")
    _expect(december.get("gates"), EXPECTED_QUALIFICATION_GATES, "December gates")

    fresh = _mapping(protocol.get("fresh_final"), "fresh_final")
    _expect(fresh.get("named_month_precommitted"), False, "fresh-final named-month rule")
    _expect(
        fresh.get("selection_rule"),
        {
            "source": "DOT_BTS_Reporting_Carrier_On_Time_Performance",
            "first_complete_month": True,
            "flight_date_strictly_after": "2026-05-31",
            "unused_by_existing_project_decisions": True,
            "unopened_for_v1_development_decisions": True,
            "availability_after_protocol_lock": True,
            "archive_identity_and_sha256_recorded_before_label_evaluation": True,
        },
        "fresh-final selection rule",
    )
    _expect(fresh.get("monthly_sample_cap"), None, "fresh-final sample cap")
    _expect(fresh.get("full_eligible_month_required"), True, "full-month rule")
    _expect(fresh.get("evaluation_count"), 1, "fresh-final evaluation count")
    _expect(fresh.get("absolute_v1_gates"), EXPECTED_QUALIFICATION_GATES, "fresh-final gates")
    _expect(fresh.get("paired_incumbent_gates"), EXPECTED_PAIRED_GATES, "paired gates")

    bootstrap = _mapping(protocol.get("uncertainty"), "uncertainty")
    _expect(
        bootstrap,
        {
            "method": "paired_day_block_bootstrap",
            "descriptive_only": True,
            "replicates": 500,
            "random_seed": 42,
            "resampling_unit": "flight_date",
            "confidence_interval": "percentile_95",
            "metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
            "contrast": "v1_minus_v0",
            "overrides_pass_fail_gates": False,
        },
        "bootstrap configuration",
    )

    future = _mapping(protocol.get("future_registry"), "future_registry")
    _expect(future.get("action_in_protocol_pr"), "none", "Registry action rule")
    _expect(future.get("overwrite_v0_prohibited"), True, "v0 overwrite rule")
    _expect(future.get("mutate_v0_bytes_prohibited"), True, "v0 byte mutation rule")
    _expect(
        future.get("production_alias_mutation_before_fresh_final_and_review_prohibited"),
        True,
        "production alias rule",
    )


def validate_v1_protocol(protocol: dict[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """Validate every precommitted v1 design boundary without running an experiment."""

    expected_sections = {
        "schema_version",
        "protocol_id",
        "state",
        "dependencies",
        "incumbent",
        "historical_test",
        "model_family",
        "feature_contract",
        "development_data",
        "rolling_origin",
        "catboost_search",
        "control",
        "refit_calibration",
        "november_selection",
        "december_qualification",
        "fresh_final",
        "uncertainty",
        "future_registry",
    }
    _expect(set(protocol), expected_sections, "top-level protocol sections")
    _expect(protocol.get("schema_version"), PROTOCOL_SCHEMA_VERSION, "protocol schema version")
    _expect(protocol.get("protocol_id"), PROTOCOL_ID, "protocol ID")
    _validate_artifact_dependencies(protocol, repository_root)
    _validate_features(protocol)
    _validate_temporal_design(protocol)
    _validate_fixed_design(protocol)
    return protocol


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = yaml.safe_load(raw) if path.suffix in {".yaml", ".yml"} else json.loads(raw)
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise V1ProtocolError(f"unable to read {label}") from error
    return _mapping(value, label)


def validate_protocol_lock(lock: dict[str, Any], *, protocol_sha256: str) -> dict[str, Any]:
    """Validate the immutable record proving that governance preceded results."""

    expected = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_path": "configs/v1_experiment_protocol.yaml",
        "protocol_sha256": protocol_sha256,
        "base_git_sha": BASE_GIT_SHA,
        "incumbent_registry_version": "v0",
        "incumbent_registry_digest": EXPECTED_INCUMBENT["registry_digest"],
        "incumbent_bundle_digest": EXPECTED_INCUMBENT["bundle_sha256"],
        "incumbent_threshold": EXPECTED_INCUMBENT["threshold"],
        "historical_test_period": HISTORICAL_TEST_PERIOD,
        "historical_test_consumed": True,
        "training_started": False,
        "wandb_runs_created": False,
        "fresh_final_accessed": False,
        "artifact_dependencies": {
            key: value for key, value in EXPECTED_DEPENDENCIES.items() if key != "base_git_sha"
        },
    }
    _expect(lock, expected, "protocol lock")
    return lock


def load_and_validate_v1_protocol(
    protocol_path: Path = Path("configs/v1_experiment_protocol.yaml"),
    *,
    lock_path: Path = Path("experiments/v1/protocol_lock.json"),
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and validate the canonical protocol and byte-level lock."""

    protocol_sha256 = sha256_file(protocol_path)
    if protocol_sha256 != CANONICAL_PROTOCOL_SHA256:
        raise V1ProtocolError("protocol SHA256 drifted from the validator's precommitted value")
    protocol = validate_v1_protocol(
        _load_mapping(protocol_path, "v1 protocol"),
        repository_root=repository_root or protocol_path.resolve().parents[1],
    )
    lock = validate_protocol_lock(
        _load_mapping(lock_path, "v1 protocol lock"), protocol_sha256=protocol_sha256
    )
    return protocol, lock, protocol_sha256
