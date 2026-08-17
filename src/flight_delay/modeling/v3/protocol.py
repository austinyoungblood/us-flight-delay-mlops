"""Strict offline validation for the immutable governed v3 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from flight_delay.data.prepare import PROCESSED_FEATURES
from flight_delay.features.leakage import FeatureLeakageError, validate_model_features
from flight_delay.modeling.v2.protocol import HISTORICAL_FEATURES as V2_HISTORICAL_FEATURES

PROTOCOL_ID = "us-flight-delay-v3-seasonal-temporal-generalization-v1"
PROTOCOL_SHA256 = "061be599fd84a4ddbf06229c300fe4670272d176b22899f1515332923376ecff"
PROTOCOL_LOCK_SHA256 = "ef0c15137032af3259df598cf196c510345c15cbd532d2ea85638145e329bcef"
BASE_GIT_SHA = "6966562dcc2a7959f27e662e97cfeec8a4aa43a6"
V2_PROTOCOL_SHA256 = "8e57b0f63656003c9981b3b5e44623e0b7c556f6e0c7222352ac38dd5119420a"

DETERMINISTIC_SEASONAL_FEATURES: tuple[str, ...] = (
    "day_of_year_sin",
    "day_of_year_cos",
    "days_to_thanksgiving",
    "is_thanksgiving_window",
    "days_to_christmas",
    "is_christmas_window",
)
SEASONAL_HISTORICAL_FEATURES: tuple[str, ...] = (
    "prior_same_calendar_month_global_delay_rate",
    "prior_same_calendar_month_carrier_delay_rate",
    "prior_same_calendar_month_origin_delay_rate",
    "prior_same_calendar_month_destination_delay_rate",
    "prior_same_calendar_month_route_delay_rate",
)
V3_HISTORICAL_FEATURES: tuple[str, ...] = (
    *V2_HISTORICAL_FEATURES,
    *SEASONAL_HISTORICAL_FEATURES,
)
V3_SCHEDULE_FEATURES: tuple[str, ...] = (
    *PROCESSED_FEATURES,
    *DETERMINISTIC_SEASONAL_FEATURES,
)
V3_FEATURES: tuple[str, ...] = (*V3_SCHEDULE_FEATURES, *V3_HISTORICAL_FEATURES)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "route",
    "Month",
    "DayOfWeek",
    "scheduled_departure_hour",
    "scheduled_arrival_hour",
)
STRING_CATEGORICAL_FEATURES: tuple[str, ...] = ("Reporting_Airline", "Origin", "Dest", "route")
INTEGER_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Month",
    "DayOfWeek",
    "scheduled_departure_hour",
    "scheduled_arrival_hour",
)

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

WEIGHT_POLICY_IDS: tuple[str, ...] = ("UNIFORM", "EXPONENTIAL_120D")
WEIGHT_POLICY_SUFFIX: dict[str, str] = {"UNIFORM": "UNIFORM", "EXPONENTIAL_120D": "EXP120"}
EXPONENTIAL_HALF_LIFE_DAYS = 120

BASE_CONFIGURATION_IDS: tuple[str, ...] = ("LGBM12", "LGBM10", "CB07", "CB04")
CANDIDATE_IDENTITY_IDS: tuple[str, ...] = (
    "LGBM12-UNIFORM",
    "LGBM12-EXP120",
    "LGBM10-UNIFORM",
    "LGBM10-EXP120",
    "CB07-UNIFORM",
    "CB07-EXP120",
    "CB04-UNIFORM",
    "CB04-EXP120",
)
ENSEMBLE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("ENS25", 0.25),
    ("ENS50", 0.50),
    ("ENS75", 0.75),
)
CALIBRATION_VARIANTS: tuple[str, ...] = ("none", "sigmoid", "isotonic")

CANDIDATE_RANKING_ORDER: tuple[str, ...] = (
    "worst_fold_operating_precision_desc",
    "fold_4_november_operating_precision_desc",
    "mean_fold_2_through_fold_4_operating_precision_desc",
    "mean_all_fold_operating_precision_desc",
    "mean_average_precision_desc",
    "mean_roc_auc_desc",
    "mean_log_loss_asc",
    "mean_brier_score_asc",
    "candidate_id_lexical_asc",
)

THANKSGIVING_WINDOW_RANGE: tuple[int, int] = (-4, 2)
CHRISTMAS_WINDOW_RANGE: tuple[int, int] = (-10, 4)
DAY_DISTANCE_CLIP: tuple[int, int] = (-30, 30)

FOLD_IDS: tuple[str, ...] = ("FOLD_1", "FOLD_2", "FOLD_3", "FOLD_4")


class V3ProtocolError(ValueError):
    """Raised when the governed v3 protocol or lock has drifted."""


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
        raise V3ProtocolError(f"{label} must be a mapping")
    return value


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise V3ProtocolError(f"{label} drifted from the frozen protocol")


def _validate_dependency_tree(value: object, root: Path, label: str = "dependencies") -> None:
    mapping = _mapping(value, label)
    if "path" in mapping or "sha256" in mapping:
        if not {"path", "sha256"} <= set(mapping):
            raise V3ProtocolError(f"{label} must contain both path and sha256")
        path = root / str(mapping["path"])
        if not path.is_file() or sha256_file(path) != mapping["sha256"]:
            raise V3ProtocolError(f"{label} artifact hash mismatch")
        if "manifest_digest" in mapping:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise V3ProtocolError(f"{label} manifest is unreadable") from error
            _expect(payload.get("manifest_digest"), mapping["manifest_digest"], label)
        return
    for key, child in mapping.items():
        if key == "base_git_sha":
            continue
        if isinstance(child, dict):
            _validate_dependency_tree(child, root, f"{label}.{key}")


def _validate_feature_contract(protocol: dict[str, Any]) -> None:
    contract = _mapping(protocol.get("feature_contract"), "feature contract")
    _expect(contract.get("base_features"), list(PROCESSED_FEATURES), "base features")
    _expect(
        contract.get("deterministic_seasonal_features"),
        list(DETERMINISTIC_SEASONAL_FEATURES),
        "deterministic seasonal features",
    )
    _expect(
        contract.get("historical_features"), list(V2_HISTORICAL_FEATURES), "v2 historical features"
    )
    _expect(
        contract.get("seasonal_historical_features"),
        list(SEASONAL_HISTORICAL_FEATURES),
        "seasonal historical features",
    )
    _expect(contract.get("total_feature_count"), 48, "feature count")
    _expect(contract.get("retained_v2_feature_count"), 37, "retained v2 feature count")
    _expect(contract.get("categorical_features"), list(CATEGORICAL_FEATURES), "categoricals")
    _expect(contract.get("native_categorical_count"), 8, "native categorical count")
    _expect(contract.get("postdeparture_features_permitted"), False, "post-departure rule")
    if len(V3_FEATURES) != 48 or len(set(V3_FEATURES)) != 48:
        raise V3ProtocolError("v3 feature schema must contain 48 unique features")
    retained = (*PROCESSED_FEATURES, *V2_HISTORICAL_FEATURES)
    if not set(retained) <= set(V3_FEATURES) or len(retained) != 37:
        raise V3ProtocolError("v3 must retain all 37 v2 features")
    if not set(CATEGORICAL_FEATURES) <= set(V3_FEATURES):
        raise V3ProtocolError("every native categorical must belong to the v3 schema")
    try:
        validate_model_features(V3_FEATURES)
    except FeatureLeakageError as error:
        raise V3ProtocolError("v3 feature contract violates the central leakage guard") from error


def _validate_seasonal_semantics(protocol: dict[str, Any]) -> None:
    semantics = _mapping(
        protocol.get("deterministic_seasonal_semantics"), "deterministic seasonal semantics"
    )
    _expect(semantics.get("computed_solely_from_scheduled_flight_date"), True, "schedule-only rule")
    _expect(semantics.get("no_target_or_label_input"), True, "seasonal label rule")
    thanksgiving = _mapping(semantics.get("thanksgiving"), "Thanksgiving")
    _expect(thanksgiving.get("definition"), "fourth_Thursday_of_November", "Thanksgiving rule")
    _expect(thanksgiving.get("anchor_tie_break"), "earlier_anchor_date", "Thanksgiving tie break")
    _expect(
        _mapping(semantics.get("christmas"), "Christmas").get("definition"),
        "december_25",
        "Christmas rule",
    )
    bounded = _mapping(semantics.get("bounded_distance"), "bounded distance")
    _expect(
        (bounded.get("clip_lower"), bounded.get("clip_upper")), DAY_DISTANCE_CLIP, "distance clip"
    )
    windows = _mapping(semantics.get("windows"), "seasonal windows")
    _expect(
        _mapping(windows.get("is_thanksgiving_window"), "Thanksgiving window").get(
            "raw_signed_delta_inclusive_range"
        ),
        list(THANKSGIVING_WINDOW_RANGE),
        "Thanksgiving window range",
    )
    _expect(
        _mapping(windows.get("is_christmas_window"), "Christmas window").get(
            "raw_signed_delta_inclusive_range"
        ),
        list(CHRISTMAS_WINDOW_RANGE),
        "Christmas window range",
    )

    state = _mapping(protocol.get("historical_feature_state"), "historical state")
    _expect(state.get("schema"), "flight-delay-historical-state-v3", "state schema")
    _expect(state.get("prior_strength"), 50, "prior strength")
    _expect(state.get("same_month_labels_permitted"), False, "same-month rule")
    _expect(state.get("future_labels_permitted"), False, "future-label rule")
    _expect(state.get("full_eligible_prior_history_required"), True, "full-history rule")
    _expect(state.get("historical_state_sampling_permitted"), False, "state sampling rule")
    seasonal = _mapping(state.get("seasonal_prior_year_rule"), "seasonal prior-year rule")
    _expect(seasonal.get("prior_or_previous_occurrences_only"), True, "prior-occurrence rule")
    _expect(
        seasonal.get("same_month_same_year_contribution_prohibited"),
        True,
        "same-year seasonal rule",
    )
    _expect(
        seasonal.get("november_2025_targets_contribute_to_november_2025_features"),
        False,
        "November self-contribution rule",
    )
    _expect(
        seasonal.get("november_2025_features_may_use_november_2024"),
        True,
        "November prior-year rule",
    )
    _expect(
        _mapping(state.get("seasonal_tables"), "seasonal tables").keys()
        == {
            "same_calendar_month_global",
            "same_calendar_month_carrier",
            "same_calendar_month_origin",
            "same_calendar_month_destination",
            "same_calendar_month_route",
        },
        True,
        "seasonal table set",
    )
    _expect(
        _mapping(state.get("november_and_december_state"), "frozen state").get("as_of_inclusive"),
        "2025-10-31",
        "November/December state cutoff",
    )


def _validate_weight_policies(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    policies = _mapping(protocol.get("weight_policies"), "weight policies")
    _expect(policies.get("precommitted_count"), 2, "weight policy count")
    _expect(policies.get("evaluation_rows_weighted"), False, "evaluation weighting rule")
    _expect(policies.get("calibration_rows_weighted"), False, "calibration weighting rule")
    _expect(policies.get("selection_rows_weighted"), False, "selection weighting rule")
    _expect(
        policies.get("weight_policy_is_part_of_candidate_identity"), True, "weight identity rule"
    )
    _expect(policies.get("backend_excluded_from_candidate_identity"), True, "backend identity rule")
    rows = policies.get("policies")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != list(WEIGHT_POLICY_IDS):
        raise V3ProtocolError("weight policy identities drifted")
    exponential = rows[1]
    _expect(exponential.get("half_life_days"), EXPONENTIAL_HALF_LIFE_DAYS, "half life")
    _expect(exponential.get("formula"), "0.5 ** (age_days / 120)", "weight formula")
    _expect(exponential.get("normalization"), "mean_1_within_fit_partition", "weight normalization")
    _expect(exponential.get("age_days_measured_backward_from_fit_cutoff"), True, "age direction")
    _expect(lock.get("weight_policies"), rows, "locked weight policies")


def _validate_candidates(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    carried = _mapping(protocol.get("carried_forward_configurations"), "carried configurations")
    _expect(carried.get("hyperparameters_unchanged_from_v2"), True, "v2 hyperparameter reuse")
    _expect(carried.get("new_hyperparameter_search_performed"), False, "no-new-search rule")
    _expect(carried.get("source_protocol_sha256"), V2_PROTOCOL_SHA256, "v2 source protocol digest")
    _expect(
        carried.get("lightgbm_common_parameters"),
        LIGHTGBM_COMMON_PARAMETERS,
        "LightGBM common parameters",
    )
    _expect(
        carried.get("catboost_common_parameters"),
        CATBOOST_COMMON_PARAMETERS,
        "CatBoost common parameters",
    )
    bases = carried.get("base_configurations")
    if not isinstance(bases, list) or [row.get("id") for row in bases] != list(
        BASE_CONFIGURATION_IDS
    ):
        raise V3ProtocolError("carried-forward base configuration identities drifted")
    _expect(lock.get("base_configurations"), bases, "locked base configurations")
    _expect(
        lock.get("base_configurations_sha256"),
        canonical_sha256(bases),
        "base configuration digest",
    )
    for row in bases:
        if "task_type" in row or "devices" in row or "n_jobs" in row or "weight_policy" in row:
            raise V3ProtocolError("backend or weight policy cannot enter hyperparameter identity")

    identities = _mapping(protocol.get("candidate_identities"), "candidate identities")
    _expect(identities.get("total"), 8, "candidate identity count")
    rows = identities.get("identities")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != list(
        CANDIDATE_IDENTITY_IDS
    ):
        raise V3ProtocolError("v3 candidate identities drifted")
    known = set(BASE_CONFIGURATION_IDS)
    for row in rows:
        if row.get("base_configuration") not in known:
            raise V3ProtocolError("candidate identity references an unknown base configuration")
        if row.get("weight_policy") not in WEIGHT_POLICY_IDS:
            raise V3ProtocolError("candidate identity references an unknown weight policy")
        suffix = WEIGHT_POLICY_SUFFIX[str(row["weight_policy"])]
        if str(row["id"]) != f"{row['base_configuration']}-{suffix}":
            raise V3ProtocolError("candidate identity ID does not match its components")
    _expect(lock.get("candidate_identities"), rows, "locked candidate identities")
    _expect(
        lock.get("candidate_identities_sha256"), canonical_sha256(rows), "candidate identity digest"
    )


def _validate_ensembles(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    ensembles = _mapping(protocol.get("ensembles"), "ensembles")
    _expect(ensembles.get("precommitted"), True, "ensemble precommitment")
    _expect(ensembles.get("post_result_definitions_prohibited"), True, "post-result ensemble rule")
    _expect(ensembles.get("additional_base_model_fits_required"), 0, "ensemble refit rule")
    _expect(
        ensembles.get("inputs"), "two_uncalibrated_authoritative_cpu_base_scores", "ensemble inputs"
    )
    weights = ensembles.get("weights")
    expected = [
        {"id": name, "lightgbm_weight": value, "catboost_weight": round(1.0 - value, 2)}
        for name, value in ENSEMBLE_WEIGHTS
    ]
    _expect(weights, expected, "ensemble weights")
    _expect(ensembles.get("variants"), list(CALIBRATION_VARIANTS), "ensemble calibration variants")
    _expect(lock.get("ensembles"), weights, "locked ensemble weights")
    calibration = _mapping(protocol.get("calibration"), "calibration")
    _expect(calibration.get("variants"), list(CALIBRATION_VARIANTS), "calibration variants")
    _expect(calibration.get("bases"), 2, "calibration base count")
    _expect(calibration.get("base_frozen_before_calibration"), True, "frozen base rule")
    _expect(
        calibration.get("november_feature_state_frozen_at"), "2025-10-31", "November state freeze"
    )
    _expect(_mapping(protocol.get("finalists"), "finalists").get("total"), 15, "finalist count")


def _validate_stages(protocol: dict[str, Any]) -> None:
    folds = _mapping(protocol.get("rolling_origin"), "rolling origin").get("folds")
    if not isinstance(folds, list) or [row.get("id") for row in folds] != list(FOLD_IDS):
        raise V3ProtocolError("rolling fold identities drifted")
    _expect(
        [row.get("evaluation_state_as_of") for row in folds],
        ["2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31"],
        "rolling feature-state cutoffs",
    )
    _expect(
        [row.get("fit_cutoff_inclusive") for row in folds],
        ["2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31"],
        "rolling fit cutoffs",
    )
    for row in folds:
        _expect(row.get("fit_start"), "2024-02-01", "fold fit start")
        if row.get("fit_end_exclusive") != row.get("evaluation_start"):
            raise V3ProtocolError("fold fit and evaluation periods must be contiguous")

    metric = _mapping(protocol.get("search_metric"), "search metric")
    _expect(
        metric.get("threshold_constraints"),
        {"recall_min": 0.60, "predicted_positive_rate_max": 0.50},
        "search threshold constraints",
    )
    _expect(metric.get("candidate_ranking"), list(CANDIDATE_RANKING_ORDER), "candidate ranking")

    advancement = _mapping(protocol.get("advancement"), "advancement")
    screening = _mapping(advancement.get("screening"), "screening")
    _expect(screening.get("total_identities"), 8, "screening identity count")
    _expect(screening.get("top_per_family_to_cpu_confirmation"), 2, "screening advancement")
    confirmation = _mapping(advancement.get("cpu_confirmation"), "confirmation")
    _expect(confirmation.get("total_identities"), 4, "confirmation identity count")
    _expect(confirmation.get("top_per_family_to_full_refit"), 1, "confirmation advancement")
    _expect(confirmation.get("authoritative_for_advancement"), True, "CPU authority")
    refit = _mapping(advancement.get("full_refit"), "full refit")
    _expect(refit.get("total_bases"), 2, "full refit base count")
    _expect(refit.get("full_eligible_rows"), True, "full refit row rule")
    _expect(refit.get("backend"), "CPU", "full refit backend")
    _expect(refit.get("end_exclusive"), "2025-11-01", "full refit end")

    sampling = _mapping(protocol.get("sampling"), "sampling")
    _expect(sampling.get("search_rows_per_month_max"), 50000, "search sampling cap")
    _expect(sampling.get("sample_seed"), 42, "sample seed")
    _expect(sampling.get("historical_lookup_inputs"), "full_eligible_prior_history", "state inputs")
    _expect(sampling.get("final_refit_inputs"), "full_eligible_model_rows", "refit inputs")

    policy = _mapping(protocol.get("execution_policy"), "execution policy")
    _expect(policy.get("sequential_candidate_execution"), True, "sequential execution")
    _expect(
        _mapping(policy.get("catboost_screening"), "CatBoost GPU").get("task_type"),
        "GPU",
        "CatBoost screening backend",
    )
    _expect(
        _mapping(policy.get("catboost_screening"), "CatBoost GPU").get("concurrent_gpu_fits"),
        False,
        "sequential GPU rule",
    )
    _expect(
        _mapping(policy.get("lightgbm_screening"), "LightGBM CPU").get("backend"),
        "CPU",
        "LightGBM screening backend",
    )
    _expect(policy.get("backend_excluded_from_candidate_identity"), True, "backend identity rule")
    _expect(policy.get("stage_runtime_logging_required"), True, "runtime logging rule")
    _expect(policy.get("dry_run_estimate_required_before_apply"), True, "dry-run rule")


def _validate_governance(protocol: dict[str, Any]) -> None:
    development = _mapping(protocol.get("development_data"), "development data")
    _expect(development.get("start"), "2024-01-01", "development start")
    _expect(development.get("end_exclusive"), "2025-12-01", "development end")
    _expect(
        _mapping(development.get("burn_in"), "burn-in").get("contributes_model_rows"),
        False,
        "January 2024 burn-in rule",
    )
    _expect(
        _mapping(development.get("model_period"), "model period").get("start"),
        "2024-02-01",
        "model period start",
    )
    _expect(
        development.get("december_2025_decoded_during_development"), False, "December decode rule"
    )
    if "data/processed/test.parquet" not in list(development.get("forbidden_sources") or []):
        raise V3ProtocolError("the sealed 2026 test split must remain a forbidden source")

    prohibited = _mapping(protocol.get("prohibited_periods"), "prohibited periods")
    _expect(
        _mapping(prohibited.get("december_2025"), "December").get("development_access_prohibited"),
        True,
        "December access rule",
    )
    historical = _mapping(prohibited.get("historical_test_2026"), "2026 test")
    _expect(historical.get("access_prohibited"), True, "2026 access rule")
    _expect(historical.get("excluded_from_v3_source_manifest"), True, "2026 manifest exclusion")

    november = _mapping(protocol.get("november_selection"), "November selection")
    gates = _mapping(november.get("acceptance_gates"), "November gates")
    _expect(gates.get("gates_relaxed_from_v2"), False, "gate relaxation rule")
    _expect(
        gates.get("operating_point"),
        {
            "recall_min": 0.60,
            "precision_min": 0.30,
            "f1_min": 0.38,
            "predicted_positive_rate_max": 0.50,
        },
        "November operating gates",
    )
    _expect(
        _mapping(november.get("threshold_objective"), "threshold objective").get("eligibility"),
        {"recall_min": 0.60, "precision_min": 0.30, "predicted_positive_rate_max": 0.50},
        "November eligibility",
    )
    _expect(november.get("every_gate_mandatory"), True, "mandatory gate rule")
    _expect(
        november.get("zero_pass_outcome"), "governed_stop_retain_production_v0", "governed stop"
    )

    december = _mapping(protocol.get("december_qualification"), "December qualification")
    _expect(december.get("development_may_open"), False, "December development rule")
    _expect(december.get("separate_cli_required"), True, "December CLI rule")
    _expect(december.get("evaluation_count"), 1, "December evaluation count")
    for action in ("refitting", "recalibration", "threshold_change", "candidate_switching"):
        if action not in list(december.get("prohibited_actions") or []):
            raise V3ProtocolError(f"December must prohibit {action}")

    fresh = _mapping(protocol.get("fresh_final"), "fresh final")
    _expect(fresh.get("earliest_eligible_date_strictly_after"), "2026-05-31", "fresh final bound")
    _expect(fresh.get("access_count_during_protocol_and_implementation"), 0, "fresh final access")

    release = _mapping(protocol.get("release"), "release")
    _expect(release.get("production_version_after_v3_development"), "v0", "production version")
    _expect(release.get("production_alias_mutation"), False, "production alias rule")
    _expect(release.get("runtime_images_unchanged"), True, "runtime image rule")
    _expect(
        release.get("runtime_images_gain_v3_modeling_dependencies"),
        False,
        "runtime dependency rule",
    )

    expansion = _mapping(protocol.get("data_expansion"), "data expansion")
    _expect(expansion.get("added_calendar_year"), 2024, "added year")
    _expect(expansion.get("v0_v1_v2_manifests_mutated"), False, "manifest immutability")
    _expect(
        expansion.get("reused_2025_archives_byte_identical_to_v0"), True, "2025 archive identity"
    )
    _expect(expansion.get("monthly_sample_cap"), None, "v3 preparation sample cap")


def _validate_semantics(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    _expect(protocol.get("schema_version"), 1, "schema version")
    _expect(protocol.get("protocol_id"), PROTOCOL_ID, "protocol ID")
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
    motivation = _mapping(protocol.get("motivation"), "motivation")
    _expect(motivation.get("gates_weakened"), False, "gate weakening rule")
    _expect(motivation.get("broad_hyperparameter_campaign"), False, "hyperparameter campaign rule")
    _expect(motivation.get("additional_model_families_added"), False, "model family rule")
    _validate_feature_contract(protocol)
    _validate_seasonal_semantics(protocol)
    _validate_weight_policies(protocol, lock)
    _validate_candidates(protocol, lock)
    _validate_ensembles(protocol, lock)
    _validate_stages(protocol)
    _validate_governance(protocol)


def load_and_validate_v3_protocol(
    protocol_path: Path,
    *,
    lock_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and validate the byte-locked v3 protocol without model imports or network access."""

    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = yaml.safe_load(protocol_bytes)
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise V3ProtocolError("v3 protocol or lock is unreadable") from error
    protocol = _mapping(protocol, "protocol")
    lock = _mapping(lock, "protocol lock")
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    _expect(protocol_sha, PROTOCOL_SHA256, "canonical protocol SHA256")
    _expect(hashlib.sha256(lock_bytes).hexdigest(), PROTOCOL_LOCK_SHA256, "protocol-lock SHA256")
    _expect(lock.get("protocol_sha256"), protocol_sha, "lock protocol SHA256")
    _expect(lock.get("protocol_id"), PROTOCOL_ID, "lock protocol ID")
    _expect(lock.get("base_git_sha"), BASE_GIT_SHA, "lock base Git SHA")
    _expect(lock.get("v2_protocol_sha256"), V2_PROTOCOL_SHA256, "lock v2 protocol SHA256")
    _expect(lock.get("training_started"), False, "lock training marker")
    _expect(lock.get("results_exist"), False, "lock results marker")
    _expect(lock.get("wandb_runs_created"), False, "lock W&B marker")
    _expect(lock.get("december_opened"), False, "lock December marker")
    _expect(lock.get("historical_test_accessed"), False, "lock historical-test marker")
    _expect(lock.get("production_registry_version"), "v0", "lock production version")
    _expect(lock.get("feature_schema"), list(V3_FEATURES), "lock feature schema")
    _expect(lock.get("feature_schema_sha256"), canonical_sha256(list(V3_FEATURES)), "lock schema")
    _validate_semantics(protocol, lock)
    _validate_dependency_tree(protocol.get("dependencies"), repository_root.resolve())
    return protocol, lock, protocol_sha
