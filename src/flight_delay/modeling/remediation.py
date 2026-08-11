"""Fixed Brief 04 model matrix, chronological folds, and deterministic ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.data.preprocessing import DataQualityError
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.candidates import MonthCyclicalAugmenter

CYCLICAL_NO_ROUTE_FEATURES: tuple[str, ...] = (
    *CANDIDATE_A_FEATURES,
    "scheduled_departure_sin",
    "scheduled_departure_cos",
    "scheduled_arrival_sin",
    "scheduled_arrival_cos",
)
CYCLICAL_MODEL_FEATURES: tuple[str, ...] = (
    *CYCLICAL_NO_ROUTE_FEATURES,
    "month_sin",
    "month_cos",
)
CATEGORICAL_FEATURES = ("Reporting_Airline", "Origin", "Dest")
EXPECTED_MATRIX: dict[str, dict[str, Any]] = {
    "R0": {
        "estimator": "sgd",
        "feature_set": "candidate_a",
        "alpha": 0.0001,
        "class_weight": "balanced",
        "average": False,
        "max_iter": 1000,
        "tol": 0.001,
    },
    "R1": {
        "estimator": "sgd",
        "feature_set": "candidate_a",
        "alpha": 0.00001,
        "class_weight": None,
        "average": True,
        "max_iter": 1000,
        "tol": 0.001,
    },
    "R2": {
        "estimator": "sgd",
        "feature_set": "candidate_a",
        "alpha": 0.0001,
        "class_weight": None,
        "average": True,
        "max_iter": 1000,
        "tol": 0.001,
    },
    "R3": {
        "estimator": "sgd",
        "feature_set": "cyclical_no_route",
        "alpha": 0.00001,
        "class_weight": None,
        "average": True,
        "max_iter": 1000,
        "tol": 0.001,
    },
    "R4": {
        "estimator": "logistic_regression",
        "feature_set": "candidate_a",
        "solver": "saga",
        "penalty": "l2",
        "C": 0.1,
        "class_weight": None,
        "max_iter": 250,
        "tol": 0.001,
    },
    "R5": {
        "estimator": "logistic_regression",
        "feature_set": "candidate_a",
        "solver": "saga",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,
        "max_iter": 250,
        "tol": 0.001,
    },
}


@dataclass(frozen=True)
class RollingFold:
    fold: int
    fit: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class RemediationPartitions:
    rolling_folds: tuple[RollingFold, ...]
    final_fit: pd.DataFrame
    calibration: pd.DataFrame
    selection: pd.DataFrame


def _period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["flight_date"], errors="coerce").dt.normalize()
    selected = frame.loc[dates.between(start, end, inclusive="left")].copy()
    return selected.sort_values("flight_date", kind="stable")


def partition_remediation_data(
    train: pd.DataFrame, november: pd.DataFrame
) -> RemediationPartitions:
    """Build Brief 04 development windows without accepting December or test inputs."""

    for name, frame in (("train", train), ("november", november)):
        if (
            "flight_date" not in frame
            or pd.to_datetime(frame["flight_date"], errors="coerce").isna().any()
        ):
            raise DataQualityError(f"{name} has missing or invalid flight_date values")
    folds = rolling_origin_folds(train)
    final_fit = _period(train, "2025-01-01", "2025-11-01")
    calibration = _period(november, "2025-11-01", "2025-11-16")
    selection = _period(november, "2025-11-16", "2025-12-01")
    frames = [
        *(item for fold in folds for item in (fold.fit, fold.evaluation)),
        final_fit,
        calibration,
        selection,
    ]
    if any(frame.empty for frame in frames):
        raise DataQualityError("a Brief 04 development partition is empty")
    if set(calibration.index) & set(selection.index):
        raise DataQualityError("November calibration and selection partitions overlap")
    if any(not frame["flight_date"].is_monotonic_increasing for frame in frames):
        raise DataQualityError("Brief 04 partitions must be stably sorted")
    return RemediationPartitions(folds, final_fit, calibration, selection)


def rolling_origin_folds(train: pd.DataFrame) -> tuple[RollingFold, ...]:
    """Build the four fixed rolling folds from January-October data only."""

    if (
        "flight_date" not in train
        or pd.to_datetime(train["flight_date"], errors="coerce").isna().any()
    ):
        raise DataQualityError("train has missing or invalid flight_date values")
    fold_periods = (
        ("2025-01-01", "2025-07-01", "2025-07-01", "2025-08-01"),
        ("2025-01-01", "2025-08-01", "2025-08-01", "2025-09-01"),
        ("2025-01-01", "2025-09-01", "2025-09-01", "2025-10-01"),
        ("2025-01-01", "2025-10-01", "2025-10-01", "2025-11-01"),
    )
    folds = tuple(
        RollingFold(
            number,
            _period(train, fit_start, fit_end),
            _period(train, evaluation_start, evaluation_end),
        )
        for number, (fit_start, fit_end, evaluation_start, evaluation_end) in enumerate(
            fold_periods, start=1
        )
    )
    if any(fold.fit.empty or fold.evaluation.empty for fold in folds):
        raise DataQualityError("a Brief 04 rolling-origin partition is empty")
    return folds


def validate_remediation_matrix(configurations: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject any expansion or mutation of the fixed R0-R5 search space."""

    if tuple(configurations) != tuple(f"R{index}" for index in range(6)):
        raise ValueError("remediation matrix must contain exactly ordered R0 through R5")
    for config_id, config in configurations.items():
        if dict(config) != EXPECTED_MATRIX[config_id]:
            raise ValueError(f"{config_id} differs from the predeclared matrix")


def _preprocessor(features: tuple[str, ...]) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric_features = [feature for feature in features if feature not in CATEGORICAL_FEATURES]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MaxAbsScaler())])
    return ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, numeric_features),
        ],
        sparse_threshold=1.0,
    )


def build_remediation_model(
    config_id: str, config: Mapping[str, Any]
) -> tuple[Pipeline, tuple[str, ...]]:
    """Build one and only one predeclared sparse linear model."""

    if config_id not in {f"R{index}" for index in range(6)}:
        raise ValueError(f"unauthorized remediation configuration: {config_id}")
    cyclical = config["feature_set"] == "cyclical_no_route"
    input_features = CYCLICAL_NO_ROUTE_FEATURES if cyclical else CANDIDATE_A_FEATURES
    model_features = CYCLICAL_MODEL_FEATURES if cyclical else input_features
    validate_model_features(input_features)
    validate_model_features(model_features)
    if "route" in input_features:
        raise ValueError("route is display-only in Brief 04")
    if config["estimator"] == "sgd":
        estimator: Any = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(config["alpha"]),
            class_weight=config.get("class_weight"),
            average=bool(config["average"]),
            max_iter=int(config["max_iter"]),
            tol=float(config["tol"]),
            random_state=42,
        )
    else:
        estimator = LogisticRegression(
            solver=str(config["solver"]),
            penalty=str(config["penalty"]),
            C=float(config["C"]),
            class_weight=config.get("class_weight"),
            max_iter=int(config["max_iter"]),
            tol=float(config["tol"]),
            random_state=42,
        )
    steps: list[tuple[str, Any]] = []
    if cyclical:
        steps.append(("month_cyclical", MonthCyclicalAugmenter()))
    steps.extend((("preprocessing", _preprocessor(model_features)), ("classifier", estimator)))
    return Pipeline(steps), input_features


def rank_base_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank completed configurations with the exact rolling-origin ordering."""

    return sorted(
        (row for row in results if row.get("status") == "completed"),
        key=lambda row: (
            -float(row["mean_average_precision"]),
            -float(row["mean_roc_auc"]),
            float(row["std_average_precision"]),
            float(row["mean_log_loss"]),
            str(row["configuration_id"]),
        ),
    )


def authorized_calibration_ids(ranked: list[dict[str, Any]]) -> tuple[str, ...]:
    """Authorize the top two bases plus R0 control, never more than three."""

    selected = [str(row["configuration_id"]) for row in ranked[:2]]
    completed_ids = {str(row["configuration_id"]) for row in ranked}
    if "R0" in completed_ids and "R0" not in selected:
        selected.append("R0")
    return tuple(selected)


def prior_scores(target: Any) -> dict[str, float]:
    """Return period-specific constant-prior Brier and log-loss values."""

    labels = np.asarray(target, dtype=int)
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("prior scoring requires both target classes")
    prevalence = float(labels.mean())
    probabilities = np.full(len(labels), prevalence)
    return {
        "prevalence": prevalence,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
