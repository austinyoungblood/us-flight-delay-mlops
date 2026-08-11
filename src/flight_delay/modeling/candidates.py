"""Brief 03 calibrated control and Candidate B model definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.baselines import candidate_a_pipeline

CANDIDATE_B_INPUT_FEATURES: tuple[str, ...] = (
    *CANDIDATE_A_FEATURES,
    "route",
    "scheduled_departure_sin",
    "scheduled_departure_cos",
    "scheduled_arrival_sin",
    "scheduled_arrival_cos",
)
CANDIDATE_B_MODEL_FEATURES: tuple[str, ...] = (
    *CANDIDATE_B_INPUT_FEATURES,
    "month_sin",
    "month_cos",
)
CANDIDATE_B_CATEGORICAL: tuple[str, ...] = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "route",
)
CANDIDATE_B_NUMERIC: tuple[str, ...] = tuple(
    feature for feature in CANDIDATE_B_MODEL_FEATURES if feature not in CANDIDATE_B_CATEGORICAL
)


class MonthCyclicalAugmenter(TransformerMixin, BaseEstimator):
    """Add month sine/cosine fields inside the serialized inference pipeline."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> MonthCyclicalAugmenter:
        validate_model_features(features.columns)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        validate_model_features(features.columns)
        result = features.copy()
        month = pd.to_numeric(result["Month"], errors="raise")
        angle = 2 * np.pi * (month - 1) / 12
        result["month_sin"] = np.sin(angle)
        result["month_cos"] = np.cos(angle)
        validate_model_features(result.columns)
        return result


def build_candidate(
    candidate_id: str, parameters: Mapping[str, Any]
) -> tuple[Any, tuple[str, ...]]:
    """Build a calibrated-control base or Candidate B base pipeline."""

    if candidate_id == "candidate_a_calibrated":
        validate_model_features(CANDIDATE_A_FEATURES)
        return candidate_a_pipeline(parameters), CANDIDATE_A_FEATURES
    if candidate_id != "candidate_b":
        raise ValueError(f"unsupported Brief 03 candidate: {candidate_id}")
    validate_model_features(CANDIDATE_B_INPUT_FEATURES)
    validate_model_features(CANDIDATE_B_MODEL_FEATURES)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MaxAbsScaler())])
    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical, list(CANDIDATE_B_CATEGORICAL)),
            ("numeric", numeric, list(CANDIDATE_B_NUMERIC)),
        ],
        sparse_threshold=1.0,
    )
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(parameters["alpha"]),
        class_weight=parameters.get("class_weight"),
        max_iter=int(parameters.get("max_iter", 1000)),
        tol=float(parameters.get("tol", 0.001)),
        average=bool(parameters.get("average", False)),
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("month_cyclical", MonthCyclicalAugmenter()),
            ("preprocessing", preprocessing),
            ("classifier", estimator),
        ]
    )
    return pipeline, CANDIDATE_B_INPUT_FEATURES
