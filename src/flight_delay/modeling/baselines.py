"""Leakage-safe Dummy and Candidate A estimator construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.features.leakage import validate_model_features

CATEGORICAL_FEATURES: tuple[str, ...] = ("Reporting_Airline", "Origin", "Dest")
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    feature for feature in CANDIDATE_A_FEATURES if feature not in CATEGORICAL_FEATURES
)


def candidate_a_pipeline(parameters: Mapping[str, Any]) -> Pipeline:
    """Build Candidate A as one serializable preprocessing/estimator pipeline."""

    validate_model_features(CANDIDATE_A_FEATURES)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MaxAbsScaler())])
    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        sparse_threshold=1.0,
    )
    estimator = SGDClassifier(
        loss=str(parameters.get("loss", "log_loss")),
        class_weight=parameters.get("class_weight", "balanced"),
        alpha=float(parameters.get("alpha", 0.0001)),
        max_iter=int(parameters.get("max_iter", 1000)),
        tol=float(parameters.get("tol", 0.001)),
        average=bool(parameters.get("average", False)),
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def build_estimator(candidate_id: str, parameters: Mapping[str, Any]) -> Any:
    """Build one of the two explicitly permitted baseline estimators."""

    validate_model_features(CANDIDATE_A_FEATURES)
    if candidate_id == "dummy":
        return DummyClassifier(strategy="prior")
    if candidate_id == "candidate_a":
        return candidate_a_pipeline(parameters)
    raise ValueError(f"unsupported baseline candidate: {candidate_id}")
