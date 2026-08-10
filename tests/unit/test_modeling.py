from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.modeling.artifacts import (
    MODEL_BUNDLE_FILES,
    build_training_baseline,
    write_model_bundle,
)
from flight_delay.modeling.baselines import build_estimator
from flight_delay.modeling.evaluation import EvaluationError, evaluate_binary


def _frame(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Reporting_Airline": ["UA" if index % 2 else "WN" for index in range(rows)],
            "Origin": ["DEN"] * rows,
            "Dest": ["LAX" if index % 3 else "SFO" for index in range(rows)],
            "Month": [1] * rows,
            "DayofMonth": [(index % 28) + 1 for index in range(rows)],
            "DayOfWeek": [(index % 7) + 1 for index in range(rows)],
            "CRSDepTime": [700 + index for index in range(rows)],
            "CRSArrTime": [900 + index for index in range(rows)],
            "CRSElapsedTime": [120.0 + index for index in range(rows)],
            "Distance": [800.0 + index for index in range(rows)],
            "scheduled_departure_hour": [7] * rows,
            "scheduled_arrival_hour": [9] * rows,
            "scheduled_departure_minute_bucket": [0] * rows,
            "scheduled_arrival_minute_bucket": [0] * rows,
            "is_weekend": [index % 2 for index in range(rows)],
            "target": [index % 2 for index in range(rows)],
        }
    )


@pytest.mark.parametrize("candidate_id", ["dummy", "candidate_a"])
def test_estimators_fit_and_predict_probabilities(candidate_id: str) -> None:
    frame = _frame()
    estimator = build_estimator(candidate_id, {})
    estimator.fit(frame.loc[:, CANDIDATE_A_FEATURES], frame["target"])
    probabilities = estimator.predict_proba(frame.loc[:, CANDIDATE_A_FEATURES])[:, 1]
    assert probabilities.shape == (len(frame),)
    assert np.isfinite(probabilities).all()


def test_evaluation_metrics_and_one_class_failure() -> None:
    result = evaluate_binary([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert result.metrics["true_positive"] == 2
    assert result.metrics["false_positive"] == 1
    assert set(result.figures) == {
        "confusion_matrix",
        "precision_recall_curve",
        "roc_curve",
        "calibration",
    }
    for figure in result.figures.values():
        plt.close(figure)
    with pytest.raises(EvaluationError, match="both target classes"):
        evaluate_binary([0, 0], [0.1, 0.2])


def test_complete_model_bundle_and_aggregate_baseline(tmp_path: Path) -> None:
    frame = _frame()
    model = build_estimator("dummy", {})
    model.fit(frame.loc[:, CANDIDATE_A_FEATURES], frame["target"])
    baseline = build_training_baseline(
        frame, dataset_artifact="entity/project/data:v0", dataset_digest="digest"
    )
    result = write_model_bundle(
        directory=tmp_path / "bundle",
        model=model,
        feature_schema=list(CANDIDATE_A_FEATURES),
        threshold=0.5,
        training_baseline=baseline,
        metrics={"accuracy": 0.5},
        metadata={"candidate_id": "dummy", "validation_only": True},
    )
    assert {path.name for path in result.directory.iterdir()} == MODEL_BUNDLE_FILES
    assert result.byte_size > 0
    assert baseline["categorical"]["carrier"]["__OTHER__"] == 0.0
