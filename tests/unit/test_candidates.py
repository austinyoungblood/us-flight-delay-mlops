from __future__ import annotations

import pandas as pd

from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.candidates import (
    CANDIDATE_B_INPUT_FEATURES,
    CANDIDATE_B_MODEL_FEATURES,
    MonthCyclicalAugmenter,
    build_candidate,
)


def test_candidate_b_additions_are_approved_and_distinct() -> None:
    assert set(CANDIDATE_A_FEATURES) < set(CANDIDATE_B_INPUT_FEATURES)
    assert {"route", "scheduled_departure_sin", "scheduled_arrival_cos"} <= set(
        CANDIDATE_B_INPUT_FEATURES
    )
    assert {"month_sin", "month_cos"} <= set(CANDIDATE_B_MODEL_FEATURES)
    validate_model_features(CANDIDATE_B_INPUT_FEATURES)
    validate_model_features(CANDIDATE_B_MODEL_FEATURES)


def test_month_features_are_derived_inside_candidate_b_pipeline() -> None:
    frame = pd.DataFrame({feature: [1, 2] for feature in CANDIDATE_B_INPUT_FEATURES})
    frame["Reporting_Airline"] = ["UA", "WN"]
    frame["Origin"] = ["DEN", "DEN"]
    frame["Dest"] = ["LAX", "SFO"]
    frame["route"] = ["DEN-LAX", "DEN-SFO"]
    augmented = MonthCyclicalAugmenter().fit_transform(frame)
    assert {"month_sin", "month_cos"} <= set(augmented)

    _, schema = build_candidate("candidate_b", {"alpha": 0.0001, "class_weight": None})
    assert schema == CANDIDATE_B_INPUT_FEATURES
