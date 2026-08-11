from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flight_delay.modeling.remediation import (
    CYCLICAL_NO_ROUTE_FEATURES,
    EXPECTED_MATRIX,
    authorized_calibration_ids,
    build_remediation_model,
    partition_remediation_data,
    prior_scores,
    rank_base_results,
    validate_remediation_matrix,
)


def _frame(start: str, periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_date": pd.date_range(start, periods=periods),
            "target": [0, 1] * (periods // 2) + ([0] if periods % 2 else []),
        }
    )


def test_remediation_partitions_are_stable_and_exclude_december() -> None:
    train = pd.concat(
        [_frame(f"2025-{month:02d}-01", 2) for month in range(1, 11)], ignore_index=True
    ).sample(frac=1, random_state=42)
    november = pd.concat([_frame("2025-11-01", 2), _frame("2025-11-16", 2)], ignore_index=True)

    partitions = partition_remediation_data(train, november)

    assert len(partitions.rolling_folds) == 4
    assert len(partitions.final_fit) == 20
    assert set(partitions.calibration.index).isdisjoint(partitions.selection.index)
    assert partitions.selection["flight_date"].max() < pd.Timestamp("2025-12-01")
    assert all(fold.fit["flight_date"].is_monotonic_increasing for fold in partitions.rolling_folds)


def test_fixed_matrix_cardinality_and_route_prohibition() -> None:
    configurations = EXPECTED_MATRIX
    validate_remediation_matrix(configurations)
    with pytest.raises(ValueError, match="exactly ordered"):
        validate_remediation_matrix({**configurations, "R6": configurations["R0"]})
    assert "route" not in CYCLICAL_NO_ROUTE_FEATURES
    changed = {key: dict(value) for key, value in configurations.items()}
    changed["R1"]["alpha"] = 0.5
    with pytest.raises(ValueError, match="differs"):
        validate_remediation_matrix(changed)


def test_model_builder_rejects_matrix_expansion() -> None:
    with pytest.raises(ValueError, match="unauthorized"):
        build_remediation_model("R6", {})


def test_base_ranking_and_authorization_are_deterministic() -> None:
    rows = [
        {
            "configuration_id": "R0",
            "status": "completed",
            "mean_average_precision": 0.30,
            "mean_roc_auc": 0.60,
            "std_average_precision": 0.02,
            "mean_log_loss": 0.6,
        },
        {
            "configuration_id": "R1",
            "status": "completed",
            "mean_average_precision": 0.32,
            "mean_roc_auc": 0.62,
            "std_average_precision": 0.01,
            "mean_log_loss": 0.5,
        },
        {
            "configuration_id": "R2",
            "status": "completed",
            "mean_average_precision": 0.32,
            "mean_roc_auc": 0.62,
            "std_average_precision": 0.01,
            "mean_log_loss": 0.5,
        },
    ]
    assert [row["configuration_id"] for row in rank_base_results(rows)] == ["R1", "R2", "R0"]
    assert authorized_calibration_ids(rank_base_results(rows)) == ("R1", "R2", "R0")


def test_period_specific_prior_scores() -> None:
    scores = prior_scores([0, 0, 1, 1])
    assert scores["prevalence"] == 0.5
    assert scores["brier_score"] == pytest.approx(0.25)
    assert scores["log_loss"] == pytest.approx(0.69314718056)


def test_prequalification_orchestrator_has_no_test_split_path() -> None:
    source = Path("scripts/remediate_model.py").read_text(encoding="utf-8")
    assert "test.parquet" not in source
