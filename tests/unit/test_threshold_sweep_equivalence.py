from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from flight_delay.modeling.v1_selection import (
    ThresholdRow,
    V1ThresholdSelection,
    choose_threshold_row,
    select_v1_threshold,
)


def _brute_force_threshold_selection(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    recall_min: float,
    precision_min: float,
    predicted_positive_rate_max: float,
) -> V1ThresholdSelection:
    """Test-only reference matching the original exhaustive implementation."""

    target = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    rows: list[ThresholdRow] = []
    for threshold in sorted(set(map(float, scores)), reverse=True):
        predicted = (scores >= threshold).astype(int)
        precision = float(precision_score(target, predicted, zero_division=0))
        recall = float(recall_score(target, predicted, zero_division=0))
        f1 = float(f1_score(target, predicted, zero_division=0))
        positive_rate = float(predicted.mean())
        rows.append(
            ThresholdRow(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                predicted_positive_rate=positive_rate,
                eligible=(
                    recall >= recall_min
                    and precision >= precision_min
                    and positive_rate <= predicted_positive_rate_max
                ),
            )
        )
    selected = choose_threshold_row(rows)
    if selected is None:
        return V1ThresholdSelection(None, None, tuple(rows))
    return V1ThresholdSelection(selected.threshold, selected, tuple(rows))


def _assert_row_equal(optimized: ThresholdRow, reference: ThresholdRow) -> None:
    assert optimized.threshold.hex() == reference.threshold.hex()
    assert optimized.precision == pytest.approx(reference.precision, rel=1e-15, abs=1e-15)
    assert optimized.recall == pytest.approx(reference.recall, rel=1e-15, abs=1e-15)
    assert optimized.f1 == pytest.approx(reference.f1, rel=1e-15, abs=1e-15)
    assert optimized.predicted_positive_rate == pytest.approx(
        reference.predicted_positive_rate, rel=1e-15, abs=1e-15
    )
    assert optimized.eligible is reference.eligible


def _assert_equivalent(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    recall_min: float,
    precision_min: float,
    predicted_positive_rate_max: float,
) -> None:
    arguments = {
        "recall_min": recall_min,
        "precision_min": precision_min,
        "predicted_positive_rate_max": predicted_positive_rate_max,
    }
    optimized = select_v1_threshold(labels, probabilities, **arguments)
    reference = _brute_force_threshold_selection(labels, probabilities, **arguments)

    assert len(optimized.threshold_table) == len(reference.threshold_table)
    for optimized_row, reference_row in zip(
        optimized.threshold_table, reference.threshold_table, strict=True
    ):
        _assert_row_equal(optimized_row, reference_row)

    if reference.selected_threshold is None:
        assert optimized.selected_threshold is None
        assert optimized.selected_metrics is None
    else:
        assert optimized.selected_threshold is not None
        assert optimized.selected_threshold.hex() == reference.selected_threshold.hex()
        assert optimized.selected_metrics is not None
        assert reference.selected_metrics is not None
        _assert_row_equal(optimized.selected_metrics, reference.selected_metrics)


@pytest.mark.parametrize("seed", range(64))
def test_optimized_threshold_sweep_matches_brute_force_properties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    row_count = int(rng.integers(8, 96))
    if seed % 5 == 0:
        labels = np.zeros(row_count, dtype=int)
        labels[int(rng.integers(0, row_count))] = 1
    else:
        labels = (rng.random(row_count) < rng.uniform(0.05, 0.70)).astype(int)
        labels[0], labels[1] = 0, 1

    mode = seed % 4
    if mode == 0:
        scores = rng.choice(np.array([0.0, 0.1, 0.3, 0.5, 0.9, 1.0]), row_count)
    elif mode == 1:
        scores = (np.arange(row_count, dtype=float) + 0.5) / row_count
        rng.shuffle(scores)
    elif mode == 2:
        scores = np.round(rng.random(row_count), decimals=2)
    else:
        scores = rng.random(row_count)
        scores[::5] = 0.25

    _assert_equivalent(
        labels,
        scores,
        recall_min=float(rng.choice([0.0, 0.3, 0.6, 0.9, 1.1])),
        precision_min=float(rng.choice([0.0, 0.3, 0.5, 0.8, 1.1])),
        predicted_positive_rate_max=float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
    )


@pytest.mark.parametrize(
    ("labels", "scores", "recall_min", "precision_min", "ppr_max"),
    [
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0, 0.0, 1.0),
        ([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2], 0.5, 0.6, 0.5),
        ([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], 1.1, 0.3, 0.5),
        (
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.7] * 10 + [0.2] * 10,
            0.6,
            0.3,
            0.5,
        ),
        (
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            0.0,
            0.0,
            1.0,
        ),
    ],
)
def test_threshold_sweep_exact_boundaries_and_selection_cases(
    labels: list[int],
    scores: list[float],
    recall_min: float,
    precision_min: float,
    ppr_max: float,
) -> None:
    _assert_equivalent(
        labels,
        scores,
        recall_min=recall_min,
        precision_min=precision_min,
        predicted_positive_rate_max=ppr_max,
    )


@pytest.mark.parametrize("scores", [[-0.0, 0.0], [0.0, -0.0]])
def test_threshold_sweep_preserves_the_first_observed_signed_zero(scores: list[float]) -> None:
    _assert_equivalent(
        [0, 1],
        scores,
        recall_min=0.0,
        precision_min=0.0,
        predicted_positive_rate_max=1.0,
    )


def test_zero_true_positive_boundary_preserves_zero_division_metrics() -> None:
    result = select_v1_threshold(
        [0, 0, 1, 1],
        [0.9, 0.8, 0.2, 0.1],
        recall_min=0.0,
        precision_min=0.0,
        predicted_positive_rate_max=1.0,
    )
    # One row is always predicted positive at an observed-score threshold; here TP is zero, which
    # is the applicable zero-division boundary for the exact unique-score sweep.
    assert result.threshold_table[0] == ThresholdRow(0.9, 0.0, 0.0, 0.0, 0.25, True)


def test_exact_boundary_row_is_eligible_and_multiple_f1_ties_use_existing_tie_breaks() -> None:
    boundary = select_v1_threshold(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.7] * 10 + [0.2] * 10,
        recall_min=0.6,
        precision_min=0.3,
        predicted_positive_rate_max=0.5,
    )
    assert boundary.selected_metrics == ThresholdRow(0.7, 0.3, 0.6, 0.4, 0.5, True)

    tied = select_v1_threshold(
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
        recall_min=0.0,
        precision_min=0.0,
        predicted_positive_rate_max=1.0,
    )
    f1_ties = [row for row in tied.threshold_table if row.f1 == pytest.approx(0.5)]
    assert [row.threshold for row in f1_ties] == [0.95, 0.55]
    assert tied.selected_threshold == 0.95
