"""Precommitted recency weight policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v3.protocol import EXPONENTIAL_HALF_LIFE_DAYS, WEIGHT_POLICY_IDS
from flight_delay.modeling.v3.weighting import V3WeightError, fit_weights, weight_summary


def dates(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


def test_exactly_two_policies_are_precommitted() -> None:
    assert WEIGHT_POLICY_IDS == ("UNIFORM", "EXPONENTIAL_120D")


def test_uniform_passes_no_sample_weight() -> None:
    assert (
        fit_weights(dates("2025-01-01", "2025-06-01"), policy="UNIFORM", fit_cutoff="2025-10-31")
        is None
    )


def test_exponential_halves_every_one_hundred_twenty_days() -> None:
    cutoff = "2025-10-31"
    series = dates("2025-10-31", "2025-07-03", "2025-03-05")
    weights = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff=cutoff)
    assert weights is not None
    ages = (pd.Timestamp(cutoff) - series).dt.days.to_numpy()
    raw = 0.5 ** (ages / EXPONENTIAL_HALF_LIFE_DAYS)
    assert weights == pytest.approx(raw / raw.mean())
    # Two rows exactly one half-life apart differ by a factor of two before normalization.
    assert raw[0] / raw[1] == pytest.approx(2.0, rel=1e-9)


def test_exponential_normalizes_to_mean_one() -> None:
    series = pd.Series(pd.date_range("2024-02-01", "2025-10-31", freq="D"))
    weights = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff="2025-10-31")
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-12)
    assert (weights > 0).all()
    assert np.isfinite(weights).all()


def test_more_recent_rows_receive_strictly_greater_weight() -> None:
    series = pd.Series(pd.date_range("2024-02-01", "2025-10-31", freq="MS"))
    weights = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff="2025-10-31")
    assert weights is not None
    assert np.all(np.diff(weights) > 0)


def test_normalized_weights_depend_only_on_relative_age() -> None:
    """Moving the cutoff scales every raw weight equally, so normalization cancels it out.

    Each fold therefore differs by which rows it includes, not by how a fixed set of rows is
    weighted relative to one another.
    """

    series = dates("2025-01-01", "2025-06-01", "2025-07-31")
    early = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff="2025-07-31")
    late = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff="2025-10-31")
    assert early is not None and late is not None
    assert np.allclose(early, late)
    assert float(early.mean()) == pytest.approx(1.0)
    assert float(late.mean()) == pytest.approx(1.0)


def test_a_longer_fold_reweights_because_it_includes_older_rows() -> None:
    short = pd.Series(pd.date_range("2025-05-01", "2025-07-31", freq="D"))
    long = pd.Series(pd.date_range("2024-02-01", "2025-07-31", freq="D"))
    short_weights = fit_weights(short, policy="EXPONENTIAL_120D", fit_cutoff="2025-07-31")
    long_weights = fit_weights(long, policy="EXPONENTIAL_120D", fit_cutoff="2025-07-31")
    assert short_weights is not None and long_weights is not None
    # The long fold's mass concentrates far more sharply on its most recent rows.
    assert long_weights.max() > short_weights.max()


def test_rows_after_the_cutoff_are_refused() -> None:
    with pytest.raises(V3WeightError, match="postdate"):
        fit_weights(dates("2025-11-15"), policy="EXPONENTIAL_120D", fit_cutoff="2025-10-31")


def test_unknown_policy_is_refused() -> None:
    with pytest.raises(V3WeightError, match="unknown"):
        fit_weights(dates("2025-01-01"), policy="LINEAR_90D", fit_cutoff="2025-10-31")


def test_invalid_dates_are_refused() -> None:
    with pytest.raises(V3WeightError):
        fit_weights(
            pd.Series([], dtype="datetime64[ns]"), policy="UNIFORM", fit_cutoff="2025-10-31"
        )
    with pytest.raises(V3WeightError):
        fit_weights(pd.Series([pd.NaT]), policy="UNIFORM", fit_cutoff="2025-10-31")


def test_summary_reports_the_normalization_contract() -> None:
    series = pd.Series(pd.date_range("2024-02-01", "2025-10-31", freq="D"))
    weights = fit_weights(series, policy="EXPONENTIAL_120D", fit_cutoff="2025-10-31")
    summary = weight_summary(weights, policy="EXPONENTIAL_120D")
    assert summary["normalized_to_mean_one"] is True
    assert summary["rows"] == len(series)
    assert summary["max"] > summary["min"]
    uniform = weight_summary(None, policy="UNIFORM")
    assert uniform["mean"] == 1.0
    assert uniform["normalized_to_mean_one"] is True
