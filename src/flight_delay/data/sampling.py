"""Deterministic bounded monthly sampling."""

from collections.abc import Hashable

import numpy as np
import pandas as pd

from flight_delay.data.preprocessing import DataQualityError


def _allocate_strata(counts: pd.Series, cap: int) -> dict[Hashable, int]:
    ideal = counts * cap / int(counts.sum())
    allocated = ideal.astype(int)
    if cap >= len(counts):
        allocated[allocated.eq(0)] = 1
    while int(allocated.sum()) > cap:
        candidates = allocated[allocated.gt(1)]
        key = candidates.sort_values(ascending=False, kind="stable").index[0]
        allocated.loc[key] -= 1
    remainder = cap - int(allocated.sum())
    fractions = (ideal - ideal.astype(int)).sort_values(ascending=False, kind="stable")
    for key in list(fractions.index) * (remainder + 1):
        if remainder == 0:
            break
        if allocated.loc[key] < counts.loc[key]:
            allocated.loc[key] += 1
            remainder -= 1
    return {key: int(value) for key, value in allocated.items()}


def deterministic_monthly_sample(
    frame: pd.DataFrame,
    *,
    max_rows_per_month: int | None,
    seed: int,
    date_column: str = "FlightDate",
    stratify_column: str | None = "target",
) -> pd.DataFrame:
    """Return at most ``max_rows_per_month`` rows, reproducibly sampled per month."""

    if date_column not in frame:
        raise DataQualityError(f"missing date column: {date_column}")
    if max_rows_per_month is None:
        return frame.copy()
    if max_rows_per_month <= 0:
        raise DataQualityError("max_rows_per_month must be positive or None")
    if stratify_column is not None and stratify_column not in frame:
        raise DataQualityError(f"missing stratification column: {stratify_column}")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise DataQualityError(f"{date_column} contains invalid or missing dates")
    working = frame.copy()
    working["__sample_month"] = dates.dt.to_period("M")
    selected: list[pd.DataFrame] = []
    master_rng = np.random.default_rng(seed)

    for _, month in working.groupby("__sample_month", sort=True):
        if len(month) <= max_rows_per_month:
            selected.append(month)
            continue
        if stratify_column is None:
            random_state = int(master_rng.integers(0, 2**32 - 1))
            selected.append(month.sample(n=max_rows_per_month, random_state=random_state))
            continue

        counts = month[stratify_column].value_counts(dropna=False, sort=False)
        allocations = _allocate_strata(counts, max_rows_per_month)
        strata: list[pd.DataFrame] = []
        for value, count in allocations.items():
            mask = month[stratify_column].isna() if pd.isna(value) else month[stratify_column].eq(value)
            random_state = int(master_rng.integers(0, 2**32 - 1))
            strata.append(month.loc[mask].sample(n=count, random_state=random_state))
        selected.append(pd.concat(strata))

    sampled = pd.concat(selected) if selected else working.iloc[0:0]
    return sampled.drop(columns="__sample_month").sort_index().copy()
