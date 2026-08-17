"""Deterministic schedule-only seasonal features with exact training-serving parity.

Every value here is a pure function of one scheduled flight date. No label, no surrounding row,
and no historical state may influence it, so the vectorized batch path and the single-row serving
path are required to agree exactly.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from flight_delay.modeling.v3.protocol import (
    CHRISTMAS_WINDOW_RANGE,
    DAY_DISTANCE_CLIP,
    DETERMINISTIC_SEASONAL_FEATURES,
    THANKSGIVING_WINDOW_RANGE,
)

THURSDAY = 3
CHRISTMAS_MONTH = 12
CHRISTMAS_DAY = 25


class V3SeasonalError(ValueError):
    """Raised when a seasonal feature cannot be derived deterministically."""


def thanksgiving(year: int) -> date:
    """Return the fourth Thursday of November, defined algorithmically."""

    november_first = date(year, 11, 1)
    first_thursday = november_first + timedelta(days=(THURSDAY - november_first.weekday()) % 7)
    return first_thursday + timedelta(days=21)


def christmas(year: int) -> date:
    """Return December 25 of the requested year."""

    return date(year, CHRISTMAS_MONTH, CHRISTMAS_DAY)


def _nearest_signed_delta(flight_date: date, anchors: tuple[date, date, date]) -> int:
    """Return the signed day delta to the nearest anchor, breaking ties toward the earlier one.

    The sign convention is ``(anchor - flight_date).days``, so a positive value means the anchor
    still lies in the future. ``anchors`` must be supplied in ascending date order, which makes the
    first minimum the earlier anchor.
    """

    best: int | None = None
    for anchor in anchors:
        delta = (anchor - flight_date).days
        if best is None or abs(delta) < abs(best):
            best = delta
    if best is None:  # pragma: no cover - anchors is a fixed-length tuple
        raise V3SeasonalError("anchor selection requires at least one candidate")
    return best


def _clip(value: int) -> int:
    lower, upper = DAY_DISTANCE_CLIP
    return int(min(max(value, lower), upper))


def _in_window(value: int, window: tuple[int, int]) -> int:
    lower, upper = window
    return int(lower <= value <= upper)


def seasonal_features_for_date(flight_date: date) -> dict[str, float]:
    """Derive all six deterministic seasonal features for one scheduled flight date."""

    if not isinstance(flight_date, date):
        raise V3SeasonalError("seasonal features require a calendar date")
    year = flight_date.year
    days_in_year = 366 if pd.Timestamp(flight_date).is_leap_year else 365
    angle = 2.0 * math.pi * (flight_date.timetuple().tm_yday - 1) / days_in_year
    thanksgiving_delta = _nearest_signed_delta(
        flight_date, (thanksgiving(year - 1), thanksgiving(year), thanksgiving(year + 1))
    )
    christmas_delta = _nearest_signed_delta(
        flight_date, (christmas(year - 1), christmas(year), christmas(year + 1))
    )
    return {
        "day_of_year_sin": math.sin(angle),
        "day_of_year_cos": math.cos(angle),
        "days_to_thanksgiving": _clip(thanksgiving_delta),
        "is_thanksgiving_window": _in_window(thanksgiving_delta, THANKSGIVING_WINDOW_RANGE),
        "days_to_christmas": _clip(christmas_delta),
        "is_christmas_window": _in_window(christmas_delta, CHRISTMAS_WINDOW_RANGE),
    }


def _anchor_ordinals(years: np.ndarray, anchor: str) -> np.ndarray:
    """Build an ascending (rows x 3) ordinal matrix of prior, current, and next anchors."""

    builder = thanksgiving if anchor == "thanksgiving" else christmas
    unique = np.unique(years)
    lookup = {
        int(year): builder(int(year)).toordinal()
        for year in range(int(unique.min()) - 1, int(unique.max()) + 2)
    }
    table = np.vectorize(lookup.__getitem__)
    return np.column_stack([table(years - 1), table(years), table(years + 1)])


def _nearest_signed_delta_vector(ordinals: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    deltas = anchors - ordinals[:, None]
    # ``argmin`` returns the first minimum, and the anchor columns ascend by date, so ties
    # resolve to the earlier anchor exactly as the scalar path does.
    chosen = np.argmin(np.abs(deltas), axis=1)
    return np.take_along_axis(deltas, chosen[:, None], axis=1).ravel()


def derive_seasonal_features(flight_date: pd.Series) -> pd.DataFrame:
    """Vectorized batch equivalent of :func:`seasonal_features_for_date`."""

    dates = pd.to_datetime(flight_date, errors="coerce").dt.normalize()
    if dates.empty or dates.isna().any():
        raise V3SeasonalError("seasonal features require valid non-empty flight dates")
    years = dates.dt.year.to_numpy()
    ordinals = np.asarray([timestamp.toordinal() for timestamp in dates], dtype=np.int64)
    days_in_year = np.where(dates.dt.is_leap_year.to_numpy(), 366, 365)
    angle = 2.0 * np.pi * (dates.dt.dayofyear.to_numpy() - 1) / days_in_year
    thanksgiving_delta = _nearest_signed_delta_vector(
        ordinals, _anchor_ordinals(years, "thanksgiving")
    )
    christmas_delta = _nearest_signed_delta_vector(ordinals, _anchor_ordinals(years, "christmas"))
    lower, upper = DAY_DISTANCE_CLIP
    thanksgiving_low, thanksgiving_high = THANKSGIVING_WINDOW_RANGE
    christmas_low, christmas_high = CHRISTMAS_WINDOW_RANGE
    result = pd.DataFrame(
        {
            "day_of_year_sin": np.sin(angle),
            "day_of_year_cos": np.cos(angle),
            "days_to_thanksgiving": np.clip(thanksgiving_delta, lower, upper),
            "is_thanksgiving_window": (
                (thanksgiving_delta >= thanksgiving_low) & (thanksgiving_delta <= thanksgiving_high)
            ).astype(int),
            "days_to_christmas": np.clip(christmas_delta, lower, upper),
            "is_christmas_window": (
                (christmas_delta >= christmas_low) & (christmas_delta <= christmas_high)
            ).astype(int),
        },
        index=dates.index,
    )
    if tuple(result.columns) != DETERMINISTIC_SEASONAL_FEATURES:
        raise V3SeasonalError("deterministic seasonal feature order drifted")
    return result
