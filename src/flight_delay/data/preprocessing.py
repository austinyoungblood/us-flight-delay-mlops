"""Small, pure transformations for BTS on-time performance records."""

from dataclasses import dataclass
from datetime import time
from numbers import Integral, Real
from typing import Any

import pandas as pd

_CANONICAL_COLUMNS = (
    "FlightDate",
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Cancelled",
    "Diverted",
    "ArrDel15",
    "ArrDelay",
    "ArrDelayMinutes",
)
_COLUMN_LOOKUP = {
    "".join(character for character in column.casefold() if character.isalnum()): column
    for column in _CANONICAL_COLUMNS
}


class DataQualityError(ValueError):
    """Raised when a BTS frame violates an explicit data contract."""


class InvalidCRSTimeError(DataQualityError):
    """Raised when a CRS time cannot represent a valid HHMM value."""


@dataclass(frozen=True)
class ExclusionCounts:
    """Mutually exclusive counts from sequential eligibility filtering."""

    input_rows: int
    cancelled: int
    diverted: int
    missing_target: int
    eligible_rows: int

    @property
    def excluded_rows(self) -> int:
        """Return the total number of excluded rows."""

        return self.cancelled + self.diverted + self.missing_target


@dataclass(frozen=True)
class EligibilityResult:
    """Eligible records plus an auditable exclusion summary."""

    frame: pd.DataFrame
    counts: ExclusionCounts


def _column_key(column: object) -> str:
    text = str(column).lstrip("\ufeff").strip()
    return "".join(character for character in text.casefold() if character.isalnum())


def normalize_bts_columns(
    frame: pd.DataFrame,
    *,
    required_columns: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Return a copy with known BTS column names normalized and required fields checked."""

    rename = {
        column: _COLUMN_LOOKUP.get(_column_key(column), str(column).lstrip("\ufeff").strip())
        for column in frame.columns
    }
    normalized_names = list(rename.values())
    duplicates = sorted({name for name in normalized_names if normalized_names.count(name) > 1})
    if duplicates:
        raise DataQualityError(f"column normalization produced duplicates: {duplicates}")

    normalized = frame.rename(columns=rename).copy()
    missing = sorted(set(required_columns).difference(normalized.columns))
    if missing:
        raise DataQualityError(f"missing required BTS columns: {missing}")
    return normalized


def parse_crs_time(value: Any) -> time:
    """Parse a BTS CRS ``HHMM`` integer/string, accepting ``0000`` and ``2400``.

    Invalid or missing values raise :class:`InvalidCRSTimeError`; they are never
    silently coerced. BTS sometimes serializes integral values as floats, so an
    integral finite float is accepted.
    """

    original = value
    if isinstance(value, bool) or value is None or value is pd.NA:
        raise InvalidCRSTimeError(f"invalid CRS time {original!r}")
    if isinstance(value, Real) and not isinstance(value, Integral):
        if pd.isna(value) or not float(value).is_integer():
            raise InvalidCRSTimeError(f"invalid CRS time {original!r}")
        value = int(value)
    if isinstance(value, Integral):
        numeric = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdigit() or len(stripped) > 4:
            raise InvalidCRSTimeError(f"invalid CRS time {original!r}")
        numeric = int(stripped)
    else:
        raise InvalidCRSTimeError(f"invalid CRS time {original!r}")

    if numeric == 2400:
        return time(0, 0)
    hours, minutes = divmod(numeric, 100)
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise InvalidCRSTimeError(f"invalid CRS time {original!r}")
    return time(hours, minutes)


def _validate_binary_indicator(series: pd.Series, name: str) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_rows = series.index[numeric.isna()]
    if len(invalid_rows):
        raise DataQualityError(f"{name} has missing/invalid values at rows {list(invalid_rows)[:10]}")
    invalid = sorted(set(numeric.unique()).difference({0, 1}))
    if invalid:
        raise DataQualityError(f"{name} must contain only 0, 1, or missing values; got {invalid}")


def filter_eligible_flights(frame: pd.DataFrame) -> EligibilityResult:
    """Remove cancelled, diverted, then missing-target rows and report each count."""

    required = frozenset({"Cancelled", "Diverted", "ArrDel15"})
    clean = normalize_bts_columns(frame, required_columns=required)
    _validate_binary_indicator(clean["Cancelled"], "Cancelled")
    _validate_binary_indicator(clean["Diverted"], "Diverted")

    cancelled_mask = pd.to_numeric(clean["Cancelled"], errors="coerce").eq(1)
    after_cancelled = clean.loc[~cancelled_mask].copy()
    diverted_mask = pd.to_numeric(after_cancelled["Diverted"], errors="coerce").eq(1)
    after_diverted = after_cancelled.loc[~diverted_mask].copy()
    missing_target_mask = after_diverted["ArrDel15"].isna()
    eligible = after_diverted.loc[~missing_target_mask].copy()

    counts = ExclusionCounts(
        input_rows=len(clean),
        cancelled=int(cancelled_mask.sum()),
        diverted=int(diverted_mask.sum()),
        missing_target=int(missing_target_mask.sum()),
        eligible_rows=len(eligible),
    )
    if counts.input_rows != counts.excluded_rows + counts.eligible_rows:
        raise DataQualityError("eligibility counts do not reconcile")
    return EligibilityResult(frame=eligible, counts=counts)


def construct_binary_target(
    frame: pd.DataFrame,
    *,
    source_column: str = "ArrDel15",
    target_column: str = "target",
) -> pd.DataFrame:
    """Return a copy with an integer binary target constructed from ``ArrDel15``."""

    if source_column not in frame:
        raise DataQualityError(f"missing target source column: {source_column}")
    numeric = pd.to_numeric(frame[source_column], errors="coerce")
    if numeric.isna().any():
        bad_rows = list(frame.index[numeric.isna()])[:10]
        raise DataQualityError(f"{source_column} has missing/invalid values at rows {bad_rows}")
    invalid = sorted(set(numeric.unique()).difference({0, 1}))
    if invalid:
        raise DataQualityError(f"{source_column} must be binary; got {invalid}")
    result = frame.copy()
    result[target_column] = numeric.astype("int8")
    return result
