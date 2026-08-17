"""Prior-year seasonal historical state layered on the immutable v2 propensity state.

The v3 state *contains* a v2 :class:`~flight_delay.modeling.v2.features.HistoricalState` rather
than reimplementing it, so all 37 retained v2 features are produced by the exact v2 code path and
cannot drift. V3 adds only the five same-calendar-month tables and the deterministic seasonal
columns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.v2.features import (
    Counts,
    HistoricalState,
    V2FeatureError,
    _key,
    _normalized,
    build_historical_state,
    transform_one,
)
from flight_delay.modeling.v3.protocol import (
    SEASONAL_HISTORICAL_FEATURES,
    V3_FEATURES,
    V3_HISTORICAL_FEATURES,
    V3_SCHEDULE_FEATURES,
    canonical_sha256,
)
from flight_delay.modeling.v3.seasonal import (
    derive_seasonal_features,
    seasonal_features_for_date,
)

STATE_SCHEMA = "flight-delay-historical-state-v3"
MODEL_PERIOD_START = pd.Period("2024-02", freq="M")
MODEL_PERIOD_END_EXCLUSIVE = pd.Period("2025-12", freq="M")
SEASONAL_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "same_calendar_month_global": (),
    "same_calendar_month_carrier": ("Reporting_Airline",),
    "same_calendar_month_origin": ("Origin",),
    "same_calendar_month_destination": ("Dest",),
    "same_calendar_month_route": ("route",),
}
SEASONAL_FEATURE_BY_TABLE: dict[str, str] = {
    "same_calendar_month_global": "prior_same_calendar_month_global_delay_rate",
    "same_calendar_month_carrier": "prior_same_calendar_month_carrier_delay_rate",
    "same_calendar_month_origin": "prior_same_calendar_month_origin_delay_rate",
    "same_calendar_month_destination": "prior_same_calendar_month_destination_delay_rate",
    "same_calendar_month_route": "prior_same_calendar_month_route_delay_rate",
}


class V3FeatureError(ValueError):
    """Raised when v3 seasonal state or transformation could leak labels."""


@contextmanager
def _as_v3_error() -> Iterator[None]:
    """Re-raise the composed v2 state's refusals under the v3 error type.

    V3 reuses the v2 builder verbatim so the retained 37 features cannot drift, but a caller
    guarding v3 should only ever have to catch :class:`V3FeatureError`.
    """

    try:
        yield
    except V2FeatureError as error:
        raise V3FeatureError(str(error)) from error


@dataclass(frozen=True)
class V3HistoricalState:
    """Immutable v2 state plus same-calendar-month tables from strictly prior years."""

    base: HistoricalState
    seasonal_tables: dict[str, dict[str, Counts]]
    same_calendar_month_max_year: dict[int, int]

    @property
    def as_of(self) -> date:
        return self.base.as_of

    @property
    def global_rate(self) -> float:
        return self.base.global_rate

    @property
    def prior_strength(self) -> int:
        return self.base.prior_strength

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "base_state": self.base.as_dict(),
            "feature_schema": list(V3_FEATURES),
            "historical_feature_schema": list(V3_HISTORICAL_FEATURES),
            "seasonal_feature_schema": list(SEASONAL_HISTORICAL_FEATURES),
            "seasonal_tables": {
                name: [
                    {
                        "key": json.loads(key),
                        "count": counts.count,
                        "positive_count": counts.positive_count,
                    }
                    for key, counts in sorted(table.items())
                ]
                for name, table in sorted(self.seasonal_tables.items())
            },
            "same_calendar_month_max_year": {
                str(month): year
                for month, year in sorted(self.same_calendar_month_max_year.items())
            },
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def schema_sha256(self) -> str:
        return canonical_sha256(list(V3_FEATURES))

    @classmethod
    def from_bytes(cls, encoded: bytes) -> V3HistoricalState:
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V3FeatureError("v3 historical-state artifact is not valid JSON") from error
        if not isinstance(payload, dict) or payload.get("schema") != STATE_SCHEMA:
            raise V3FeatureError("v3 historical-state schema mismatch")
        if payload.get("feature_schema") != list(V3_FEATURES):
            raise V3FeatureError("v3 historical-state feature schema mismatch")
        if payload.get("historical_feature_schema") != list(V3_HISTORICAL_FEATURES):
            raise V3FeatureError("v3 historical-state derived feature schema mismatch")
        base_payload = payload.get("base_state")
        if not isinstance(base_payload, dict):
            raise V3FeatureError("v3 historical state requires its embedded v2 base state")
        try:
            base = HistoricalState.from_bytes(
                json.dumps(base_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        except V2FeatureError as error:
            raise V3FeatureError("embedded v2 base state is invalid") from error

        tables: dict[str, dict[str, Counts]] = {}
        encoded_tables = payload.get("seasonal_tables")
        if not isinstance(encoded_tables, dict) or set(encoded_tables) != set(
            SEASONAL_TABLE_COLUMNS
        ):
            raise V3FeatureError("v3 seasonal lookup table set mismatch")
        for name, entries in encoded_tables.items():
            if not isinstance(entries, list):
                raise V3FeatureError("v3 seasonal lookup entries must be a list")
            table: dict[str, Counts] = {}
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("key"), list):
                    raise V3FeatureError("v3 seasonal lookup entry is invalid")
                key = _key(*entry["key"])
                if key in table:
                    raise V3FeatureError("v3 seasonal lookup contains a duplicate key")
                counts = Counts(int(entry.get("count", -1)), int(entry.get("positive_count", -1)))
                if counts.count <= 0 or not 0 <= counts.positive_count <= counts.count:
                    raise V3FeatureError("v3 seasonal counts are invalid")
                table[key] = counts
            tables[str(name)] = table

        max_year = payload.get("same_calendar_month_max_year")
        if not isinstance(max_year, dict):
            raise V3FeatureError("v3 seasonal state requires its prior-year ledger")
        return cls(
            base=base,
            seasonal_tables=tables,
            same_calendar_month_max_year={int(k): int(v) for k, v in max_year.items()},
        )


@dataclass(frozen=True)
class V3TrainingTransform:
    features: pd.DataFrame
    target: pd.Series
    flight_date: pd.Series
    monthly_state_sha256: dict[str, str]


def _aggregate_seasonal(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Counts]:
    """Aggregate labels by calendar month plus the table's entity columns."""

    working = frame.copy(deep=False)
    working["__calendar_month"] = working["flight_date"].dt.month
    grouped = working.groupby(["__calendar_month", *columns], sort=True, dropna=False)[
        "target"
    ].agg(["count", "sum"])
    result: dict[str, Counts] = {}
    for index, row in grouped.iterrows():
        values = index if isinstance(index, tuple) else (index,)
        result[_key(*values)] = Counts(int(row["count"]), int(row["sum"]))
    return result


def build_v3_historical_state(history: pd.DataFrame, *, as_of: str | date) -> V3HistoricalState:
    """Build v3 state from full eligible prior history, refusing labels after the cutoff."""

    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    with _as_v3_error():
        base = build_historical_state(history, as_of=cutoff)
        normalized = _normalized(history, require_target=True)
    seasonal_tables = {
        name: _aggregate_seasonal(normalized, columns)
        for name, columns in SEASONAL_TABLE_COLUMNS.items()
    }
    max_year = (
        normalized.groupby(normalized["flight_date"].dt.month)["flight_date"]
        .max()
        .dt.year.to_dict()
    )
    return V3HistoricalState(
        base=base,
        seasonal_tables=seasonal_tables,
        same_calendar_month_max_year={int(month): int(year) for month, year in max_year.items()},
    )


def _smoothed(state: V3HistoricalState, counts: Counts | None) -> float:
    if counts is None:
        return state.global_rate
    return float(
        (counts.positive_count + state.prior_strength * state.global_rate)
        / (counts.count + state.prior_strength)
    )


def seasonal_historical_features(
    row: pd.Series | dict[str, Any], state: V3HistoricalState
) -> dict[str, float]:
    """Look up the five prior-year seasonal rates for one scheduled request."""

    values = dict(row)
    flight_date = pd.Timestamp(values["flight_date"])
    calendar_month = int(flight_date.month)
    contributing_year = state.same_calendar_month_max_year.get(calendar_month)
    if contributing_year is not None and contributing_year >= int(flight_date.year):
        raise V3FeatureError(
            "same-calendar-month state contains the model row's own year; only prior-year or "
            "previous occurrences may contribute"
        )
    entities: dict[str, tuple[Any, ...]] = {
        "same_calendar_month_global": (),
        "same_calendar_month_carrier": (values["Reporting_Airline"],),
        "same_calendar_month_origin": (values["Origin"],),
        "same_calendar_month_destination": (values["Dest"],),
        "same_calendar_month_route": (values["route"],),
    }
    return {
        SEASONAL_FEATURE_BY_TABLE[table]: _smoothed(
            state, state.seasonal_tables[table].get(_key(calendar_month, *keys))
        )
        for table, keys in entities.items()
    }


def transform_one_v3(row: pd.Series | dict[str, Any], state: V3HistoricalState) -> dict[str, float]:
    """Transform one scheduled request using only its fields and frozen state."""

    values = dict(row)
    flight_date = pd.Timestamp(values["flight_date"]).date()
    with _as_v3_error():
        base_features = transform_one(values, state.base)
    return {
        **{name: float(value) for name, value in seasonal_features_for_date(flight_date).items()},
        **base_features,
        **seasonal_historical_features(values, state),
    }


def _counts_table(table: dict[str, Counts]) -> pd.DataFrame:
    """Reshape a JSON-keyed lookup table into a frame indexed by its parsed key tuple.

    Keys are serialized as canonical JSON for deterministic hashing, but the batch path needs to
    align them against millions of rows, so they are parsed back into a pandas index here. The
    scalar types survive the round trip unchanged, which is what keeps the batch and single-row
    paths in exact agreement.
    """

    parsed = [json.loads(key) for key in table]
    values = [(counts.count, counts.positive_count) for counts in table.values()]
    if not parsed:
        return pd.DataFrame({"count": [], "positive": []}, dtype=float)
    arity = len(parsed[0])
    index: pd.Index = (
        pd.Index([row[0] for row in parsed])
        if arity == 1
        else pd.MultiIndex.from_tuples([tuple(row) for row in parsed])
    )
    return pd.DataFrame(values, columns=["count", "positive"], index=index)


def _target_index(frame: pd.DataFrame, columns: tuple[Any, ...]) -> pd.Index:
    if len(columns) == 1:
        return pd.Index(columns[0])
    return pd.MultiIndex.from_arrays(list(columns))


def _aligned_counts(table: dict[str, Counts], index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    aligned = _counts_table(table).reindex(index)
    return (
        aligned["count"].to_numpy(dtype=float),
        aligned["positive"].to_numpy(dtype=float),
    )


def _smoothed_vector(
    state: V3HistoricalState, count: np.ndarray, positive: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of :func:`_smoothed`, falling back to the global rate when unseen."""

    rate = state.global_rate
    strength = state.prior_strength
    with np.errstate(invalid="ignore"):
        smoothed = (positive + strength * rate) / (count + strength)
    return np.where(np.isnan(count), rate, smoothed)


def _base_historical_frame(
    normalized: pd.DataFrame, state: V3HistoricalState
) -> dict[str, np.ndarray]:
    """Vectorized reproduction of the v2 seventeen-feature lookup for a whole batch."""

    base = state.base
    carrier = normalized["Reporting_Airline"]
    origin = normalized["Origin"]
    destination = normalized["Dest"]
    route = normalized["route"]
    departure_hour = normalized["scheduled_departure_hour"]
    arrival_hour = normalized["scheduled_arrival_hour"]
    rows = len(normalized)

    def full(table: str, *columns: Any) -> tuple[np.ndarray, np.ndarray]:
        return _aligned_counts(base.full_tables[table], _target_index(normalized, columns))

    def recent(table: str, *columns: Any) -> tuple[np.ndarray, np.ndarray]:
        return _aligned_counts(base.recent_tables[table], _target_index(normalized, columns))

    route_count, route_positive = full("route", route)
    carrier_route_count, carrier_route_positive = full("carrier_route", carrier, route)
    recent_global = _smoothed_vector(
        state,
        np.full(rows, float(base.recent_global_counts.count)),
        np.full(rows, float(base.recent_global_counts.positive_count)),
    )
    derived: dict[str, np.ndarray] = {
        "prior_global_delay_rate": np.full(rows, base.global_rate),
        "prior_carrier_delay_rate": _smoothed_vector(state, *full("carrier", carrier)),
        "prior_origin_delay_rate": _smoothed_vector(state, *full("origin", origin)),
        "prior_destination_delay_rate": _smoothed_vector(state, *full("destination", destination)),
        "prior_route_delay_rate": _smoothed_vector(state, route_count, route_positive),
        "prior_carrier_route_delay_rate": _smoothed_vector(
            state, carrier_route_count, carrier_route_positive
        ),
        "prior_carrier_origin_delay_rate": _smoothed_vector(
            state, *full("carrier_origin", carrier, origin)
        ),
        "prior_carrier_destination_delay_rate": _smoothed_vector(
            state, *full("carrier_destination", carrier, destination)
        ),
        "prior_origin_departure_hour_delay_rate": _smoothed_vector(
            state, *full("origin_departure_hour", origin, departure_hour)
        ),
        "prior_destination_arrival_hour_delay_rate": _smoothed_vector(
            state, *full("destination_arrival_hour", destination, arrival_hour)
        ),
        "log_route_support": np.log1p(np.nan_to_num(route_count, nan=0.0)),
        "log_carrier_route_support": np.log1p(np.nan_to_num(carrier_route_count, nan=0.0)),
        "recent_global_delay_rate_3m": recent_global,
        "recent_carrier_delay_rate_3m": _smoothed_vector(state, *recent("carrier", carrier)),
        "recent_origin_delay_rate_3m": _smoothed_vector(state, *recent("origin", origin)),
        "recent_destination_delay_rate_3m": _smoothed_vector(
            state, *recent("destination", destination)
        ),
        "recent_route_delay_rate_3m": _smoothed_vector(state, *recent("route", route)),
    }
    return derived


def _seasonal_historical_frame(
    normalized: pd.DataFrame, state: V3HistoricalState
) -> dict[str, np.ndarray]:
    """Vectorized same-calendar-month lookup with the prior-year invariant enforced."""

    calendar_month = normalized["flight_date"].dt.month.astype(int)
    years = normalized["flight_date"].dt.year.astype(int)
    for month, year in zip(calendar_month, years, strict=True):
        contributing = state.same_calendar_month_max_year.get(int(month))
        if contributing is not None and contributing >= int(year):
            raise V3FeatureError(
                "same-calendar-month state contains the model row's own year; only prior-year or "
                "previous occurrences may contribute"
            )
    columns: dict[str, tuple[Any, ...]] = {
        "same_calendar_month_global": (calendar_month,),
        "same_calendar_month_carrier": (calendar_month, normalized["Reporting_Airline"]),
        "same_calendar_month_origin": (calendar_month, normalized["Origin"]),
        "same_calendar_month_destination": (calendar_month, normalized["Dest"]),
        "same_calendar_month_route": (calendar_month, normalized["route"]),
    }
    return {
        SEASONAL_FEATURE_BY_TABLE[table]: _smoothed_vector(
            state, *_aligned_counts(state.seasonal_tables[table], _target_index(normalized, keys))
        )
        for table, keys in columns.items()
    }


def transform_with_v3_state(rows: pd.DataFrame, state: V3HistoricalState) -> pd.DataFrame:
    """Apply the serving transformer to a batch and retain the exact 48-feature order.

    This is the vectorized twin of :func:`transform_one_v3`; ``test_v3_features`` asserts the two
    agree exactly, which is the training-serving parity requirement.
    """

    validate_model_features(V3_FEATURES)
    with _as_v3_error():
        normalized = _normalized(rows, require_target=False)
    month_starts = normalized["flight_date"].dt.to_period("M").dt.start_time.dt.date
    if (month_starts <= state.as_of).any():
        raise V3FeatureError("feature state must end before every model-row month")
    seasonal = derive_seasonal_features(normalized["flight_date"])
    derived = pd.DataFrame(
        {
            **_base_historical_frame(normalized, state),
            **_seasonal_historical_frame(normalized, state),
        },
        index=normalized.index,
    )
    schedule = pd.concat([normalized.loc[:, V3_SCHEDULE_FEATURES[:20]], seasonal], axis=1)
    result = pd.concat([schedule, derived.loc[:, list(V3_HISTORICAL_FEATURES)]], axis=1)
    if tuple(result.columns) != V3_FEATURES:
        raise V3FeatureError("v3 transformed feature schema drifted")
    if not np.isfinite(result.loc[:, V3_HISTORICAL_FEATURES].to_numpy(dtype=float)).all():
        raise V3FeatureError("v3 historical features must be finite")
    return result


def transform_v3_training_rows(
    full_history: pd.DataFrame, model_rows: pd.DataFrame
) -> V3TrainingTransform:
    """Transform model rows month by month from full eligible prior history."""

    with _as_v3_error():
        history = _normalized(full_history, require_target=True)
        rows = _normalized(model_rows, require_target=True)
    model_months = rows["flight_date"].dt.to_period("M")
    if model_months.min() < MODEL_PERIOD_START:
        raise V3FeatureError("January 2024 is burn-in and cannot contribute model rows")
    if model_months.max() >= MODEL_PERIOD_END_EXCLUSIVE:
        raise V3FeatureError("v3 development transformation stops before December 2025")
    transformed: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    dates: list[pd.Series] = []
    digests: dict[str, str] = {}
    for month, monthly_rows in rows.groupby(model_months, sort=True):
        month_start = month.start_time
        cutoff = (month_start - pd.Timedelta(days=1)).date()
        eligible_history = history.loc[history["flight_date"].lt(month_start)]
        state = build_v3_historical_state(eligible_history, as_of=cutoff)
        transformed.append(transform_with_v3_state(monthly_rows, state))
        targets.append(monthly_rows["target"].astype(int))
        dates.append(monthly_rows["flight_date"])
        digests[str(month)] = state.sha256
    features = pd.concat(transformed).sort_index()
    target = pd.concat(targets).loc[features.index]
    flight_date = pd.concat(dates).loc[features.index]
    return V3TrainingTransform(features, target, flight_date, digests)
