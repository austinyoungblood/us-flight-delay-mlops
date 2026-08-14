"""Prior-month-only historical propensity state and serving-parity transforms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from flight_delay.features.leakage import FORBIDDEN_FEATURES, validate_model_features
from flight_delay.modeling.v2.protocol import HISTORICAL_FEATURES, V2_FEATURES

STATE_SCHEMA = "flight-delay-historical-state-v1"
PRIOR_STRENGTH = 50
TRAILING_MONTHS = 3
STATE_COLUMNS: tuple[str, ...] = (
    "flight_date",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "route",
    "scheduled_departure_hour",
    "scheduled_arrival_hour",
    "target",
)
FULL_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "carrier": ("Reporting_Airline",),
    "origin": ("Origin",),
    "destination": ("Dest",),
    "route": ("route",),
    "carrier_route": ("Reporting_Airline", "route"),
    "carrier_origin": ("Reporting_Airline", "Origin"),
    "carrier_destination": ("Reporting_Airline", "Dest"),
    "origin_departure_hour": ("Origin", "scheduled_departure_hour"),
    "destination_arrival_hour": ("Dest", "scheduled_arrival_hour"),
}
RECENT_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    name: FULL_TABLE_COLUMNS[name] for name in ("carrier", "origin", "destination", "route")
}


class V2FeatureError(ValueError):
    """Raised when historical state or transformation could leak labels."""


@dataclass(frozen=True)
class Counts:
    count: int
    positive_count: int

    @property
    def rate(self) -> float:
        if self.count <= 0:
            raise V2FeatureError("historical count must be positive")
        return self.positive_count / self.count


@dataclass(frozen=True)
class HistoricalState:
    """Immutable lookup state built from labels no later than ``as_of``."""

    as_of: date
    global_counts: Counts
    recent_global_counts: Counts
    full_tables: dict[str, dict[str, Counts]]
    recent_tables: dict[str, dict[str, Counts]]
    prior_strength: int = PRIOR_STRENGTH

    @property
    def global_rate(self) -> float:
        return self.global_counts.rate

    def _serialized_tables(
        self, tables: dict[str, dict[str, Counts]]
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [
                {
                    "key": json.loads(key),
                    "count": counts.count,
                    "positive_count": counts.positive_count,
                }
                for key, counts in sorted(table.items())
            ]
            for name, table in sorted(tables.items())
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "as_of": self.as_of.isoformat(),
            "prior_strength": self.prior_strength,
            "trailing_months": TRAILING_MONTHS,
            "feature_schema": list(V2_FEATURES),
            "historical_feature_schema": list(HISTORICAL_FEATURES),
            "global_counts": {
                "count": self.global_counts.count,
                "positive_count": self.global_counts.positive_count,
            },
            "recent_global_counts": {
                "count": self.recent_global_counts.count,
                "positive_count": self.recent_global_counts.positive_count,
            },
            "full_tables": self._serialized_tables(self.full_tables),
            "recent_tables": self._serialized_tables(self.recent_tables),
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
        payload = json.dumps(list(V2_FEATURES), separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_bytes(cls, encoded: bytes) -> HistoricalState:
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V2FeatureError("historical-state artifact is not valid JSON") from error
        if not isinstance(payload, dict) or payload.get("schema") != STATE_SCHEMA:
            raise V2FeatureError("historical-state schema mismatch")
        if payload.get("feature_schema") != list(V2_FEATURES):
            raise V2FeatureError("historical-state feature schema mismatch")
        if payload.get("historical_feature_schema") != list(HISTORICAL_FEATURES):
            raise V2FeatureError("historical-state derived feature schema mismatch")
        if payload.get("prior_strength") != PRIOR_STRENGTH:
            raise V2FeatureError("historical-state prior strength mismatch")

        def counts(value: object) -> Counts:
            if not isinstance(value, dict):
                raise V2FeatureError("historical counts must be a mapping")
            result = Counts(int(value.get("count", -1)), int(value.get("positive_count", -1)))
            if result.count <= 0 or not 0 <= result.positive_count <= result.count:
                raise V2FeatureError("historical counts are invalid")
            return result

        def tables(value: object, expected: set[str]) -> dict[str, dict[str, Counts]]:
            if not isinstance(value, dict) or set(value) != expected:
                raise V2FeatureError("historical lookup table set mismatch")
            result: dict[str, dict[str, Counts]] = {}
            for name, entries in value.items():
                if not isinstance(entries, list):
                    raise V2FeatureError("historical lookup entries must be a list")
                table: dict[str, Counts] = {}
                for entry in entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get("key"), list):
                        raise V2FeatureError("historical lookup entry is invalid")
                    key = _key(*entry["key"])
                    if key in table:
                        raise V2FeatureError("historical lookup contains a duplicate key")
                    table[key] = counts(entry)
                result[str(name)] = table
            return result

        try:
            as_of = date.fromisoformat(str(payload["as_of"]))
        except (KeyError, ValueError) as error:
            raise V2FeatureError("historical-state as-of date is invalid") from error
        return cls(
            as_of=as_of,
            global_counts=counts(payload.get("global_counts")),
            recent_global_counts=counts(payload.get("recent_global_counts")),
            full_tables=tables(payload.get("full_tables"), set(FULL_TABLE_COLUMNS)),
            recent_tables=tables(payload.get("recent_tables"), set(RECENT_TABLE_COLUMNS)),
        )


@dataclass(frozen=True)
class TrainingTransform:
    features: pd.DataFrame
    target: pd.Series
    flight_date: pd.Series
    monthly_state_sha256: dict[str, str]


def _scalar(value: Any) -> str | int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value) if isinstance(value, str) else value


def _key(*values: Any) -> str:
    return json.dumps(
        [_scalar(value) for value in values],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalized(frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    required = set(STATE_COLUMNS if require_target else STATE_COLUMNS[:-1])
    missing = required - set(frame)
    if missing:
        raise V2FeatureError(f"historical transformation is missing columns: {sorted(missing)}")
    forbidden = set(frame) & FORBIDDEN_FEATURES
    if forbidden:
        raise V2FeatureError(f"post-outcome features are prohibited: {sorted(forbidden)}")
    result = frame.copy(deep=True)
    dates = pd.to_datetime(result["flight_date"], errors="coerce").dt.normalize()
    if result.empty or dates.isna().any():
        raise V2FeatureError("historical transformation requires valid non-empty flight dates")
    result["flight_date"] = dates
    for column in ("Reporting_Airline", "Origin", "Dest", "route"):
        values = result[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise V2FeatureError(f"historical key {column} must be non-empty")
        result[column] = values.astype(str)
    for column in ("scheduled_departure_hour", "scheduled_arrival_hour"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not values.between(0, 23).all():
            raise V2FeatureError(f"historical key {column} must be an hour from 0 through 23")
        result[column] = values.astype(int)
    if require_target:
        target = pd.to_numeric(result["target"], errors="coerce")
        if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
            raise V2FeatureError("historical target must be binary")
        result["target"] = target.astype(int)
    return result.sort_values("flight_date", kind="stable")


def _counts(frame: pd.DataFrame) -> Counts:
    result = Counts(len(frame), int(frame["target"].sum()))
    if result.count <= 0:
        raise V2FeatureError("historical state cannot be built from empty history")
    return result


def _aggregate(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Counts]:
    result: dict[str, Counts] = {}
    grouped = frame.groupby(list(columns), sort=True, dropna=False)["target"].agg(["count", "sum"])
    for index, row in grouped.iterrows():
        values = index if isinstance(index, tuple) else (index,)
        result[_key(*values)] = Counts(int(row["count"]), int(row["sum"]))
    return result


def build_historical_state(history: pd.DataFrame, *, as_of: str | date) -> HistoricalState:
    """Build deterministic state while refusing any label after the declared cutoff."""

    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    normalized = _normalized(history, require_target=True)
    dates = normalized["flight_date"].dt.date
    if dates.max() > cutoff:
        raise V2FeatureError("history contains labels after the state cutoff")
    recent_start = (pd.Period(cutoff, freq="M") - (TRAILING_MONTHS - 1)).start_time.date()
    recent = normalized.loc[dates.ge(recent_start)]
    if recent.empty:
        raise V2FeatureError("historical state has no rows in its trailing three-month window")
    return HistoricalState(
        as_of=cutoff,
        global_counts=_counts(normalized),
        recent_global_counts=_counts(recent),
        full_tables={
            name: _aggregate(normalized, columns) for name, columns in FULL_TABLE_COLUMNS.items()
        },
        recent_tables={
            name: _aggregate(recent, columns) for name, columns in RECENT_TABLE_COLUMNS.items()
        },
    )


def _smoothed(state: HistoricalState, counts: Counts | None) -> float:
    if counts is None:
        return state.global_rate
    return float(
        (counts.positive_count + state.prior_strength * state.global_rate)
        / (counts.count + state.prior_strength)
    )


def _lookup(
    state: HistoricalState, table: str, values: tuple[Any, ...], *, recent: bool = False
) -> Counts | None:
    tables = state.recent_tables if recent else state.full_tables
    return tables[table].get(_key(*values))


def transform_one(row: pd.Series | dict[str, Any], state: HistoricalState) -> dict[str, float]:
    """Transform one scheduled request using only its fields and frozen state."""

    values = dict(row)
    carrier = values["Reporting_Airline"]
    origin = values["Origin"]
    destination = values["Dest"]
    route = values["route"]
    departure_hour = values["scheduled_departure_hour"]
    arrival_hour = values["scheduled_arrival_hour"]
    route_counts = _lookup(state, "route", (route,))
    carrier_route_counts = _lookup(state, "carrier_route", (carrier, route))
    return {
        "prior_global_delay_rate": state.global_rate,
        "prior_carrier_delay_rate": _smoothed(state, _lookup(state, "carrier", (carrier,))),
        "prior_origin_delay_rate": _smoothed(state, _lookup(state, "origin", (origin,))),
        "prior_destination_delay_rate": _smoothed(
            state, _lookup(state, "destination", (destination,))
        ),
        "prior_route_delay_rate": _smoothed(state, route_counts),
        "prior_carrier_route_delay_rate": _smoothed(state, carrier_route_counts),
        "prior_carrier_origin_delay_rate": _smoothed(
            state, _lookup(state, "carrier_origin", (carrier, origin))
        ),
        "prior_carrier_destination_delay_rate": _smoothed(
            state, _lookup(state, "carrier_destination", (carrier, destination))
        ),
        "prior_origin_departure_hour_delay_rate": _smoothed(
            state, _lookup(state, "origin_departure_hour", (origin, departure_hour))
        ),
        "prior_destination_arrival_hour_delay_rate": _smoothed(
            state,
            _lookup(state, "destination_arrival_hour", (destination, arrival_hour)),
        ),
        "log_route_support": math.log1p(route_counts.count if route_counts else 0),
        "log_carrier_route_support": math.log1p(
            carrier_route_counts.count if carrier_route_counts else 0
        ),
        "recent_global_delay_rate_3m": _smoothed(state, state.recent_global_counts),
        "recent_carrier_delay_rate_3m": _smoothed(
            state, _lookup(state, "carrier", (carrier,), recent=True)
        ),
        "recent_origin_delay_rate_3m": _smoothed(
            state, _lookup(state, "origin", (origin,), recent=True)
        ),
        "recent_destination_delay_rate_3m": _smoothed(
            state, _lookup(state, "destination", (destination,), recent=True)
        ),
        "recent_route_delay_rate_3m": _smoothed(
            state, _lookup(state, "route", (route,), recent=True)
        ),
    }


def transform_with_state(rows: pd.DataFrame, state: HistoricalState) -> pd.DataFrame:
    """Apply the serving transformer to a batch and retain the exact 37-feature order."""

    validate_model_features(V2_FEATURES)
    normalized = _normalized(rows, require_target=False)
    for timestamp in normalized["flight_date"]:
        month_start = timestamp.to_period("M").start_time.date()
        if state.as_of >= month_start:
            raise V2FeatureError("feature state must end before every model-row month")
    derived = pd.DataFrame(
        [transform_one(row, state) for _, row in normalized.iterrows()], index=normalized.index
    )
    result = pd.concat([normalized.loc[:, V2_FEATURES[:20]], derived], axis=1)
    if tuple(result.columns) != V2_FEATURES:
        raise V2FeatureError("v2 transformed feature schema drifted")
    if not np.isfinite(result.loc[:, HISTORICAL_FEATURES].to_numpy(dtype=float)).all():
        raise V2FeatureError("historical features must be finite")
    return result


def transform_training_rows(
    full_history: pd.DataFrame, model_rows: pd.DataFrame
) -> TrainingTransform:
    """Transform February-October model rows from full eligible prior-month history."""

    history = _normalized(full_history, require_target=True)
    rows = _normalized(model_rows, require_target=True)
    model_months = rows["flight_date"].dt.to_period("M")
    if model_months.min() < pd.Period("2025-02", freq="M"):
        raise V2FeatureError("January 2025 is burn-in and cannot contribute model rows")
    if model_months.max() >= pd.Period("2025-11", freq="M"):
        raise V2FeatureError("development model transformation stops before November")
    transformed: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    dates: list[pd.Series] = []
    digests: dict[str, str] = {}
    for month, monthly_rows in rows.groupby(model_months, sort=True):
        month_start = month.start_time
        cutoff = (month_start - pd.Timedelta(days=1)).date()
        eligible_history = history.loc[history["flight_date"].lt(month_start)]
        state = build_historical_state(eligible_history, as_of=cutoff)
        transformed.append(transform_with_state(monthly_rows, state))
        targets.append(monthly_rows["target"].astype(int))
        dates.append(monthly_rows["flight_date"])
        digests[str(month)] = state.sha256
    features = pd.concat(transformed).sort_index()
    target = pd.concat(targets).loc[features.index]
    flight_date = pd.concat(dates).loc[features.index]
    return TrainingTransform(features, target, flight_date, digests)
