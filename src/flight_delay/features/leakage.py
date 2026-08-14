"""Central model-feature allowlist and leakage guard."""

from collections.abc import Iterable

ALLOWED_MODEL_FEATURES: frozenset[str] = frozenset(
    {
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
        "route",
        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_departure_minute_bucket",
        "scheduled_arrival_minute_bucket",
        "is_weekend",
        "scheduled_departure_sin",
        "scheduled_departure_cos",
        "scheduled_arrival_sin",
        "scheduled_arrival_cos",
        "month_sin",
        "month_cos",
        "prior_global_delay_rate",
        "prior_carrier_delay_rate",
        "prior_origin_delay_rate",
        "prior_destination_delay_rate",
        "prior_route_delay_rate",
        "prior_carrier_route_delay_rate",
        "prior_carrier_origin_delay_rate",
        "prior_carrier_destination_delay_rate",
        "prior_origin_departure_hour_delay_rate",
        "prior_destination_arrival_hour_delay_rate",
        "log_route_support",
        "log_carrier_route_support",
        "recent_global_delay_rate_3m",
        "recent_carrier_delay_rate_3m",
        "recent_origin_delay_rate_3m",
        "recent_destination_delay_rate_3m",
        "recent_route_delay_rate_3m",
    }
)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "DepTime",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "DepartureDelayGroups",
        "TaxiOut",
        "WheelsOff",
        "AirTime",
        "WheelsOn",
        "TaxiIn",
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrDel15",
        "ArrivalDelayGroups",
        "Cancelled",
        "CancellationCode",
        "Diverted",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "FirstDepTime",
        "TotalAddGTime",
        "LongestAddGTime",
        "DivAirportLandings",
        "DivReachedDest",
        "DivActualElapsedTime",
        "DivArrDelay",
        "DivDistance",
    }
)


class FeatureLeakageError(ValueError):
    """Raised when a model schema contains forbidden or unapproved fields."""

    def __init__(
        self,
        *,
        forbidden: frozenset[str] = frozenset(),
        unapproved: frozenset[str] = frozenset(),
    ) -> None:
        self.forbidden = forbidden
        self.unapproved = unapproved
        details: list[str] = []
        if forbidden:
            details.append(f"forbidden post-outcome features: {sorted(forbidden)}")
        if unapproved:
            details.append(f"features not in ALLOWED_MODEL_FEATURES: {sorted(unapproved)}")
        super().__init__("Unsafe model feature schema; " + "; ".join(details))


def _normalized_lookup(features: frozenset[str]) -> dict[str, str]:
    return {_comparison_key(feature): feature for feature in features}


def _comparison_key(feature: str) -> str:
    return "".join(character for character in feature.casefold() if character.isalnum())


def validate_model_features(proposed_features: Iterable[str]) -> frozenset[str]:
    """Validate and return a model schema, rejecting leakage and unknown fields.

    Matching is case- and punctuation-insensitive, so aliases such as ``dep_delay``
    cannot bypass the central guard.
    """

    proposed = frozenset(proposed_features)
    if not proposed:
        raise FeatureLeakageError(unapproved=frozenset({"<empty schema>"}))
    if any(not isinstance(feature, str) or not feature.strip() for feature in proposed):
        raise FeatureLeakageError(unapproved=frozenset({"<invalid feature name>"}))

    forbidden_lookup = _normalized_lookup(FORBIDDEN_FEATURES)
    allowed_lookup = _normalized_lookup(ALLOWED_MODEL_FEATURES)
    forbidden = frozenset(
        forbidden_lookup[_comparison_key(feature)]
        for feature in proposed
        if _comparison_key(feature) in forbidden_lookup
    )
    unapproved = frozenset(
        feature
        for feature in proposed
        if _comparison_key(feature) not in allowed_lookup
        and _comparison_key(feature) not in forbidden_lookup
    )
    if forbidden or unapproved:
        raise FeatureLeakageError(forbidden=forbidden, unapproved=unapproved)
    return proposed
