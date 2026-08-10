"""Historical carrier-route and all-carrier reliability aggregation."""

import pandas as pd

from flight_delay.data.preprocessing import DataQualityError, normalize_bts_columns


def _summarize(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    min_support: int,
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    summary = grouped["ArrDel15"].agg(eligible_flights="size", delayed_count="sum").reset_index()
    summary["delayed_count"] = summary["delayed_count"].astype(int)
    summary["eligible_flights"] = summary["eligible_flights"].astype(int)
    summary["on_time_count"] = summary["eligible_flights"] - summary["delayed_count"]
    summary["delayed_rate"] = summary["delayed_count"] / summary["eligible_flights"]
    summary["on_time_rate"] = summary["on_time_count"] / summary["eligible_flights"]
    if "ArrDelay" in frame:
        delays = grouped["ArrDelay"].agg(
            mean_arrival_delay_minutes="mean", median_arrival_delay_minutes="median"
        )
        summary = summary.merge(delays.reset_index(), on=group_columns, how="left")
    else:
        summary["mean_arrival_delay_minutes"] = pd.NA
        summary["median_arrival_delay_minutes"] = pd.NA
    summary["meets_minimum_support"] = summary["eligible_flights"].ge(min_support)
    summary["scope"] = scope
    summary["route"] = summary["Origin"].astype(str) + "-" + summary["Dest"].astype(str)
    return summary


def compute_route_reliability(frame: pd.DataFrame, *, min_support: int = 30) -> pd.DataFrame:
    """Aggregate eligible completed flights at carrier-route and route levels."""

    if min_support <= 0:
        raise DataQualityError("min_support must be positive")
    required = frozenset({"Reporting_Airline", "Origin", "Dest", "ArrDel15"})
    clean = normalize_bts_columns(frame, required_columns=required)
    target = pd.to_numeric(clean["ArrDel15"], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise DataQualityError("ArrDel15 must contain only eligible binary target values")
    clean["ArrDel15"] = target.astype(int)
    if "ArrDelay" in clean:
        arrival_delay = pd.to_numeric(clean["ArrDelay"], errors="coerce")
        invalid_delay = clean["ArrDelay"].notna() & arrival_delay.isna()
        if invalid_delay.any():
            raise DataQualityError(
                f"ArrDelay has invalid values at rows {list(clean.index[invalid_delay])[:10]}"
            )
        clean["ArrDelay"] = arrival_delay

    carrier = _summarize(
        clean,
        group_columns=["Reporting_Airline", "Origin", "Dest"],
        scope="carrier_route",
        min_support=min_support,
    )
    all_carriers = _summarize(
        clean,
        group_columns=["Origin", "Dest"],
        scope="all_carriers",
        min_support=min_support,
    )
    all_carriers.insert(0, "Reporting_Airline", pd.NA)
    columns = [
        "scope",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "route",
        "eligible_flights",
        "on_time_count",
        "on_time_rate",
        "delayed_count",
        "delayed_rate",
        "mean_arrival_delay_minutes",
        "median_arrival_delay_minutes",
        "meets_minimum_support",
    ]
    return pd.concat([carrier[columns], all_carriers[columns]], ignore_index=True)
