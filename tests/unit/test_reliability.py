import pandas as pd
import pytest

from flight_delay.data.preprocessing import DataQualityError
from flight_delay.data.reliability import compute_route_reliability


def test_route_reliability_math_and_minimum_support() -> None:
    frame = pd.DataFrame(
        {
            "Reporting_Airline": ["UA", "UA", "WN", "UA"],
            "Origin": ["DEN", "DEN", "DEN", "DEN"],
            "Dest": ["LAX", "LAX", "LAX", "SEA"],
            "ArrDel15": [0, 1, 0, 1],
            "ArrDelay": [-5, 20, 5, 30],
        }
    )
    result = compute_route_reliability(frame, min_support=3)
    ua_lax = result[
        result["scope"].eq("carrier_route")
        & result["Reporting_Airline"].eq("UA")
        & result["route"].eq("DEN-LAX")
    ].iloc[0]
    assert ua_lax["eligible_flights"] == 2
    assert ua_lax["delayed_count"] == 1
    assert ua_lax["on_time_rate"] == pytest.approx(0.5)
    assert bool(ua_lax["meets_minimum_support"]) is False

    all_lax = result[result["scope"].eq("all_carriers") & result["route"].eq("DEN-LAX")].iloc[0]
    assert all_lax["eligible_flights"] == 3
    assert all_lax["delayed_rate"] == pytest.approx(1 / 3)
    assert all_lax["mean_arrival_delay_minutes"] == pytest.approx(20 / 3)
    assert bool(all_lax["meets_minimum_support"]) is True


def test_route_reliability_rejects_invalid_arrival_delay() -> None:
    frame = pd.DataFrame(
        {
            "Reporting_Airline": ["UA"],
            "Origin": ["DEN"],
            "Dest": ["LAX"],
            "ArrDel15": [0],
            "ArrDelay": ["unknown"],
        }
    )
    with pytest.raises(DataQualityError, match="ArrDelay has invalid"):
        compute_route_reliability(frame)


def test_categorical_inputs_do_not_emit_unobserved_route_combinations() -> None:
    frame = pd.DataFrame(
        {
            "Reporting_Airline": pd.Categorical(["UA", "WN"]),
            "Origin": pd.Categorical(["DEN", "SFO"]),
            "Dest": pd.Categorical(["LAX", "SEA"]),
            "ArrDel15": [0, 1],
            "ArrDelay": [-5, 30],
        }
    )
    result = compute_route_reliability(frame)
    assert len(result[result["scope"].eq("carrier_route")]) == 2
    assert len(result[result["scope"].eq("all_carriers")]) == 2
