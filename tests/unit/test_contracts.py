from datetime import date, time

import pytest
from pydantic import ValidationError

from flight_delay.contracts import FeedbackRequest, FlightPredictionRequest, TrafficSource


def valid_request() -> dict[str, object]:
    return {
        "carrier": " ua ",
        "origin": "den",
        "destination": "lax",
        "flight_date": date(2026, 8, 18),
        "scheduled_departure": time(7, 30),
        "scheduled_arrival": time(9, 0),
        "scheduled_elapsed_minutes": 150,
        "distance_miles": 862.0,
    }


def test_prediction_request_normalizes_codes() -> None:
    request = FlightPredictionRequest.model_validate(valid_request())
    assert request.carrier == "UA"
    assert request.origin == "DEN"
    assert request.destination == "LAX"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("carrier", "UNITED"),
        ("origin", "D3N"),
        ("destination", "DEN"),
        ("scheduled_elapsed_minutes", 0),
        ("distance_miles", -1),
    ],
)
def test_prediction_request_rejects_invalid_input(field: str, value: object) -> None:
    payload = valid_request()
    payload[field] = value
    with pytest.raises(ValidationError):
        FlightPredictionRequest.model_validate(payload)


def test_feedback_contract_rejects_unknown_and_extreme_values() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(actual_delayed=True, arrival_delay_minutes=5_000)
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate({"actual_delayed": False, "unknown": "nope"})


def test_traffic_source_contract_is_closed_and_not_a_model_request_fact() -> None:
    assert [source.value for source in TrafficSource] == [
        "traveler_ui",
        "synthetic_load_test",
        "api_unspecified",
        "legacy_unattributed",
    ]
    assert "traffic_source" not in FlightPredictionRequest.model_fields
    with pytest.raises(ValueError):
        TrafficSource("uncontrolled")
