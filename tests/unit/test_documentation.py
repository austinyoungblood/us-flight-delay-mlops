from __future__ import annotations

import re
from pathlib import Path

from flight_delay.contracts import FlightPredictionRequest, FlightPredictionResponse

ROOT = Path(__file__).resolve().parents[2]


def test_readme_worked_prediction_example_matches_public_contracts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Worked prediction example", maxsplit=1)[1].split(
        "Interactive OpenAPI documentation", maxsplit=1
    )[0]
    request_match = re.search(r"--data '(\{.*?\})'", section, flags=re.DOTALL)
    response_match = re.search(r"```json\n(\{.*?\})\n```", section, flags=re.DOTALL)
    assert request_match is not None
    assert response_match is not None

    request = FlightPredictionRequest.model_validate_json(request_match.group(1))
    response = FlightPredictionResponse.model_validate_json(response_match.group(1))
    assert (request.carrier, request.origin, request.destination) == ("UA", "DEN", "LAX")
    assert response.predicted_delayed is True
    assert response.delay_probability == 0.20431692609038393
    assert response.classification_threshold == 0.1840285229739868
    assert response.delay_probability < 0.5
    assert response.delay_probability >= response.classification_threshold
    assert "$API_BASE_URL/predict" in section
