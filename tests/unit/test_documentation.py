from __future__ import annotations

import re
from pathlib import Path

from flight_delay.contracts import FlightPredictionRequest

ROOT = Path(__file__).resolve().parents[2]


def test_readme_prediction_example_and_api_surface_match_public_contracts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Worked prediction example", maxsplit=1)[1].split(
        "Interactive OpenAPI documentation", maxsplit=1
    )[0]
    request_match = re.search(r"--data '(\{.*?\})'", section, flags=re.DOTALL)
    assert request_match is not None

    request = FlightPredictionRequest.model_validate_json(request_match.group(1))
    assert (request.carrier, request.origin, request.destination) == ("UA", "DEN", "LAX")
    assert "$API_BASE_URL/predict" in section
    assert "0.1840285229739868" in section
    assert "does not mean the probability exceeds 50%" in section
    assert "only after persistence" in section
    assert "persisted prediction identifier" in section

    for endpoint in (
        "GET /health",
        "GET /model-info",
        "POST /predict",
        "GET /route-reliability",
        "GET /predictions/{prediction_id}",
        "POST /feedback/{prediction_id}",
    ):
        assert f"`{endpoint}`" in readme

    assert '"prediction_id"' not in section
    assert '"created_at"' not in section
    assert '"inference_latency_ms"' not in section
