from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from flight_delay.contracts import FlightPredictionResponse, TrafficSource
from flight_delay.load_testing.traffic import (
    MAX_RATE_PER_SECOND,
    MAX_REQUEST_COUNT,
    TrafficPlan,
    generate_requests,
    run_monitoring_traffic,
    validate_api_base_url,
    write_audit_summary,
)
from flight_delay.ui import ApiClientError

ROOT = Path(__file__).resolve().parents[2]


def _aws_sdk_forbidden_environment(tmp_path: Path) -> dict[str, str]:
    """Install an import guard before application imports and omit AWS configuration."""

    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import sys

            FORBIDDEN_ROOTS = {"boto3", "botocore"}
            preloaded = sorted(
                name for name in sys.modules if name.partition(".")[0] in FORBIDDEN_ROOTS
            )
            if preloaded:
                raise RuntimeError(f"AWS SDK was preloaded: {preloaded}")

            class DenyAwsSdkImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.partition(".")[0] in FORBIDDEN_ROOTS:
                        raise RuntimeError(f"AWS SDK import attempted: {fullname}")
                    return None

            sys.meta_path.insert(0, DenyAwsSdkImports())
            """
        ).lstrip(),
        encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(ROOT / "src")))
    return environment


def response(prediction_id: str) -> FlightPredictionResponse:
    return FlightPredictionResponse(
        prediction_id=prediction_id,
        delay_probability=0.2,
        predicted_delayed=True,
        risk_band="medium",
        classification_threshold=0.1840285229739868,
        model_alias="production",
        model_version="v0",
        model_digest="digest",
        cache_hit=False,
        latency_ms=10,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def test_generation_is_deterministic_varied_and_contract_valid() -> None:
    plan = TrafficPlan(count=20, seed=7, start_date=date(2026, 8, 14))
    first = generate_requests(plan)
    second = generate_requests(plan)
    assert first == second
    assert len(first) == 20
    assert len({(item.carrier, item.origin, item.destination) for item in first}) > 1
    assert all(item.origin != item.destination for item in first)
    assert all(date(2026, 8, 14) <= item.flight_date <= date(2026, 8, 27) for item in first)


@pytest.mark.parametrize("count", [0, MAX_REQUEST_COUNT + 1, True])
def test_count_bounds_are_enforced(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        TrafficPlan(count=count)


@pytest.mark.parametrize("rate", [0, -1, float("inf"), MAX_RATE_PER_SECOND + 0.1, True])
def test_rate_bounds_are_enforced(rate: float) -> None:
    with pytest.raises(ValueError, match="rate_per_second"):
        TrafficPlan(rate_per_second=rate)


@pytest.mark.parametrize("seed", [True, "42"])
def test_seed_must_be_an_integer(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        TrafficPlan(seed=seed)


def test_start_date_must_be_a_date() -> None:
    with pytest.raises(ValueError, match="start_date"):
        TrafficPlan(start_date="2026-08-14")


def test_dry_run_never_calls_sender() -> None:
    def unexpected_sender(
        request: object, *, traffic_source: TrafficSource
    ) -> FlightPredictionResponse:
        raise AssertionError("dry run attempted a network operation")

    audit = run_monitoring_traffic(
        TrafficPlan(count=3),
        sender=unexpected_sender,
        now=lambda: datetime(2026, 8, 13, tzinfo=UTC),
    )
    assert audit.mode == "dry-run"
    assert audit.planned_count == 3
    assert audit.attempted_count == 0
    assert audit.returned_prediction_ids == ()
    assert audit.traffic_source is TrafficSource.SYNTHETIC_LOAD_TEST


def test_apply_records_success_failure_and_rate_limit() -> None:
    calls = 0
    sleeps: list[float] = []
    sources: list[TrafficSource] = []

    def sender(request: object, *, traffic_source: TrafficSource) -> FlightPredictionResponse:
        nonlocal calls
        calls += 1
        sources.append(traffic_source)
        if calls == 2:
            raise ApiClientError("sanitized failure", status_code=503)
        return response(f"prediction-{calls}")

    audit = run_monitoring_traffic(
        TrafficPlan(count=3, rate_per_second=4),
        apply=True,
        sender=sender,
        now=lambda: datetime(2026, 8, 13, 12, tzinfo=UTC),
        sleep=sleeps.append,
    )
    assert audit.attempted_count == 3
    assert audit.successful_count == 2
    assert audit.failed_count == 1
    assert audit.returned_prediction_ids == ("prediction-1", "prediction-3")
    assert sleeps == [0.25, 0.25]
    assert sources == [TrafficSource.SYNTHETIC_LOAD_TEST] * 3
    assert audit.traffic_source is TrafficSource.SYNTHETIC_LOAD_TEST


def test_apply_requires_sender_and_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="prediction sender"):
        run_monitoring_traffic(TrafficPlan(count=1), apply=True)
    with pytest.raises(ValueError, match="timezone-aware"):
        run_monitoring_traffic(TrafficPlan(count=1), now=lambda: datetime(2026, 8, 13), apply=False)


@pytest.mark.parametrize(
    "value",
    [
        "api.example.com",
        "ftp://api.example.com",
        "https://user:secret@api.example.com",
        "https://api.example.com/predict",
        "https://api.example.com?token=x",
    ],
)
def test_api_base_url_rejects_unsafe_or_non_origin_values(value: str) -> None:
    with pytest.raises(ValueError, match="credential-free"):
        validate_api_base_url(value)
    assert validate_api_base_url("https://api.example.com/") == "https://api.example.com"


def test_empty_templates_are_rejected_and_audit_is_atomic(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="route template"):
        generate_requests(TrafficPlan(count=1), templates=[])
    audit = run_monitoring_traffic(
        TrafficPlan(count=1), now=lambda: datetime(2026, 8, 13, tzinfo=UTC)
    )
    output = tmp_path / "nested" / "summary.json"
    write_audit_summary(output, audit)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["traffic_kind"] == "synthetic_load_test_via_prediction_api"
    assert payload["traffic_source"] == "synthetic_load_test"
    assert "api_base_url" not in payload


def test_cli_defaults_to_offline_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_monitoring_traffic.py",
            "--api-base-url",
            "https://unreachable.invalid",
            "--count",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_aws_sdk_forbidden_environment(tmp_path),
        cwd=ROOT,
    )
    assert json.loads(result.stdout)["mode"] == "dry-run"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["attempted_count"] == 0
    assert payload["traffic_source"] == "synthetic_load_test"


def test_load_testing_module_import_does_not_require_aws_sdk(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import flight_delay.load_testing.traffic; "
                "assert 'boto3' not in sys.modules; "
                "assert 'botocore' not in sys.modules; "
                "print('aws-sdk-isolated')"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_aws_sdk_forbidden_environment(tmp_path),
        cwd=ROOT,
    )
    assert result.stdout.strip() == "aws-sdk-isolated"
