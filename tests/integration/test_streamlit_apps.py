from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from flight_delay.contracts import (
    DependencyHealth,
    FeedbackRecord,
    FlightPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskBand,
    RouteReliability,
    TrafficSource,
)
from flight_delay.monitoring.demo import demo_events
from flight_delay.ui import ApiClientError

ROOT = Path(__file__).resolve().parents[2]


class FakeApiClient:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.predictions = 0
        self.prediction_sources: list[TrafficSource] = []
        self.feedback = 0

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ready" if self.ready else "degraded",
            model_loaded=self.ready,
            database_connected=self.ready,
            dependencies={
                "registry": DependencyHealth(
                    status="ready" if self.ready else "unavailable",
                    detail="loaded" if self.ready else "not configured",
                ),
                "dynamodb": DependencyHealth(
                    status="ready" if self.ready else "unavailable",
                    detail="connected" if self.ready else "not configured",
                ),
            },
        )

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            registry_path="registry",
            serving_alias="production",
            registry_version="v0",
            registry_digest="digest",
            source_artifact_digest="source",
            bundle_digest="bundle",
            selection_lock_sha256="lock",
            route_asset_sha256="route",
            classification_threshold=0.18,
            feature_schema=["Origin"],
            training_partitions={"base_fit": "2025"},
            release_decision={
                "serving_alias": "production",
                "internal_production_gate_passed": False,
                "deployment_purpose": "academic_demo",
            },
            release_git_sha="abc",
            loaded_at=datetime(2026, 8, 10, tzinfo=UTC),
            internal_production_gate_passed=False,
            deployment_purpose="academic_demo",
            governance_notice="Academic demonstration — internal production gate failed.",
            serving_stage_notice="Academic demonstration — internal production gate failed.",
        )

    def route_reliability(self, **kwargs: Any) -> list[RouteReliability]:
        return [
            RouteReliability(
                scope="all_carriers",
                origin="DEN",
                destination="LAX",
                eligible_flights=100,
                on_time_count=70,
                on_time_rate=0.7,
                delayed_count=30,
                delayed_rate=0.3,
                meets_minimum_support=True,
            )
        ]

    def predict(
        self,
        request: Any,
        *,
        traffic_source: TrafficSource = TrafficSource.API_UNSPECIFIED,
    ) -> FlightPredictionResponse:
        self.predictions += 1
        self.prediction_sources.append(traffic_source)
        return FlightPredictionResponse(
            prediction_id="prediction-one",
            delay_probability=0.3,
            predicted_delayed=True,
            risk_band="high",
            classification_threshold=0.18,
            route_reliability=self.route_reliability(),
            model_alias="production",
            model_version="v0",
            model_digest="digest",
            cache_hit=False,
            latency_ms=12,
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
        )

    def submit_feedback(self, prediction_id: str, feedback: Any) -> FeedbackRecord:
        self.feedback += 1
        return FeedbackRecord(
            **feedback.model_dump(),
            feedback_correct=True,
            feedback_at=datetime(2026, 8, 10, tzinfo=UTC),
            feedback_revision=self.feedback,
        )


class FakeMonitorRepository:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.revision = 0

    def query_predictions(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        return self.items

    def get_model_metadata(self, version: str | None = None) -> dict[str, Any]:
        return {
            "model_metadata": {
                "serving_alias": "production",
                "registry_version": "v0",
                "registry_digest": "digest",
                "bundle_digest": "bundle",
                "internal_production_gate_passed": False,
                "deployment_purpose": "academic_demo",
                "governance_notice": "Academic demonstration — internal production gate failed.",
                "release_decision": {
                    "internal_production_gate_passed": False,
                    "deployment_purpose": "academic_demo",
                },
                "training_baseline": {
                    "target_prevalence": 0.22,
                    "numeric": {},
                    "categorical": {},
                },
            }
        }

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return next((item for item in self.items if item["prediction_id"] == prediction_id), None)

    def update_feedback(self, prediction_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
        self.revision += 1
        return {"feedback_revision": self.revision, "feedback": feedback}


def provenance_demo_events() -> list[dict[str, Any]]:
    items = demo_events(batch_id="app-test", count=6, start_date=date(2026, 8, 8))
    sources = [
        TrafficSource.TRAVELER_UI,
        TrafficSource.SYNTHETIC_LOAD_TEST,
        TrafficSource.API_UNSPECIFIED,
        None,
        TrafficSource.TRAVELER_UI,
        TrafficSource.SYNTHETIC_LOAD_TEST,
    ]
    for item, source in zip(items, sources, strict=True):
        if source is not None:
            item["traffic_source"] = source.value
    return items


def test_traveler_ready_academic_production_prediction_and_feedback() -> None:
    fake = FakeApiClient()
    app = AppTest.from_file(str(ROOT / "services/user_ui/app.py"))
    app.session_state["_api_client"] = fake
    app.run(timeout=10)
    assert not app.exception
    assert any("Academic demonstration" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Estimate delay risk").click().run()
    assert fake.predictions == 1
    assert fake.prediction_sources == [TrafficSource.TRAVELER_UI]
    assert any(metric.label == "Delay probability" for metric in app.metric)
    assert any(
        metric.label == "Threshold signal" and metric.value == "Above model threshold"
        for metric in app.metric
    )
    assert not any(metric.label == "Classification" for metric in app.metric)
    assert any("Decision threshold: 18.0%" in markdown.value for markdown in app.markdown)
    assert any("more likely than not" in caption.value for caption in app.caption)
    next(button for button in app.button if button.label == "Save feedback").click().run()
    assert fake.feedback == 1
    assert any("revision 1" in success.value for success in app.success)
    next(button for button in app.button if button.label == "Save feedback").click().run()
    assert fake.feedback == 2
    assert any("revision 2" in success.value for success in app.success)


def test_traveler_degraded_state_disables_prediction() -> None:
    app = AppTest.from_file(str(ROOT / "services/user_ui/app.py"))
    app.session_state["_api_client"] = FakeApiClient(ready=False)
    app.run(timeout=10)
    assert not app.exception
    assert any("degraded" in error.value for error in app.error)
    button = next(button for button in app.button if button.label == "Estimate delay risk")
    assert button.disabled


def test_traveler_uses_below_threshold_signal_instead_of_on_time_classification() -> None:
    class BelowThresholdApiClient(FakeApiClient):
        def predict(
            self,
            request: Any,
            *,
            traffic_source: TrafficSource = TrafficSource.API_UNSPECIFIED,
        ) -> FlightPredictionResponse:
            prediction = super().predict(request, traffic_source=traffic_source)
            return prediction.model_copy(
                update={
                    "delay_probability": 0.1,
                    "predicted_delayed": False,
                    "risk_band": RiskBand.LOW,
                }
            )

    app = AppTest.from_file(str(ROOT / "services/user_ui/app.py"))
    app.session_state["_api_client"] = BelowThresholdApiClient()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Estimate delay risk").click().run()

    assert not app.exception
    assert any(
        metric.label == "Threshold signal" and metric.value == "Below model threshold"
        for metric in app.metric
    )
    assert not any(metric.value == "On time" for metric in app.metric)


def test_traveler_safely_renders_prediction_api_error() -> None:
    class FailingApiClient(FakeApiClient):
        def predict(
            self,
            request: Any,
            *,
            traffic_source: TrafficSource = TrafficSource.API_UNSPECIFIED,
        ) -> FlightPredictionResponse:
            raise ApiClientError("safe dependency failure", status_code=503)

    app = AppTest.from_file(str(ROOT / "services/user_ui/app.py"))
    app.session_state["_api_client"] = FailingApiClient()
    app.run(timeout=10)
    next(button for button in app.button if button.label == "Estimate delay risk").click().run()

    assert not app.exception
    assert any(error.value == "safe dependency failure" for error in app.error)


def test_monitor_empty_window() -> None:
    app = AppTest.from_file(str(ROOT / "services/monitor_ui/app.py"))
    app.session_state["_monitor_repository"] = FakeMonitorRepository([])
    app.run(timeout=10)
    assert not app.exception
    assert any("No prediction events" in info.value for info in app.info)


def test_monitor_populated_metrics_demo_warning_and_inspector() -> None:
    items = provenance_demo_events()
    app = AppTest.from_file(str(ROOT / "services/monitor_ui/app.py"))
    app.session_state["_monitor_repository"] = FakeMonitorRepository(items)
    app.run(timeout=10)
    assert not app.exception
    assert any("Academic demonstration" in warning.value for warning in app.warning)
    assert any("demo records" in warning.value for warning in app.warning)
    assert any(metric.label == "Requests" and metric.value == "6" for metric in app.metric)
    assert any(metric.label == "Predicted delayed" for metric in app.metric)
    assert any(metric.label == "Predicted on time" for metric in app.metric)
    assert any(metric.label == "traveler_ui" and metric.value == "2" for metric in app.metric)
    assert any(
        metric.label == "synthetic_load_test" and metric.value == "2" for metric in app.metric
    )
    assert any(metric.label == "api_unspecified" and metric.value == "1" for metric in app.metric)
    assert any(
        metric.label == "legacy_unattributed" and metric.value == "1" for metric in app.metric
    )
    assert any(
        metric.label == "Absolute prevalence delta" and metric.value != "N/A"
        for metric in app.metric
    )
    assert any(metric.label == "Coverage" and "n=" in metric.delta for metric in app.metric)
    assert any("Prediction inspector" in subheader.value for subheader in app.subheader)
    next(
        button for button in app.button if button.label == "Create / correct feedback"
    ).click().run()
    assert any("revision 1" in success.value for success in app.success)


def test_monitor_filters_and_demo_exclusion() -> None:
    items = provenance_demo_events()
    app = AppTest.from_file(str(ROOT / "services/monitor_ui/app.py"))
    app.session_state["_monitor_repository"] = FakeMonitorRepository(items)
    app.run(timeout=10)
    next(select for select in app.selectbox if select.label == "Carrier").set_value("UA").run()
    assert any(metric.label == "Requests" and metric.value == "2" for metric in app.metric)

    app = AppTest.from_file(str(ROOT / "services/monitor_ui/app.py"))
    app.session_state["_monitor_repository"] = FakeMonitorRepository(items)
    app.run(timeout=10)
    next(select for select in app.selectbox if select.label == "Traffic source").set_value(
        "synthetic_load_test"
    ).run()
    assert any(metric.label == "Requests" and metric.value == "2" for metric in app.metric)
    assert any(
        metric.label == "Successful predictions" and metric.value == "2" for metric in app.metric
    )
    assert not any(metric.label == "traveler_ui" for metric in app.metric)

    app = AppTest.from_file(str(ROOT / "services/monitor_ui/app.py"))
    app.session_state["_monitor_repository"] = FakeMonitorRepository(items)
    app.run(timeout=10)
    next(box for box in app.checkbox if box.label == "Exclude demo data").check().run()
    assert any("No prediction events" in info.value for info in app.info)


def test_ui_import_boundaries_are_explicit() -> None:
    traveler = (ROOT / "services/user_ui/app.py").read_text()
    monitor = (ROOT / "services/monitor_ui/app.py").read_text()
    for forbidden in ("boto3", "wandb", "flight_delay.persistence", "flight_delay.serving"):
        assert forbidden not in traveler
    for forbidden in ("httpx", "wandb", "flight_delay.ui", "flight_delay.serving", "model.joblib"):
        assert forbidden not in monitor
