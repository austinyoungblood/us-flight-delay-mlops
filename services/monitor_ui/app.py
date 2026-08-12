"""DynamoDB-only operational and model monitoring Streamlit dashboard."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from flight_delay.monitoring import (
    MonitoringRepository,
    feedback_metrics,
    jensen_shannon_divergence,
    operational_metrics,
    population_stability_index,
    prediction_frame,
    target_drift,
)
from flight_delay.monitoring.metrics import finite_metric
from flight_delay.persistence import PersistenceConflict, PersistenceError


def _settings() -> dict[str, Any]:
    return {
        "table_name": os.getenv("DYNAMODB_TABLE", "flight-delay-events"),
        "region_name": os.getenv("AWS_REGION", "us-west-2"),
        "endpoint_url": os.getenv("DYNAMODB_ENDPOINT_URL") or None,
        "max_days": int(os.getenv("MONITOR_MAX_DAYS", "31")),
    }


@st.cache_resource
def monitor_repository() -> MonitoringRepository:
    repository = MonitoringRepository(**_settings())
    repository.connect()
    return repository


def _repository() -> MonitoringRepository:
    return st.session_state.get("_monitor_repository") or monitor_repository()


@st.cache_data(ttl=int(os.getenv("MONITOR_QUERY_CACHE_TTL_SECONDS", "30")))
def cached_query(start_date: date, end_date: date) -> list[dict[str, Any]]:
    repository = MonitoringRepository(**_settings())
    repository.connect()
    try:
        return repository.query_predictions(start_date, end_date)
    finally:
        repository.close()


def _query(
    repository: MonitoringRepository, start_date: date, end_date: date
) -> list[dict[str, Any]]:
    if st.session_state.get("_monitor_repository") is not None:
        return repository.query_predictions(start_date, end_date)
    return cached_query(start_date, end_date)


def _baseline(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    payload = metadata.get("model_metadata") or metadata
    return payload.get("training_baseline") or {}


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


st.set_page_config(page_title="Flight Delay Monitoring", page_icon="📈", layout="wide")
st.title("Flight Delay Operations & Model Monitoring")
st.caption(
    "DynamoDB data plane only. GSI aggregates are eventually consistent and may briefly lag writes."
)

default_days = max(1, int(os.getenv("MONITOR_DEFAULT_DAYS", "7")))
max_days = max(default_days, int(os.getenv("MONITOR_MAX_DAYS", "31")))
today = datetime.now(UTC).date()

with st.sidebar:
    st.header("UTC query controls")
    date_range = st.date_input(
        "Date range",
        value=(today - timedelta(days=default_days - 1), today),
        max_value=today,
    )
    if st.button("Refresh data"):
        cached_query.clear()
        st.rerun()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
if (end_date - start_date).days + 1 > max_days:
    st.error(f"Interactive monitoring range cannot exceed {max_days} UTC days.")
    st.stop()

try:
    repository = _repository()
    items = _query(repository, start_date, end_date)
except (PersistenceError, ValueError) as error:
    st.error(str(error))
    st.stop()

all_frame = prediction_frame(items)
with st.sidebar:
    carrier_options = sorted(all_frame["carrier"].dropna().unique()) if not all_frame.empty else []
    route_options = sorted(all_frame["route"].dropna().unique()) if not all_frame.empty else []
    status_options = (
        sorted(all_frame["request_status"].dropna().unique()) if not all_frame.empty else []
    )
    version_options = (
        sorted(all_frame["model_version"].dropna().unique()) if not all_frame.empty else []
    )
    carrier_filter = st.selectbox("Carrier", ["All", *carrier_options])
    route_filter = st.selectbox("Route", ["All", *route_options])
    status_filter = st.selectbox("Request status", ["All", *status_options])
    version_filter = st.selectbox("Model version", ["All", *version_options])
    exclude_demo = st.checkbox("Exclude demo data", value=False)

frame = all_frame.copy()
for column, selected in (
    ("carrier", carrier_filter),
    ("route", route_filter),
    ("request_status", status_filter),
    ("model_version", version_filter),
):
    if selected != "All" and not frame.empty:
        frame = frame[frame[column] == selected]
if exclude_demo and not frame.empty:
    frame = frame[~frame["demo_data"]]

demo_count = int(frame["demo_data"].sum()) if not frame.empty else 0
if demo_count:
    st.warning(
        f"Selected window includes {demo_count} deterministic demo records. "
        "They are not real traffic or model inference."
    )

metadata_version = None if version_filter == "All" else version_filter
if metadata_version is None and version_options:
    metadata_version = version_options[-1]
try:
    metadata = repository.get_model_metadata(metadata_version)
except PersistenceError as error:
    st.warning(str(error))
    metadata = None

st.subheader("Active model")
if metadata:
    model = metadata.get("model_metadata") or metadata
    alias = model.get("serving_alias", "unknown")
    release_decision = model.get("release_decision") or {}
    deployment_purpose = model.get("deployment_purpose", release_decision.get("deployment_purpose"))
    internal_gate = model.get(
        "internal_production_gate_passed",
        release_decision.get("internal_production_gate_passed"),
    )
    if deployment_purpose == "academic_demo" or internal_gate is False:
        st.warning(
            model.get("governance_notice")
            or "Academic demonstration — W&B production alias used for course deployment; "
            "the model did not pass the project's stricter internal production-quality gate."
        )
    st.write(
        f"**Alias/version:** {alias} / {model.get('registry_version', 'unknown')}  \n"
        f"**Registry digest:** `{model.get('registry_digest', 'unknown')}`  \n"
        f"**Bundle hash:** `{model.get('bundle_digest', 'unknown')}`"
    )
else:
    st.info("No active-model metadata is available in the selected local data plane.")

if frame.empty:
    st.info("No prediction events are available for the selected UTC window and filters.")
    st.stop()

successful = frame[frame["request_status"] == "success"]
operational = operational_metrics(frame)
st.subheader("Operational overview")
columns = st.columns(5)
columns[0].metric("Requests", operational["request_count"])
columns[1].metric("Success", operational["success_count"], _pct(operational["success_rate"]))
columns[2].metric("Errors", operational["error_count"])
columns[3].metric("Cache hits", operational["cache_hit_count"], _pct(operational["cache_hit_rate"]))
columns[4].metric("Latency p95", f"{finite_metric(operational['latency_ms']['p95'])} ms")

latency_rows = []
for name in ("latency_ms", "inference_latency_ms", "persistence_latency_ms"):
    latency_rows.append({"metric": name, **operational[name]})
st.dataframe(pd.DataFrame(latency_rows), hide_index=True, use_container_width=True)
volume = frame.set_index("created_at").resample("D").size().rename("requests")
st.line_chart(volume)
st.caption("Request volume by UTC day")
latency_over_time = (
    frame.set_index("created_at")[["latency_ms", "inference_latency_ms", "persistence_latency_ms"]]
    .apply(pd.to_numeric, errors="coerce")
    .resample("D")
    .median()
    .dropna(how="all")
)
if not latency_over_time.empty:
    st.line_chart(latency_over_time)
    st.caption("Median total, inference, and persistence latency by UTC day")

st.subheader("Prediction distribution / target drift indicator")
baseline = _baseline(metadata)
drift = target_drift(successful, baseline.get("target_prevalence"))
delayed_count = int(successful["predicted_delayed"].fillna(False).astype(bool).sum())
on_time_count = len(successful) - delayed_count
delayed_rate = delayed_count / len(successful) if len(successful) else None
on_time_rate = on_time_count / len(successful) if len(successful) else None
dist_columns = st.columns(5)
dist_columns[0].metric("Successful predictions", drift["n_success"])
dist_columns[1].metric("Predicted delayed", delayed_count, _pct(delayed_rate))
dist_columns[2].metric("Predicted on time", on_time_count, _pct(on_time_rate))
dist_columns[3].metric("Training prevalence", _pct(drift["training_delayed_prevalence"]))
dist_columns[4].metric("Absolute prevalence delta", _pct(drift["absolute_prevalence_delta"]))
st.caption("The prevalence delta is a prediction/target drift indicator, not measured accuracy.")
left, right = st.columns(2)
left.bar_chart(successful["risk_band"].value_counts())
right.bar_chart(successful["model_version"].value_counts())
st.caption("Risk-band distribution (left) and model-version distribution (right)")
probability_histogram = (
    pd.cut(successful["delay_probability"], bins=10, include_lowest=True)
    .value_counts()
    .sort_index()
)
probability_histogram.index = probability_histogram.index.astype(str)
st.bar_chart(probability_histogram)
st.caption("Predicted delay-probability histogram")

st.subheader("Input drift")
numeric_map = {
    "distance": "distance",
    "scheduled_elapsed_time": "scheduled_elapsed_time",
    "scheduled_departure_hour": "scheduled_departure_hour",
}
drift_rows: list[dict[str, Any]] = []
for column, baseline_key in numeric_map.items():
    result = population_stability_index(
        successful[column], baseline.get("numeric", {}).get(baseline_key)
    )
    drift_rows.append({"feature": column, "method": "PSI", **result})
for column, baseline_key in {
    "carrier": "carrier",
    "origin": "origin",
    "destination": "destination",
    "month": "month",
}.items():
    result = jensen_shannon_divergence(
        successful[column].astype("string"), baseline.get("categorical", {}).get(baseline_key)
    )
    drift_rows.append({"feature": column, "method": "Jensen-Shannon", **result})
st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)
st.caption("PSI/JS use epsilon smoothing; missing baseline data is reported, never fabricated.")

st.subheader("Feedback performance")
feedback = feedback_metrics(frame)
feedback_columns = st.columns(6)
feedback_columns[0].metric("Coverage", _pct(feedback["coverage"]), f"n={feedback['n_feedback']}")
for widget, label, key in zip(
    feedback_columns[1:],
    ("Accuracy", "Precision", "Recall", "F1", "Brier"),
    ("accuracy", "precision", "recall", "f1", "brier_score"),
    strict=True,
):
    widget.metric(label, finite_metric(feedback[key]), f"n={feedback['n_feedback']}")
st.write(
    f"Correct: {feedback['correct_count']} · Incorrect: {feedback['incorrect_count']} · "
    f"Labeled: {feedback['n_feedback']} / {feedback['n_success']} successful predictions"
)
feedback_times = (
    frame[frame["actual_delayed"].notna()].set_index("feedback_at").resample("D").size()
)
if not feedback_times.empty:
    st.line_chart(feedback_times.rename("feedback"))

st.subheader("Prediction inspector / feedback adjudication")
prediction_ids = successful["prediction_id"].dropna().astype(str).tolist()
selected_id = st.selectbox("Prediction ID", prediction_ids)
selected = repository.get_prediction(selected_id) if selected_id else None
if selected:
    st.json(
        {
            "request": selected.get("request"),
            "delay_probability": selected.get("delay_probability"),
            "predicted_delayed": selected.get("predicted_delayed"),
            "risk_band": selected.get("risk_band"),
            "model_alias": selected.get("model_alias"),
            "model_version": selected.get("model_version"),
            "model_digest": selected.get("model_digest"),
            "latency_ms": selected.get("latency_ms"),
            "cache_hit": selected.get("cache_hit"),
            "feedback": selected.get("feedback"),
            "feedback_revision": selected.get("feedback_revision", 0),
        }
    )
    with st.form("adjudication_form"):
        actual_delayed = st.checkbox("Actual delayed", value=False)
        arrival_delay = st.number_input(
            "Observed arrival delay minutes", min_value=-300.0, max_value=2880.0, value=0.0
        )
        notes = st.text_area("Adjudication notes", max_chars=1000)
        adjudicate = st.form_submit_button("Create / correct feedback")
    if adjudicate:
        feedback_payload = {
            "actual_delayed": actual_delayed,
            "arrival_delay_minutes": arrival_delay,
            "notes": notes or None,
            "source": "monitor-ui",
            "feedback_correct": actual_delayed == bool(selected.get("predicted_delayed")),
            "feedback_at": datetime.now(UTC),
        }
        try:
            updated = repository.update_feedback(selected_id, feedback_payload)
            revision = updated.get("feedback_revision") if updated else None
            cached_query.clear()
            st.success(f"Feedback saved at revision {revision}.")
        except (PersistenceConflict, PersistenceError) as error:
            st.error(str(error))
