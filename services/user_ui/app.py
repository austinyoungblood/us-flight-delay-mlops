"""Traveler / operations Streamlit application; FastAPI is its only data plane."""

from __future__ import annotations

import os
from datetime import date, time

import streamlit as st

from flight_delay.contracts import FeedbackRequest, FlightPredictionRequest
from flight_delay.ui import ApiClientError, FlightDelayApiClient


@st.cache_resource
def api_client() -> FlightDelayApiClient:
    return FlightDelayApiClient(
        os.getenv("API_BASE_URL", "http://api:8000"),
        connect_timeout_seconds=float(os.getenv("API_CONNECT_TIMEOUT_SECONDS", "5")),
        read_timeout_seconds=float(os.getenv("API_READ_TIMEOUT_SECONDS", "15")),
    )


def _client() -> FlightDelayApiClient:
    return st.session_state.get("_api_client") or api_client()


def _show_reliability(records: list[object]) -> None:
    if not records:
        st.caption("No historical route evidence is available.")
        return
    st.subheader("Historical route reliability")
    st.caption("Historical context only — not live flight status or a guarantee.")
    for record in records:
        label = "Carrier route" if record.scope == "carrier_route" else "All carriers"
        st.write(
            f"**{label}:** {record.on_time_rate:.1%} on time across "
            f"{record.eligible_flights:,} eligible flights"
        )
        if not record.meets_minimum_support:
            st.warning(f"{label} evidence is below minimum support.")


st.set_page_config(page_title="Flight Delay Estimate", page_icon="✈️", layout="wide")
st.title("U.S. Flight Delay Estimate")
st.caption(
    "Pre-departure academic estimate from scheduled information; not live status or a guarantee."
)

client = _client()
ready = False
try:
    health = client.health()
    ready = health.status == "ready"
    if ready:
        st.success("Prediction API is ready.")
    else:
        details = "; ".join(
            f"{name}: {dependency.detail}" for name, dependency in health.dependencies.items()
        )
        st.error(f"Prediction API is degraded — {details}")
except ApiClientError as error:
    st.error(error.detail)

model_info = None
if ready:
    try:
        model_info = client.model_info()
        if (
            model_info.deployment_purpose == "academic_demo"
            or not model_info.internal_production_gate_passed
        ):
            st.warning(model_info.governance_notice)
        st.caption(
            f"Model {model_info.registry_version} · alias {model_info.serving_alias} · "
            f"digest {model_info.registry_digest[:12]}…"
        )
    except ApiClientError as error:
        st.error(error.detail)
        ready = False

with st.form("prediction_form"):
    st.subheader("Scheduled flight")
    left, middle, right = st.columns(3)
    carrier = left.text_input("Carrier", value="UA", max_chars=2)
    origin = middle.text_input("Origin", value="DEN", max_chars=3)
    destination = right.text_input("Destination", value="LAX", max_chars=3)
    flight_date = left.date_input("Flight date", value=date.today())
    scheduled_departure = middle.time_input("Scheduled departure", value=time(8, 0))
    scheduled_arrival = right.time_input("Scheduled arrival", value=time(9, 30))
    elapsed = left.number_input("Scheduled elapsed minutes", min_value=1, max_value=1500, value=150)
    distance = middle.number_input("Distance miles", min_value=1.0, max_value=10000.0, value=862.0)
    show_route_context = st.checkbox("Fetch historical route context before prediction", value=True)
    submitted = st.form_submit_button("Estimate delay risk", disabled=not ready)

if submitted:
    try:
        request = FlightPredictionRequest(
            carrier=carrier,
            origin=origin,
            destination=destination,
            flight_date=flight_date,
            scheduled_departure=scheduled_departure,
            scheduled_arrival=scheduled_arrival,
            scheduled_elapsed_minutes=int(elapsed),
            distance_miles=float(distance),
        )
        if show_route_context:
            try:
                st.session_state["route_context"] = client.route_reliability(
                    origin=request.origin,
                    destination=request.destination,
                    carrier=request.carrier,
                )
            except ApiClientError as error:
                if error.status_code != 404:
                    st.warning(error.detail)
                st.session_state["route_context"] = []
        prediction = client.predict(request)
        st.session_state["last_prediction"] = prediction
    except (ApiClientError, ValueError) as error:
        detail = error.detail if isinstance(error, ApiClientError) else str(error)
        st.error(detail)

prediction = st.session_state.get("last_prediction")
if prediction is not None:
    st.divider()
    st.subheader("Estimate")
    one, two, three = st.columns(3)
    one.metric("Delay probability", f"{prediction.delay_probability:.1%}")
    threshold_signal = (
        "Above model threshold" if prediction.predicted_delayed else "Below model threshold"
    )
    two.metric("Threshold signal", threshold_signal)
    three.metric("Risk band", prediction.risk_band.value.title())
    st.write(f"**Decision threshold: {prediction.classification_threshold:.1%}**")
    st.caption(
        "The threshold signal indicates whether the estimated probability exceeds the model's "
        "selected operating point. It does not mean the flight is more likely than not to be "
        "delayed."
    )
    if prediction.support_warning:
        st.warning(prediction.support_warning)
    _show_reliability(prediction.route_reliability or st.session_state.get("route_context", []))
    st.caption(f"Prediction ID: `{prediction.prediction_id}`")
    if model_info is not None:
        st.warning(model_info.governance_notice)
    with st.expander("Technical details"):
        st.write(f"Threshold: {prediction.classification_threshold:.6f}")
        st.write(f"Raw predicted_delayed: {prediction.predicted_delayed}")
        st.write(f"Model: {prediction.model_alias} / {prediction.model_version}")
        st.write(f"Total latency: {prediction.latency_ms:.2f} ms")
        st.write(f"Inference cache hit: {prediction.cache_hit}")

    with st.form("feedback_form"):
        st.subheader("Observed outcome / correction")
        actual_delayed = st.checkbox("Flight actually arrived at least 15 minutes late")
        arrival_delay = st.number_input(
            "Arrival delay minutes", min_value=-300.0, max_value=2880.0, value=0.0
        )
        notes = st.text_area("Notes", max_chars=1000)
        feedback_submitted = st.form_submit_button("Save feedback")
    if feedback_submitted:
        try:
            feedback = client.submit_feedback(
                prediction.prediction_id,
                FeedbackRequest(
                    actual_delayed=actual_delayed,
                    arrival_delay_minutes=arrival_delay,
                    notes=notes or None,
                    source="traveler-ui",
                ),
            )
            st.session_state["last_feedback"] = feedback
            st.success(f"Feedback saved (revision {feedback.feedback_revision}).")
        except ApiClientError as error:
            st.error(error.detail)
