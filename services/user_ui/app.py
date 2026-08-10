"""Traveler-facing Streamlit placeholder for Brief 01."""

import streamlit as st

st.set_page_config(page_title="Flight Delay Estimate", page_icon="✈️")
st.title("U.S. Flight Delay Estimate")
st.info(
    "This interface is a scaffold. Prediction, route reliability, and feedback are not "
    "implemented in Brief 01. No model is loaded by this application."
)
st.caption(
    "Future estimates will use scheduled flight information and historical data; they will not "
    "represent live flight status or a guarantee."
)
