"""Correlation and beta placeholder page."""

import streamlit as st

from mrrp.dashboard.components import (
    render_context_summary,
    render_disclaimer,
    render_page_header,
    render_placeholder,
)
from mrrp.dashboard.state import get_dashboard_state


state = get_dashboard_state(st.session_state)

render_page_header(
    "Correlation & Beta",
    "Dependence and benchmark sensitivity without return forecasts.",
)
render_context_summary(state)
render_placeholder(
    "Correlation structure",
    "Current and rolling asset-correlation outputs will appear here.",
)
render_placeholder(
    "Benchmark beta",
    "Portfolio and holding beta estimates will appear here.",
)
render_placeholder(
    "Diversification diagnostics",
    "Correlation regime and diversification-ratio context will appear here.",
)
render_disclaimer()
