"""Data quality placeholder page."""

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
    "Data Quality",
    "Visibility into the inputs supporting every historical risk estimate.",
)
render_context_summary(state)
render_placeholder(
    "Coverage",
    "Available history and ticker coverage checks will appear here.",
)
render_placeholder(
    "Missing observations",
    "Missing-value and alignment diagnostics will appear here.",
)
render_placeholder(
    "Freshness and provenance",
    "Cache timestamps and source metadata will appear here.",
)
render_disclaimer()
