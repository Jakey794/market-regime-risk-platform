"""Portfolio overview placeholder page."""

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
    "Portfolio Overview",
    "A high-level view of portfolio behavior and historical risk.",
)
render_context_summary(state)
render_placeholder(
    "Portfolio snapshot",
    "Summary cards will use the existing portfolio-risk summary API.",
)
render_placeholder(
    "Historical performance and drawdown",
    "Portfolio and benchmark history will appear here in a later iteration.",
)
render_placeholder(
    "Exposure overview",
    "Asset, region, style, and sector-proxy exposures will appear here.",
)
render_disclaimer()
