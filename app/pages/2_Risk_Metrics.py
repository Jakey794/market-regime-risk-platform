"""Risk metrics placeholder page."""

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
    "Risk Metrics",
    "Historical portfolio-risk measures from the existing deterministic engine.",
)
render_context_summary(state)
render_placeholder(
    "Volatility and downside risk",
    "Annualized volatility, drawdown, Sharpe, and Sortino outputs will appear here.",
)
render_placeholder(
    "Tail risk",
    "Historical VaR, CVaR, and worst-period observations will appear here.",
)
render_placeholder(
    "Concentration risk",
    "Effective holdings and concentration diagnostics will appear here.",
)
render_disclaimer()
