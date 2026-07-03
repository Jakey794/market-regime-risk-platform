"""Reusable Streamlit components for the dashboard shell."""

from __future__ import annotations

import streamlit as st

from mrrp.dashboard.formatting import format_date_range, humanize_identifier
from mrrp.dashboard.state import DashboardState


def render_page_header(title: str, description: str) -> None:
    """Render a consistent dashboard page heading."""
    st.title(title)
    st.caption(description)


def render_context_summary(state: DashboardState) -> None:
    """Render the shared portfolio, benchmark, and date selections."""
    columns = st.columns(3)
    columns[0].metric("Portfolio", humanize_identifier(state.portfolio))
    columns[1].metric("Benchmark", state.benchmark)
    columns[2].metric(
        "Analysis period",
        format_date_range(state.start_date, state.end_date),
    )


def render_placeholder(title: str, description: str) -> None:
    """Render an explicit placeholder for a future dashboard section."""
    with st.container(border=True):
        st.subheader(title)
        st.write(description)
        st.caption("Placeholder—no analytical output is calculated here yet.")


def render_disclaimer() -> None:
    """Render the dashboard's scope and advice disclaimer."""
    st.divider()
    st.caption(
        "This is a historical risk modeling dashboard, not a prediction app. "
        "It does not provide financial advice."
    )
