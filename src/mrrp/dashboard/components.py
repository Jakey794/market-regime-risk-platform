"""Reusable Streamlit components for the dashboard shell."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from mrrp.dashboard.formatting import format_date_range, humanize_identifier
from mrrp.dashboard.loaders import DashboardData
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


def render_metric_cards(cards: Sequence[tuple[str, str]], columns: int = 3) -> None:
    """Render formatted metric labels and values in a reusable grid."""
    if columns < 1:
        raise ValueError("columns must be positive")
    for offset in range(0, len(cards), columns):
        row = st.columns(columns)
        for column, (label, value) in zip(row, cards[offset : offset + columns]):
            column.metric(label, value)


def render_placeholder(title: str, description: str) -> None:
    """Render an explicit placeholder for a future dashboard section."""
    with st.container(border=True):
        st.subheader(title)
        st.write(description)
        st.caption("Placeholder—no analytical output is calculated here yet.")


def render_data_warnings(data: DashboardData) -> None:
    """Display non-fatal data-quality limitations for the current selection."""
    required_tickers = list(data.portfolio_config.holdings.index)
    if data.portfolio_config.benchmark not in required_tickers:
        required_tickers.append(data.portfolio_config.benchmark)
    missing_observations = int(data.prices.loc[:, required_tickers].isna().sum().sum())
    if missing_observations:
        st.warning(
            f"The selected inputs contain {missing_observations:,} missing price "
            "observations. Metrics will use their existing alignment rules."
        )


def render_disclaimer() -> None:
    """Render the dashboard's scope and advice disclaimer."""
    st.divider()
    st.caption(
        "This is a historical risk modeling dashboard, not a prediction app. "
        "It does not provide financial advice."
    )
