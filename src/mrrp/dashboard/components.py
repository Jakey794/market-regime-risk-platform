"""Reusable Streamlit components for the dashboard shell."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from mrrp.data.validators import report_missing_data
from mrrp.dashboard.formatting import format_date_range, humanize_identifier
from mrrp.dashboard.loaders import DashboardData
from mrrp.dashboard.state import DashboardState


SHORT_RANGE_OBSERVATIONS = 252


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


def render_metric_cards(
    cards: Sequence[tuple[str, str] | tuple[str, str, str]],
    columns: int = 3,
) -> None:
    """Render formatted metric labels and values in a reusable grid.

    Each card is a ``(label, value)`` pair, or a ``(label, value, help)``
    triple to attach a tooltip.
    """
    if columns < 1:
        raise ValueError("columns must be positive")
    for offset in range(0, len(cards), columns):
        row = st.columns(columns)
        for column, card in zip(row, cards[offset : offset + columns]):
            label, value, *help_text = card
            column.metric(label, value, help=help_text[0] if help_text else None)


def render_placeholder(title: str, description: str) -> None:
    """Render an explicit placeholder for a future dashboard section."""
    with st.container(border=True):
        st.subheader(title)
        st.write(description)
        st.caption("Placeholder—no analytical output is calculated here yet.")


def render_data_warnings(data: DashboardData) -> None:
    """Display non-fatal data-quality limitations for the current selection."""
    missing_observations = int(
        data.prices.loc[:, required_tickers(data)].isna().sum().sum()
    )
    if missing_observations:
        st.warning(
            f"The selected inputs contain {missing_observations:,} missing price "
            "observations. Metrics will use their existing alignment rules."
        )


def render_disclaimer() -> None:
    """Render the dashboard's scope and advice disclaimer."""
    st.divider()
    st.caption(
        "For portfolio risk research and education only. Not investment "
        "advice. Does not predict returns."
    )


def render_quality_warnings(data: DashboardData) -> None:
    """Display non-fatal data-quality warnings for the current selection."""
    relevant_prices = data.prices.loc[:, required_tickers(data)]

    duplicate_dates = count_duplicate_dates(relevant_prices)
    if duplicate_dates:
        st.warning(
            f"The selected prices contain {duplicate_dates:,} duplicate date(s)."
        )

    lagging_tickers = find_unequal_coverage_tickers(relevant_prices)
    if lagging_tickers:
        st.warning(
            "Some tickers have less price history than others in the selected "
            f"range: {', '.join(lagging_tickers)}."
        )

    if data.complete_observations < SHORT_RANGE_OBSERVATIONS:
        st.warning(
            f"The selected range has only {data.complete_observations} complete "
            f"return observations, fewer than the {SHORT_RANGE_OBSERVATIONS} "
            "typically used for stable long-window metrics."
        )


def required_tickers(data: DashboardData) -> list[str]:
    """Return the portfolio's holding tickers plus its benchmark, in order."""
    tickers = list(data.portfolio_config.holdings.index)
    if data.portfolio_config.benchmark not in tickers:
        tickers.append(data.portfolio_config.benchmark)
    return tickers


def count_duplicate_dates(prices: pd.DataFrame) -> int:
    """Return the number of duplicate dates in a price DataFrame's index."""
    return int(prices.index.duplicated().sum())


def find_unequal_coverage_tickers(prices: pd.DataFrame) -> list[str]:
    """Return tickers with fewer observations than the best-covered ticker."""
    report = report_missing_data(prices)
    if report["observation_count"].nunique() <= 1:
        return []
    max_count = report["observation_count"].max()
    lagging = report.loc[report["observation_count"] < max_count, "ticker"]
    return lagging.tolist()
