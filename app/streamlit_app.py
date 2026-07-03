"""Entrypoint and shared controls for the Streamlit dashboard shell."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mrrp.dashboard.components import render_disclaimer, render_placeholder
from mrrp.dashboard.state import (
    BENCHMARK_KEY,
    DATE_RANGE_KEY,
    PORTFOLIO_KEY,
    initialize_dashboard_state,
)
from mrrp.data.cache import load_parquet
from mrrp.portfolio import load_portfolio_config


st.set_page_config(
    page_title="Market Regime + Portfolio Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT_DIR / "data" / "processed" / "adjusted_close.parquet"
PORTFOLIO_CONFIG_PATH = ROOT_DIR / "configs" / "sample_portfolio.yaml"


@st.cache_data(show_spinner=False)
def load_dashboard_options(
    prices_path: str,
    portfolio_config_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    """Load the values needed by shared shell controls."""
    prices = load_parquet(prices_path)
    portfolio = load_portfolio_config(portfolio_config_path)
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Processed prices must have a non-empty DatetimeIndex")

    portfolios = (portfolio.name,)
    benchmarks = tuple(str(column) for column in prices.columns)
    return portfolios, benchmarks, prices.index.min(), prices.index.max()


def render_shared_sidebar(
    portfolios: tuple[str, ...],
    benchmarks: tuple[str, ...],
    minimum_date: pd.Timestamp,
    maximum_date: pd.Timestamp,
) -> None:
    """Render controls that persist across every dashboard page."""
    initialize_dashboard_state(
        st.session_state,
        portfolios=portfolios,
        benchmarks=benchmarks,
        minimum_date=minimum_date.date(),
        maximum_date=maximum_date.date(),
        default_benchmark="SPY",
    )

    with st.sidebar:
        st.header("Analysis settings")
        st.selectbox("Portfolio", portfolios, key=PORTFOLIO_KEY)
        st.selectbox("Benchmark", benchmarks, key=BENCHMARK_KEY)
        st.date_input(
            "Analysis period",
            min_value=minimum_date.date(),
            max_value=maximum_date.date(),
            key=DATE_RANGE_KEY,
        )
        st.divider()
        st.caption("Risk modeling dashboard—not a prediction app or financial advice.")


def run_navigation() -> None:
    """Run modern navigation, with a basic landing page for older Streamlit."""
    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page(
                "pages/1_Portfolio_Overview.py",
                title="Portfolio Overview",
                icon=":material/dashboard:",
                default=True,
            ),
            st.Page(
                "pages/2_Risk_Metrics.py",
                title="Risk Metrics",
                icon=":material/monitoring:",
            ),
            st.Page(
                "pages/3_Correlation_Beta.py",
                title="Correlation & Beta",
                icon=":material/hub:",
            ),
            st.Page(
                "pages/4_Data_Quality.py",
                title="Data Quality",
                icon=":material/fact_check:",
            ),
        ]
        st.navigation(pages).run()
        return

    st.title("Market Regime + Portfolio Risk Platform")
    render_placeholder(
        "Dashboard pages",
        "Use the Pages navigation in the sidebar to open a dashboard section.",
    )
    render_disclaimer()


def main() -> None:
    """Initialize shared state and run the selected dashboard page."""
    try:
        portfolios, benchmarks, minimum_date, maximum_date = load_dashboard_options(
            str(PRICES_PATH),
            str(PORTFOLIO_CONFIG_PATH),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Unable to initialize dashboard controls: {exc}")
        st.stop()

    render_shared_sidebar(portfolios, benchmarks, minimum_date, maximum_date)
    run_navigation()


if __name__ == "__main__":
    main()
