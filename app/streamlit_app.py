"""Entrypoint and shared controls for the Streamlit dashboard shell."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mrrp.dashboard.components import (
    render_data_warnings,
    render_disclaimer,
    render_placeholder,
)
from mrrp.dashboard.loaders import (
    DashboardOptions,
    InsufficientObservationsError,
    load_dashboard_dataset,
    load_dashboard_options,
)
from mrrp.dashboard.state import (
    BENCHMARK_KEY,
    DATE_RANGE_KEY,
    PORTFOLIO_KEY,
    get_dashboard_state,
    initialize_dashboard_state,
    set_dashboard_data,
)

st.set_page_config(
    page_title="Market Regime + Portfolio Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT_DIR / "data" / "processed" / "adjusted_close.parquet"
PORTFOLIO_CONFIG_PATH = ROOT_DIR / "configs" / "sample_portfolio.yaml"


def render_shared_sidebar(options: DashboardOptions) -> None:
    """Render controls that persist across every dashboard page."""
    initialize_dashboard_state(
        st.session_state,
        portfolios=options.portfolios,
        benchmarks=options.benchmarks,
        minimum_date=options.minimum_date.date(),
        maximum_date=options.maximum_date.date(),
        default_benchmark="SPY",
    )

    with st.sidebar:
        st.header("Analysis settings")
        st.selectbox("Portfolio", options.portfolios, key=PORTFOLIO_KEY)
        st.selectbox("Benchmark", options.benchmarks, key=BENCHMARK_KEY)
        st.date_input(
            "Analysis period",
            min_value=options.minimum_date.date(),
            max_value=options.maximum_date.date(),
            key=DATE_RANGE_KEY,
        )
        st.divider()
        if st.button(
            "Refresh cached data",
            help="Clear cached price and portfolio data and reload. Run "
            "`make data` first if the underlying file itself is stale.",
        ):
            load_dashboard_options.clear()
            load_dashboard_dataset.clear()
            st.rerun()
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
        with st.spinner("Loading market data..."):
            options = load_dashboard_options(
                str(PRICES_PATH),
                str(PORTFOLIO_CONFIG_PATH),
            )
    except FileNotFoundError:
        st.error("Processed price data is unavailable. Run `make data` and try again.")
        st.stop()
    except (OSError, ValueError) as exc:
        st.error(f"Unable to initialize dashboard controls: {exc}")
        st.stop()

    render_shared_sidebar(options)
    state = get_dashboard_state(st.session_state)
    try:
        with st.spinner("Preparing portfolio analytics..."):
            data = load_dashboard_dataset(
                str(PRICES_PATH),
                str(PORTFOLIO_CONFIG_PATH),
                start_date=state.start_date,
                end_date=state.end_date,
                benchmark=state.benchmark,
            )
    except InsufficientObservationsError as exc:
        st.warning(str(exc))
        st.stop()
    except FileNotFoundError:
        st.error("Processed price data is unavailable. Run `make data` and try again.")
        st.stop()
    except (OSError, ValueError) as exc:
        st.error(f"Unable to prepare dashboard data: {exc}")
        st.stop()

    set_dashboard_data(st.session_state, data)
    render_data_warnings(data)
    run_navigation()


if __name__ == "__main__":
    main()
