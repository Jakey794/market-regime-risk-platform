"""Data quality dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mrrp.data.validators import report_missing_data
from mrrp.dashboard.components import (
    render_context_summary,
    render_disclaimer,
    render_metric_cards,
    render_page_header,
    render_quality_warnings,
    required_tickers,
)
from mrrp.dashboard.formatting import format_decimal, format_percentage
from mrrp.dashboard.loaders import load_dashboard_options
from mrrp.dashboard.state import get_dashboard_data, get_dashboard_state
from mrrp.utils.dates import days_since


ROOT_DIR = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT_DIR / "data" / "processed" / "adjusted_close.parquet"
PORTFOLIO_CONFIG_PATH = ROOT_DIR / "configs" / "sample_portfolio.yaml"
STALE_THRESHOLD_DAYS = 7
RUN_MAKE_DATA_MESSAGE = "Run `make data` if data is missing or stale."


render_page_header(
    "Data Quality",
    "Visibility into the inputs supporting every historical risk estimate.",
)

try:
    state = get_dashboard_state(st.session_state)
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.error(f"Dashboard inputs are unavailable: {exc}")
    st.stop()

render_context_summary(state)

if data.prices.empty or data.portfolio_returns.dropna().empty:
    st.error("The selected portfolio contains no usable observations.")
    st.stop()

tickers = required_tickers(data)
relevant_prices = data.prices.loc[:, tickers]
missing_report = report_missing_data(relevant_prices)

st.subheader("Coverage")
render_metric_cards(
    [
        ("Tickers", format_decimal(len(tickers), decimals=0)),
        ("Benchmark", data.portfolio_config.benchmark),
        ("Selected range start", str(data.prices.index.min().date())),
        ("Selected range end", str(data.prices.index.max().date())),
        (
            "Complete return observations",
            format_decimal(data.complete_observations, decimals=0),
            "Observations remaining after aligning every holding and the "
            "benchmark and dropping any incomplete rows.",
        ),
    ]
)
render_quality_warnings(data)

st.subheader("Missing observations")
st.dataframe(missing_report, width="stretch")
dropped_rows = len(data.prices) - data.complete_observations
render_metric_cards(
    [
        (
            "Duplicate dates",
            format_decimal(int(relevant_prices.index.duplicated().sum()), decimals=0),
        ),
        (
            "Dropped rows after alignment",
            format_decimal(dropped_rows, decimals=0),
            "Rows present in the selected range but removed because at least "
            "one holding or the benchmark was missing on that date.",
        ),
    ]
)

st.subheader("Freshness and provenance")
try:
    options = load_dashboard_options(str(PRICES_PATH), str(PORTFOLIO_CONFIG_PATH))
    latest_available_date = options.maximum_date.date()
    staleness_days = days_since(latest_available_date)
    cache_refreshed = pd.Timestamp(PRICES_PATH.stat().st_mtime, unit="s")
except (FileNotFoundError, OSError, ValueError) as exc:
    st.warning(f"Unable to check data freshness: {exc}. {RUN_MAKE_DATA_MESSAGE}")
else:
    render_metric_cards(
        [
            ("Latest available date", str(latest_available_date)),
            ("Days since latest date", format_decimal(staleness_days, decimals=0)),
            ("Cache last refreshed", cache_refreshed.strftime("%Y-%m-%d %H:%M")),
        ]
    )
    if staleness_days > STALE_THRESHOLD_DAYS:
        st.warning(
            f"The latest available date is {staleness_days} days old. "
            f"{RUN_MAKE_DATA_MESSAGE}"
        )
    else:
        st.caption("Data looks up to date.")

st.subheader("Validation")
weight_sum = float(data.portfolio_config.holdings.sum())
render_metric_cards(
    [
        (
            "Portfolio weight sum",
            format_percentage(weight_sum),
            "Portfolio weights are validated to sum to 1.0 (within tolerance) "
            "when the dashboard loads.",
        ),
        ("Allow short positions", str(data.portfolio_config.allow_short)),
    ]
)
if data.portfolio_config.benchmark in data.prices.columns:
    st.caption(
        f"Benchmark {data.portfolio_config.benchmark} is present in the "
        "selected prices."
    )
else:
    st.error(
        f"Benchmark {data.portfolio_config.benchmark} is missing from processed prices."
    )
    st.stop()

render_disclaimer()
