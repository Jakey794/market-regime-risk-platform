"""Main portfolio overview dashboard page."""

from __future__ import annotations

import streamlit as st

from mrrp.dashboard.components import (
    render_context_summary,
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.formatting import format_decimal, format_percentage
from mrrp.dashboard.state import get_dashboard_data, get_dashboard_state
from mrrp.portfolio import build_portfolio_risk_summary, compute_cumulative_returns
from mrrp.reporting import (
    build_return_comparison_figure,
    build_time_series_figure,
    build_weights_figure,
)
from mrrp.risk import (
    compute_hhi,
    compute_rolling_portfolio_beta,
)
from mrrp.risk.drawdown import rolling_max_drawdown
from mrrp.risk.volatility import rolling_volatility


ROLLING_WINDOW = 63


render_page_header(
    "Portfolio Overview",
    "This dashboard summarizes risk, drawdown, beta, concentration, and "
    "benchmark-relative behavior. It does not predict returns.",
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
if data.portfolio_config.benchmark not in data.prices.columns:
    st.error(
        f"Benchmark {data.portfolio_config.benchmark} is missing from processed prices."
    )
    st.stop()

try:
    summary = build_portfolio_risk_summary(
        data.prices,
        data.portfolio_config,
    )
    cumulative_portfolio = compute_cumulative_returns(data.portfolio_returns)
    cumulative_benchmark = compute_cumulative_returns(data.benchmark_returns)
    rolling_volatility_values = rolling_volatility(
        data.portfolio_returns,
        window=ROLLING_WINDOW,
    ).rename("Rolling volatility")
    rolling_drawdown_values = rolling_max_drawdown(
        data.portfolio_returns,
        window=ROLLING_WINDOW,
    ).rename("Rolling max drawdown")
    rolling_beta_values = compute_rolling_portfolio_beta(
        data.portfolio_returns,
        data.benchmark_returns,
        window=ROLLING_WINDOW,
    ).rename("Rolling beta")
    valid_rolling_beta = rolling_beta_values.dropna()
    if valid_rolling_beta.empty:
        raise ValueError("Rolling beta has no valid observations for this date range")
    latest_rolling_beta = float(valid_rolling_beta.iloc[-1])
    hhi = compute_hhi(data.portfolio_config.holdings)
except ValueError as exc:
    st.error(f"Unable to calculate portfolio overview: {exc}")
    st.stop()

render_metric_cards(
    [
        ("Annualized return", format_percentage(summary.annualized_return)),
        ("Annualized volatility", format_percentage(summary.annualized_volatility)),
        ("Current drawdown", format_percentage(summary.current_drawdown)),
        ("Max drawdown", format_percentage(summary.max_drawdown)),
        (
            "Rolling beta vs benchmark",
            format_decimal(latest_rolling_beta),
            "Sensitivity to the benchmark over the trailing window; 1.0 means "
            "the portfolio tends to move in line with the benchmark.",
        ),
        (
            "Historical VaR 95",
            format_percentage(summary.var_95),
            "The 5th-percentile historical daily return; a rough estimate of "
            "a bad but not worst-case daily loss.",
        ),
        (
            "Historical CVaR 95",
            format_percentage(summary.cvar_95),
            "The average daily return on days at or below the VaR 95 threshold.",
        ),
        (
            "HHI concentration",
            format_decimal(hhi, decimals=3),
            "Herfindahl-Hirschman Index of holding weights; ranges from "
            "near 0 (very diversified) to 1.0 (a single holding).",
        ),
        (
            "Effective number of holdings",
            format_decimal(summary.effective_holdings),
            "Inverse of HHI; the number of equally-weighted holdings that "
            "would produce the same concentration.",
        ),
    ]
)

st.subheader("Portfolio and benchmark history")
st.plotly_chart(
    build_return_comparison_figure(
        cumulative_portfolio,
        cumulative_benchmark,
        benchmark_name=data.portfolio_config.benchmark,
    ),
    use_container_width=True,
)

st.subheader("Rolling risk")
left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(
        build_time_series_figure(
            rolling_volatility_values,
            title=f"Rolling {ROLLING_WINDOW}-day volatility",
            yaxis_title="Annualized volatility",
            tickformat=".0%",
        ),
        use_container_width=True,
    )
with right_column:
    st.plotly_chart(
        build_time_series_figure(
            rolling_drawdown_values,
            title=f"Rolling {ROLLING_WINDOW}-day maximum drawdown",
            yaxis_title="Drawdown",
            tickformat=".0%",
        ),
        use_container_width=True,
    )

left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(
        build_time_series_figure(
            rolling_beta_values,
            title=f"Rolling {ROLLING_WINDOW}-day beta",
            yaxis_title="Beta",
            tickformat=".2f",
        ),
        use_container_width=True,
    )
with right_column:
    st.plotly_chart(
        build_weights_figure(data.portfolio_config.holdings),
        use_container_width=True,
    )

render_disclaimer()
