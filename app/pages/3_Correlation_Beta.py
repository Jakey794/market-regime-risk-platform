"""Correlation and beta dashboard page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mrrp.dashboard.components import (
    render_context_summary,
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.formatting import format_decimal, format_percentage
from mrrp.dashboard.state import get_dashboard_data, get_dashboard_state
from mrrp.portfolio import compute_group_exposure, load_asset_metadata
from mrrp.reporting import (
    build_bar_figure,
    build_correlation_heatmap_figure,
    build_time_series_figure,
)
from mrrp.risk import (
    build_correlation_summary,
    compute_asset_betas,
    compute_correlation_matrix,
    compute_effective_num_holdings,
    compute_hhi,
    compute_percent_risk_contribution,
    compute_portfolio_beta,
    compute_rolling_mean_pairwise_correlation,
    compute_rolling_portfolio_beta,
    compute_top_n_weight,
)
from mrrp.utils.config import ConfigError


ROLLING_WINDOW = 63
ASSET_METADATA_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "asset_metadata.yaml"
)


render_page_header(
    "Correlation & Beta",
    "Am I actually diversified, or do I have hidden overlap? Correlation, "
    "benchmark beta, and concentration diagnostics from the existing "
    "deterministic engine. This does not predict future performance.",
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
    holdings = data.portfolio_config.holdings
    correlation_summary = build_correlation_summary(
        data.asset_returns,
        holdings,
        window=ROLLING_WINDOW,
    ).iloc[0]
    rolling_correlation = compute_rolling_mean_pairwise_correlation(
        data.asset_returns,
        window=ROLLING_WINDOW,
    ).rename("Rolling mean pairwise correlation")
    correlation_matrix = compute_correlation_matrix(data.asset_returns)

    portfolio_beta = compute_portfolio_beta(
        data.portfolio_returns, data.benchmark_returns
    )
    rolling_portfolio_beta = compute_rolling_portfolio_beta(
        data.portfolio_returns,
        data.benchmark_returns,
        window=ROLLING_WINDOW,
    ).rename("Rolling portfolio beta")
    asset_betas = compute_asset_betas(
        data.asset_returns, data.benchmark_returns
    ).sort_values(ascending=False)

    hhi = compute_hhi(holdings)
    effective_holdings = compute_effective_num_holdings(holdings)
    top_1_weight = compute_top_n_weight(holdings, n=1)
    top_3_weight = compute_top_n_weight(holdings, n=3)
    top_5_weight = compute_top_n_weight(holdings, n=5)
    risk_contribution = compute_percent_risk_contribution(
        data.asset_returns,
        holdings,
    ).sort_values(ascending=False)
except ValueError as exc:
    st.error(f"Unable to calculate correlation and beta diagnostics: {exc}")
    st.stop()

st.subheader("Correlation")
with st.expander("What do these metrics mean?"):
    st.write(
        "High pairwise correlation between holdings means they tend to move "
        "together, reducing the diversification benefit of holding several "
        "positions. A correlation near 1.0 across most pairs suggests hidden "
        "overlap even if the holdings look different on paper."
    )
render_metric_cards(
    [
        (
            "Mean pairwise correlation",
            format_decimal(correlation_summary["mean_pairwise_corr"]),
            "Average correlation across all holding pairs; values near 1.0 "
            "suggest holdings move together with little diversification "
            "benefit.",
        ),
        (
            "Max pairwise correlation",
            format_decimal(correlation_summary["max_pairwise_corr"]),
        ),
        (
            "Latest rolling correlation",
            format_decimal(correlation_summary["current_rolling_corr"]),
        ),
    ]
)
st.caption(f"Current correlation regime: {correlation_summary['correlation_regime']}")

left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(
        build_time_series_figure(
            rolling_correlation,
            title=f"Rolling {ROLLING_WINDOW}-day mean pairwise correlation",
            yaxis_title="Mean pairwise correlation",
            tickformat=".2f",
        ),
        use_container_width=True,
    )
with right_column:
    st.plotly_chart(
        build_correlation_heatmap_figure(
            correlation_matrix,
            title="Latest asset correlation",
        ),
        use_container_width=True,
    )

st.subheader("Beta")
with st.expander("What do these metrics mean?"):
    st.write(
        "Beta measures sensitivity to the selected benchmark. A portfolio or "
        "holding with a beta near 1.0 tends to move with the benchmark; "
        "higher values amplify benchmark moves and lower values dampen them."
    )
render_metric_cards(
    [
        (
            f"Portfolio beta vs {data.portfolio_config.benchmark}",
            format_decimal(portfolio_beta),
        ),
    ]
)
left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(
        build_time_series_figure(
            rolling_portfolio_beta,
            title=f"Rolling {ROLLING_WINDOW}-day portfolio beta",
            yaxis_title="Beta",
            tickformat=".2f",
        ),
        use_container_width=True,
    )
with right_column:
    st.plotly_chart(
        build_bar_figure(
            asset_betas,
            title=f"Asset beta vs {data.portfolio_config.benchmark}",
            xaxis_title="Asset",
            yaxis_title="Beta",
            tickformat=".2f",
        ),
        use_container_width=True,
    )

st.subheader("Concentration")
with st.expander("What do these metrics mean?"):
    st.write(
        "HHI and effective number of holdings summarize how spread out the "
        "portfolio's weights are. Risk contribution shows each holding's "
        "share of total portfolio variance, which can differ from its "
        "weight if it is more volatile or more correlated with the rest of "
        "the portfolio."
    )
render_metric_cards(
    [
        (
            "HHI",
            format_decimal(hhi, decimals=3),
            "Herfindahl-Hirschman Index of holding weights; ranges from "
            "near 0 (very diversified) to 1.0 (a single holding).",
        ),
        ("Effective number of holdings", format_decimal(effective_holdings)),
        ("Top-1 weight", format_percentage(top_1_weight)),
        ("Top-3 weight", format_percentage(top_3_weight)),
        ("Top-5 weight", format_percentage(top_5_weight)),
    ]
)
st.plotly_chart(
    build_bar_figure(
        risk_contribution,
        title="Asset contribution to portfolio variance",
        xaxis_title="Asset",
        yaxis_title="Risk contribution",
        tickformat=".0%",
    ),
    use_container_width=True,
)

st.subheader("Sector/factor proxy exposure")
with st.expander("What does this show?"):
    st.write(
        "This groups holdings using a fixed, approximate label per sample "
        "ETF (for example, 'us_growth_tech_heavy' for QQQ). It is a proxy "
        "for illustration only, not a true factor model, and only covers "
        "the sample ETF universe."
    )
try:
    asset_metadata = load_asset_metadata(ASSET_METADATA_PATH)
    factor_exposure = compute_group_exposure(
        holdings, asset_metadata, group_key="factor_proxy"
    )
except (FileNotFoundError, ConfigError) as exc:
    st.warning(f"Sector/factor proxy exposure is unavailable: {exc}")
else:
    st.caption(
        "Approximate proxy grouping for sample ETFs only, not a true factor model."
    )
    st.plotly_chart(
        build_bar_figure(
            factor_exposure,
            title="Approximate sector/factor proxy exposure",
            xaxis_title="Factor proxy",
            yaxis_title="Portfolio weight",
            tickformat=".0%",
        ),
        use_container_width=True,
    )

render_disclaimer()
