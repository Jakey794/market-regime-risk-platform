"""Quarterly portfolio-risk memo preview and download page."""

from __future__ import annotations

import streamlit as st

from mrrp.dashboard.components import render_disclaimer, render_page_header
from mrrp.dashboard.state import get_dashboard_data
from mrrp.reporting.memo import memo_inputs_from_summary, render_quarterly_memo


render_page_header(
    "Quarterly Memo",
    "A deterministic Markdown risk-review template populated only by available "
    "dashboard outputs. Missing empirical fields are marked unavailable.",
)
try:
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.warning(str(exc))
    render_disclaimer()
    st.stop()

inputs = memo_inputs_from_summary(
    as_of=str(data.prices.index.max().date()),
    portfolio_name=data.portfolio_config.name,
    benchmark=data.portfolio_config.benchmark,
    summary_cards={},
    regime_info={},
    stress_results={},
    backtest_metrics={},
    top_risk_contributors={},
    factor_proxy_exposure={},
)
memo = render_quarterly_memo(inputs)
st.subheader("Memo preview")
st.markdown(memo)
st.download_button(
    "Download Markdown",
    memo,
    file_name="quarterly_portfolio_risk_memo.md",
    mime="text/markdown",
)
st.caption(
    "Unavailable metrics are not estimated or invented. Generate model, stress, and "
    "backtest artifacts before producing a fully populated research memo."
)
render_disclaimer()
