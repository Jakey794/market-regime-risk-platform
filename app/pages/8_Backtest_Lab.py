"""No-look-ahead backtest dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mrrp.backtest import BacktestConfig, RuleConfig, run_backtest
from mrrp.dashboard.components import render_disclaimer, render_page_header
from mrrp.dashboard.paths import regime_feature_paths
from mrrp.dashboard.state import get_dashboard_data
from mrrp.data.cache import load_parquet


render_page_header(
    "Backtest Lab",
    "Historical simulations with shifted signals, explicit rebalancing, turnover, "
    "and proportional costs. Results are not guaranteed alpha or future performance.",
)
try:
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.warning(str(exc))
    render_disclaimer()
    st.stop()

rule_name = st.selectbox(
    "Allocation rule",
    ["static_benchmark", "volatility_target", "combined_risk_aware"],
)
frequency = st.selectbox("Rebalance frequency", ["M", "W", "Q"])
cost_bps = st.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0)
try:
    features = load_parquet(regime_feature_paths().raw).reindex(data.prices.index)
except (FileNotFoundError, OSError, ValueError):
    features = pd.DataFrame(index=data.prices.index)
    if rule_name != "static_benchmark":
        st.warning("Risk-aware rules require feature artifacts. Run `make features`.")
        render_disclaimer()
        st.stop()

rule = RuleConfig(
    name=rule_name,
    base_weights=data.portfolio_config.holdings,
    defensive_weights=data.portfolio_config.holdings,
)
try:
    result = run_backtest(
        data.prices,
        features,
        data.portfolio_config.benchmark,
        BacktestConfig(
            rule=rule,
            rebalance_frequency=frequency,
            transaction_cost_bps=cost_bps,
        ),
        config_name=rule_name,
    )
except ValueError as exc:
    st.warning(f"Backtest unavailable for this selection: {exc}")
    render_disclaimer()
    st.stop()

metrics = result.metrics.to_dict()
columns = st.columns(4)
for column, key in zip(
    columns,
    ["cagr", "annualized_volatility", "max_drawdown", "transaction_cost_drag"],
    strict=True,
):
    column.metric(key.replace("_", " ").title(), f"{float(metrics[key]):.2%}")
wealth = pd.DataFrame(
    {
        "Strategy": (1 + result.strategy_returns).cumprod(),
        "Benchmark": (1 + result.benchmark_returns).cumprod(),
    }
)
st.line_chart(wealth)
st.subheader("Portfolio weights")
st.area_chart(result.weights)
st.subheader("Diagnostics")
st.dataframe(pd.Series(metrics, name="value").to_frame(), width="stretch")
st.warning(
    "Do not interpret this historical simulation as guaranteed alpha, investment "
    "advice, or evidence that the rule will reduce future losses."
)
render_disclaimer()
