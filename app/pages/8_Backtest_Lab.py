"""No-look-ahead backtest dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mrrp.backtest import BacktestConfig, RuleConfig, run_backtest
from mrrp.dashboard.components import (
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.backtest import load_or_build_backtest_features
from mrrp.dashboard.paths import DEFAULT_CONFIG_DIR, regime_feature_paths
from mrrp.dashboard.state import get_dashboard_data


RULE_OPTIONS = [
    "static_benchmark",
    "volatility_target",
    "drawdown_guardrail",
    "high_vol_derisk",
    "high_corr_derisk",
    "combined_risk_aware",
]


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

rule_name = st.selectbox("Allocation rule", RULE_OPTIONS)
frequency = st.selectbox("Rebalance frequency", ["M", "W", "Q"])
cost_bps = st.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0)
try:
    features, used_session_fallback = load_or_build_backtest_features(
        data.prices,
        data.portfolio_config,
        artifact_path=regime_feature_paths().raw,
        feature_config_path=DEFAULT_CONFIG_DIR / "regime_features.yaml",
    )
except ValueError as exc:
    st.warning(f"Backtest unavailable for this selection: {exc}")
    render_disclaimer()
    st.stop()

if used_session_fallback:
    st.info(
        "Regime feature artifacts are unavailable, so this session derived "
        "leakage-safe features from the selected demo prices."
    )

# Defensive sleeve reduces concentration toward equal-weight risk-off mix.
# Cash/risk-off is represented by blending toward these defensive weights when
# the risk scalar falls below 1 (no unlisted cash ticker required).
base = data.portfolio_config.holdings
defensive = pd.Series(1.0 / len(base), index=base.index)

rule = RuleConfig(
    name=rule_name,
    base_weights=base,
    defensive_weights=defensive,
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
render_metric_cards(
    [
        ("CAGR", f"{float(metrics['cagr']):.2%}"),
        ("Ann. volatility", f"{float(metrics['annualized_volatility']):.2%}"),
        ("Sharpe", f"{float(metrics['sharpe']):.2f}"),
        ("Sortino", f"{float(metrics['sortino']):.2f}"),
        ("Max drawdown", f"{float(metrics['max_drawdown']):.2%}"),
        ("Calmar", f"{float(metrics['calmar']):.2f}"),
        ("Tracking error", f"{float(metrics['tracking_error']):.2%}"),
        ("Turnover", f"{float(metrics['turnover']):.2%}"),
        ("Cost drag", f"{float(metrics['transaction_cost_drag']):.2%}"),
        ("Worst month", f"{float(metrics['worst_month']):.2%}"),
        (
            "Monthly hit rate vs benchmark",
            f"{float(metrics['fraction_outperforming_months']):.2%}",
        ),
        ("Rebalance dates", str(len(result.decision_dates))),
    ],
    columns=4,
)
wealth = pd.DataFrame(
    {
        "Strategy": (1 + result.strategy_returns).cumprod(),
        "Benchmark": (1 + result.benchmark_returns).cumprod(),
    }
)
st.subheader("Equity curves")
st.line_chart(wealth)
drawdown = wealth["Strategy"] / wealth["Strategy"].cummax() - 1.0
st.subheader("Strategy drawdown")
st.area_chart(drawdown.rename("drawdown"))
st.subheader("Portfolio weights")
st.area_chart(result.weights)
st.subheader("Diagnostics")
st.dataframe(pd.Series(metrics, name="value").to_frame(), width="stretch")
st.caption(
    "Timing contract: signals from date t are shifted before execution so same-close "
    "trading cannot occur. Cash/risk-off is handled by blending toward defensive "
    "weights when the risk scalar falls below 1."
)
st.warning(
    "Do not interpret this historical simulation as guaranteed alpha, investment "
    "advice, or evidence that the rule will reduce future losses."
)
render_disclaimer()
