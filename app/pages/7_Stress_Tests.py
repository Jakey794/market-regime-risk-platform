"""Portfolio stress-test dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mrrp.dashboard.components import render_disclaimer, render_page_header
from mrrp.dashboard.state import get_dashboard_data
from mrrp.risk.scenarios import DeterministicShockScenario, HistoricalWindowScenario
from mrrp.risk.stress import (
    deterministic_asset_shock,
    historical_window_stress,
    regime_conditioned_stress,
    worst_rolling_stress,
)


render_page_header(
    "Stress Tests",
    "Historical replays and explicit deterministic estimates for portfolio-risk "
    "research. Scenario impacts are not predictions.",
)
try:
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.warning(str(exc))
    render_disclaimer()
    st.stop()

historical_tab, deterministic_tab, custom_tab, regime_tab = st.tabs(
    ["Historical", "Deterministic", "Custom", "Regime-conditioned"]
)
with historical_tab:
    start = data.prices.index[max(0, len(data.prices) - 63)]
    scenario = HistoricalWindowScenario(
        "Recent 63-session window",
        str(start.date()),
        str(data.prices.index[-1].date()),
        "Observed recent sample window.",
    )
    replay = historical_window_stress(
        data.prices.loc[:, data.portfolio_config.holdings.index],
        data.portfolio_config.holdings,
        scenario,
    )
    worst = worst_rolling_stress(
        data.prices.loc[:, data.portfolio_config.holdings.index],
        data.portfolio_config.holdings,
        21,
    )
    st.metric("Historical replay impact", f"{replay.portfolio_impact:.2%}")
    st.metric("Worst trailing 21-session impact", f"{worst.portfolio_impact:.2%}")
    st.caption(
        "Methodology: exact historical replay of observed adjusted-close returns."
    )

with deterministic_tab:
    shocks = {ticker: -0.10 for ticker in data.portfolio_config.holdings.index}
    result = deterministic_asset_shock(
        data.portfolio_config.holdings,
        DeterministicShockScenario(
            "Broad equity -10%",
            "deterministic_asset_shock",
            shocks,
            "Direct return shock.",
            {},
        ),
    )
    st.metric("Deterministic impact", f"{result.portfolio_impact:.2%}")
    st.dataframe(result.asset_contribution.to_frame(), width="stretch")

with custom_tab:
    shock = st.slider("Uniform asset shock", -0.50, 0.20, -0.10, 0.01)
    custom = deterministic_asset_shock(
        data.portfolio_config.holdings,
        DeterministicShockScenario(
            "Custom uniform shock",
            "deterministic_asset_shock",
            {ticker: shock for ticker in data.portfolio_config.holdings.index},
            "User-selected direct return shock.",
            {},
        ),
    )
    st.metric("Custom scenario impact", f"{custom.portfolio_impact:.2%}")

with regime_tab:
    regimes = pd.Series(
        "all available observations", index=data.portfolio_returns.index
    )
    selected = st.selectbox("Conditioning sample", ["all available observations"])
    conditioned = regime_conditioned_stress(
        data.portfolio_returns,
        data.asset_returns,
        data.portfolio_config.holdings,
        regimes,
        regime_label=selected,
    )
    st.dataframe(
        pd.DataFrame(
            {
                "scenario": [item.name for item in conditioned],
                "impact": [item.portfolio_impact for item in conditioned],
                "methodology": [item.methodology for item in conditioned],
            }
        ),
        width="stretch",
    )
    st.caption(
        "A fitted regime-state artifact was not selected, so this fallback summarizes "
        "all observations. Regime conditioning is retrospective, not predictive."
    )

render_disclaimer()
