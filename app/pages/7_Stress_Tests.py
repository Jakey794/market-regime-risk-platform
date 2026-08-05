"""Portfolio stress-test dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mrrp.dashboard.components import (
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.state import get_dashboard_data
from mrrp.portfolio.metadata import compute_group_exposure, load_asset_metadata
from mrrp.risk.scenarios import DeterministicShockScenario, HistoricalWindowScenario
from mrrp.risk.stress import (
    benchmark_beta_shock,
    correlation_shock_estimate,
    deterministic_asset_shock,
    historical_window_stress,
    rank_stress_results,
    regime_conditioned_stress,
    volatility_shock_estimate,
    worst_loss_contributor,
    worst_rolling_stress,
)


render_page_header(
    "Stress Tests",
    "Historical replays and explicit deterministic estimates for portfolio-risk "
    "research. Scenario impacts are not predictions or forecasts.",
)
try:
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.warning(str(exc))
    render_disclaimer()
    st.stop()

holdings = data.portfolio_config.holdings
holding_prices = data.prices.loc[:, holdings.index.intersection(data.prices.columns)]
results = []

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
    replay = historical_window_stress(holding_prices, holdings, scenario)
    worst_1m = worst_rolling_stress(holding_prices, holdings, 21)
    worst_3m = worst_rolling_stress(holding_prices, holdings, 63)
    worst_6m = worst_rolling_stress(holding_prices, holdings, 126)
    results.extend([replay, worst_1m, worst_3m, worst_6m])
    render_metric_cards(
        [
            ("Historical replay", f"{replay.portfolio_impact:.2%}"),
            ("Worst 1M (21d)", f"{worst_1m.portfolio_impact:.2%}"),
            ("Worst 3M (63d)", f"{worst_3m.portfolio_impact:.2%}"),
            ("Worst 6M (126d)", f"{worst_6m.portfolio_impact:.2%}"),
        ],
        columns=4,
    )
    st.caption(
        "Methodology: exact historical replay of observed adjusted-close returns."
    )
    st.dataframe(worst_1m.asset_contribution.to_frame("contribution"), width="stretch")
    st.caption(f"Worst contributor (1M): {worst_loss_contributor(worst_1m)}")

with deterministic_tab:
    shocks = {ticker: -0.10 for ticker in holdings.index}
    equity = deterministic_asset_shock(
        holdings,
        DeterministicShockScenario(
            "Broad equity -10%",
            "deterministic_asset_shock",
            shocks,
            "Direct return shock.",
            {},
        ),
    )
    vol = volatility_shock_estimate(data.asset_returns, holdings, 1.5)
    corr = correlation_shock_estimate(data.asset_returns, holdings, 0.90)
    beta = benchmark_beta_shock(
        data.asset_returns,
        data.benchmark_returns,
        holdings,
        -0.10,
    )
    results.extend([equity, vol, corr, beta])
    render_metric_cards(
        [
            ("Equity -10%", f"{equity.portfolio_impact:.2%}"),
            ("Vol shock estimate", f"{vol.portfolio_impact:.2%}"),
            ("Correlation shock", f"{corr.portfolio_impact:.2%}"),
            ("Beta/benchmark -10%", f"{beta.portfolio_impact:.2%}"),
        ],
        columns=4,
    )
    st.caption(
        "Volatility, correlation, and beta shocks are explicit approximations — "
        "not historical replays."
    )
    try:
        metadata = load_asset_metadata("configs/asset_metadata.yaml")
        regional = compute_group_exposure(holdings, metadata, "region")
        sector = compute_group_exposure(holdings, metadata, "sector_proxy")
        st.subheader("Sector / regional proxy exposure")
        st.dataframe(
            pd.concat(
                [sector.rename("sector_proxy"), regional.rename("region")],
                axis=1,
            ),
            width="stretch",
        )
        if "Technology" in sector.index:
            tech_tickers = [
                ticker
                for ticker, attrs in metadata.items()
                if attrs.get("sector_proxy") == "Technology"
                and ticker in holdings.index
            ]
            tech_shocks = {ticker: -0.15 for ticker in tech_tickers}
            if tech_shocks:
                sector_shock = deterministic_asset_shock(
                    holdings,
                    DeterministicShockScenario(
                        "Technology sector proxy -15%",
                        "sector_proxy_shock",
                        tech_shocks,
                        "Sector ETF proxy shock, not a statistical factor model.",
                        {"proxy_disclaimer": "Not a full factor model."},
                    ),
                )
                results.append(sector_shock)
                st.metric(
                    "Technology sector proxy shock",
                    f"{sector_shock.portfolio_impact:.2%}",
                )
    except (OSError, ValueError) as exc:
        st.caption(f"Sector/regional exposure unavailable: {exc}")

with custom_tab:
    shock = st.slider("Uniform asset shock", -0.50, 0.20, -0.10, 0.01)
    custom = deterministic_asset_shock(
        holdings,
        DeterministicShockScenario(
            "Custom uniform shock",
            "deterministic_asset_shock",
            {ticker: shock for ticker in holdings.index},
            "User-selected direct return shock.",
            {},
        ),
    )
    results.append(custom)
    st.metric("Custom scenario impact", f"{custom.portfolio_impact:.2%}")
    st.dataframe(custom.asset_contribution.to_frame("contribution"), width="stretch")
    st.caption(f"Worst contributor: {worst_loss_contributor(custom)}")

with regime_tab:
    regimes = pd.Series(
        "all available observations", index=data.portfolio_returns.index
    )
    selected = st.selectbox("Conditioning sample", ["all available observations"])
    conditioned = regime_conditioned_stress(
        data.portfolio_returns,
        data.asset_returns,
        holdings,
        regimes,
        regime_label=selected,
    )
    results.extend(conditioned)
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

st.subheader("Scenario ranking")
ranking = rank_stress_results(results)
st.dataframe(ranking, width="stretch")
st.info(
    "Assumptions and limitations: historical windows depend on ETF history coverage; "
    "deterministic shocks are linear; covariance-based estimates ignore crisis "
    "non-linearities; none of these outputs are forecasts."
)
render_disclaimer()
