"""Quarterly portfolio-risk memo preview and download page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from mrrp.backtest import BacktestConfig, RuleConfig, run_backtest
from mrrp.reporting.memo_context import (
    build_memo_summary_cards,
    classify_correlation_regime_from_features,
    classify_volatility_regime_from_features,
    regime_agreement_text,
)
from mrrp.dashboard.components import render_disclaimer, render_page_header
from mrrp.dashboard.paths import processed_data_dir, regime_feature_paths
from mrrp.dashboard.state import get_dashboard_data
from mrrp.data.cache import load_parquet
from mrrp.models import (
    GMMConfig,
    GMMRegimeModel,
    ThresholdConfig,
    ThresholdRegimeModel,
)
from mrrp.portfolio.metadata import compute_group_exposure, load_asset_metadata
from mrrp.reporting.memo import memo_inputs_from_summary, render_quarterly_memo
from mrrp.risk.scenarios import DeterministicShockScenario, HistoricalWindowScenario
from mrrp.risk.stress import (
    deterministic_asset_shock,
    historical_window_stress,
    worst_rolling_stress,
)


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

summary = build_memo_summary_cards(
    data.portfolio_returns,
    data.benchmark_returns,
    data.portfolio_config.holdings,
    data.asset_returns,
)

regime_info = {
    "volatility_regime": "unavailable",
    "correlation_regime": "unavailable",
    "agreement": "unavailable",
}
feature_paths = regime_feature_paths()
try:
    raw_features = load_parquet(feature_paths.raw).dropna(how="all")
    scaled = load_parquet(feature_paths.scaled).dropna()
    regime_info["volatility_regime"] = classify_volatility_regime_from_features(
        raw_features
    )
    regime_info["correlation_regime"] = classify_correlation_regime_from_features(
        raw_features
    )
    train = scaled.iloc[: max(60, int(len(scaled) * 0.7))]
    threshold = ThresholdRegimeModel(ThresholdConfig(n_states=3))
    gmm = GMMRegimeModel(
        GMMConfig(n_states=3, feature_columns=tuple(scaled.columns), random_seed=7)
    )
    threshold.fit(train)
    gmm.fit(train)
    labels = {
        "threshold": str(threshold.transform(scaled).labeled_states().iloc[-1]),
        "gmm": str(gmm.transform(scaled).labeled_states().iloc[-1]),
    }
    regime_info["agreement"] = regime_agreement_text(labels)
except (FileNotFoundError, OSError, RuntimeError, ValueError):
    pass

stress_results: dict[str, float] = {}
stress_path = processed_data_dir() / "stress_results.json"
if stress_path.exists():
    payload = json.loads(stress_path.read_text(encoding="utf-8"))
    for item in payload.get("results", []):
        stress_results[str(item["name"])] = float(item["portfolio_impact"])
else:
    holdings = data.portfolio_config.holdings
    prices = data.prices.loc[:, holdings.index.intersection(data.prices.columns)]
    start = prices.index[max(0, len(prices) - 63)]
    replay = historical_window_stress(
        prices,
        holdings,
        HistoricalWindowScenario(
            "recent_63d",
            str(start.date()),
            str(prices.index[-1].date()),
            "Recent window",
        ),
    )
    stress_results[replay.name] = replay.portfolio_impact
    for window in (21, 63, 126):
        worst = worst_rolling_stress(prices, holdings, window)
        stress_results[worst.name] = worst.portfolio_impact
    equity = deterministic_asset_shock(
        holdings,
        DeterministicShockScenario(
            "equity_down_10",
            "deterministic_asset_shock",
            {ticker: -0.10 for ticker in holdings.index},
            "Direct shock",
            {},
        ),
    )
    stress_results[equity.name] = equity.portfolio_impact

backtest_metrics: dict[str, float] = {}
try:
    features = load_parquet(feature_paths.raw).reindex(data.prices.index)
    backtest = run_backtest(
        data.prices,
        features if not features.empty else pd.DataFrame(index=data.prices.index),
        data.portfolio_config.benchmark,
        BacktestConfig(
            rule=RuleConfig(
                name="static_benchmark",
                base_weights=data.portfolio_config.holdings,
                defensive_weights=data.portfolio_config.holdings,
            ),
            transaction_cost_bps=10.0,
        ),
        config_name="static_benchmark",
    )
    backtest_metrics = {
        key: float(value) for key, value in backtest.metrics.to_dict().items()
    }
except (FileNotFoundError, OSError, ValueError):
    pass

factor_proxy_exposure: dict[str, float] = {}
try:
    metadata = load_asset_metadata(Path("configs/asset_metadata.yaml"))
    exposure = compute_group_exposure(
        data.portfolio_config.holdings, metadata, "factor_proxy"
    )
    factor_proxy_exposure = {str(k): float(v) for k, v in exposure.items()}
except (OSError, ValueError):
    pass

inputs = memo_inputs_from_summary(
    as_of=str(data.prices.index.max().date()),
    portfolio_name=data.portfolio_config.name,
    benchmark=data.portfolio_config.benchmark,
    summary_cards=summary,
    regime_info=regime_info,
    stress_results=stress_results,
    backtest_metrics=backtest_metrics,
    top_risk_contributors=summary["top_risk_contributors"],
    factor_proxy_exposure=factor_proxy_exposure,
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
    "Memo fields are populated from computed portfolio, regime, stress, and backtest "
    "outputs when available. Unavailable metrics are not estimated or invented."
)
render_disclaimer()
