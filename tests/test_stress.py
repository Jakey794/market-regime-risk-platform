from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.scenarios import DeterministicShockScenario, HistoricalWindowScenario
from mrrp.risk.stress import (
    benchmark_beta_shock,
    correlation_shock_estimate,
    deterministic_asset_shock,
    historical_window_stress,
    rank_stress_results,
    volatility_shock_estimate,
    worst_loss_contributor,
    worst_rolling_stress,
)


def _portfolio() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {"A": np.linspace(-0.01, 0.01, 80), "B": np.linspace(0.005, -0.005, 80)},
        index=index,
    )
    prices = 100 * (1 + returns).cumprod()
    return prices, pd.Series({"A": 0.6, "B": 0.4})


def test_historical_and_worst_rolling_stress_are_computed() -> None:
    prices, weights = _portfolio()
    scenario = HistoricalWindowScenario(
        "sample",
        str(prices.index[1].date()),
        str(prices.index[-1].date()),
        "synthetic",
    )
    replay = historical_window_stress(prices, weights, scenario)
    worst = worst_rolling_stress(prices, weights, 10)
    assert np.isfinite(replay.portfolio_impact)
    assert worst.portfolio_impact <= 0
    assert replay.methodology == "historical_replay"


def test_deterministic_and_volatility_shocks_are_explicit_estimates() -> None:
    prices, weights = _portfolio()
    scenario = DeterministicShockScenario(
        "direct",
        "deterministic_asset_shock",
        {"A": -0.1, "B": -0.2},
        "synthetic",
        {},
    )
    direct = deterministic_asset_shock(weights, scenario)
    estimate = volatility_shock_estimate(prices.pct_change().dropna(), weights, 1.5)
    assert direct.portfolio_impact == -0.14
    assert estimate.methodology == "covariance_volatility_estimate"
    assert estimate.warnings
    assert worst_loss_contributor(direct) == "B"


def test_correlation_and_beta_shocks_disclose_approximation() -> None:
    prices, weights = _portfolio()
    returns = prices.pct_change().dropna()
    corr = correlation_shock_estimate(returns, weights, 0.95)
    beta = benchmark_beta_shock(returns, returns["A"], weights, -0.10)
    assert corr.methodology == "covariance_volatility_estimate"
    assert beta.methodology == "beta_approximation"
    assert corr.warnings
    assert beta.warnings
    with pytest.raises(ValueError, match="target_correlation"):
        correlation_shock_estimate(returns, weights, 1.5)


def test_rank_stress_results_orders_worst_first() -> None:
    prices, weights = _portfolio()
    mild = deterministic_asset_shock(
        weights,
        DeterministicShockScenario(
            "mild", "deterministic_asset_shock", {"A": -0.05}, "", {}
        ),
    )
    severe = deterministic_asset_shock(
        weights,
        DeterministicShockScenario(
            "severe", "deterministic_asset_shock", {"A": -0.20, "B": -0.20}, "", {}
        ),
    )
    ranking = rank_stress_results([mild, severe])
    assert ranking.iloc[0]["scenario"] == "severe"
    assert ranking.iloc[0]["rank"] == 1
