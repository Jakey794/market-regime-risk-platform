from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.risk.scenarios import DeterministicShockScenario, HistoricalWindowScenario
from mrrp.risk.stress import (
    deterministic_asset_shock,
    historical_window_stress,
    volatility_shock_estimate,
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
