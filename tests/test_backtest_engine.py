from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.backtest.engine import BacktestConfig, run_backtest
from mrrp.backtest.rules import RuleConfig


def backtest_inputs() -> tuple[pd.DataFrame, pd.DataFrame, BacktestConfig]:
    index = pd.date_range("2021-01-01", periods=320, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(np.full(len(index), 1.0004)),
            "B": 100 * np.cumprod(np.full(len(index), 1.0001)),
            "SPY": 100 * np.cumprod(np.full(len(index), 1.0003)),
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "portfolio_vol_63d": np.where(np.arange(len(index)) < 160, 0.1, 0.3),
            "mean_corr_63d": 0.4,
            "portfolio_drawdown": 0.0,
            "regime": np.where(np.arange(len(index)) < 160, 0, 1),
        },
        index=index,
    )
    rule = RuleConfig(
        name="volatility_target",
        base_weights=pd.Series({"A": 0.7, "B": 0.3}),
        defensive_weights=pd.Series({"A": 0.2, "B": 0.8}),
        vol_target=0.12,
    )
    return prices, features, BacktestConfig(rule=rule)


def test_engine_outputs_aligned_returns_weights_and_costs() -> None:
    prices, features, config = backtest_inputs()
    result = run_backtest(prices, features, "SPY", config)
    assert result.weights.index.equals(result.strategy_returns.index)
    assert result.costs.index.equals(result.strategy_returns.index)
    assert result.weights.sum(axis=1).round(12).eq(1.0).all()
    assert any("not guarantees" in warning for warning in result.warnings)
