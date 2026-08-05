from __future__ import annotations

import numpy as np

from mrrp.backtest.engine import run_backtest
from test_backtest_engine import backtest_inputs


def test_future_price_feature_and_regime_mutations_do_not_change_past() -> None:
    prices, features, config = backtest_inputs()
    baseline = run_backtest(prices, features, "SPY", config)
    cutoff = prices.index[220]

    mutated_prices = prices.copy()
    mutated_prices.loc[cutoff:, ["A", "B", "SPY"]] *= 3.0
    mutated_features = features.copy()
    mutated_features.loc[cutoff:, "portfolio_vol_63d"] = 99.0
    mutated_features.loc[cutoff:, "regime"] = 999
    changed = run_backtest(mutated_prices, mutated_features, "SPY", config)

    past = baseline.weights.index < cutoff
    np.testing.assert_allclose(baseline.weights.loc[past], changed.weights.loc[past])
    np.testing.assert_allclose(
        baseline.strategy_returns.loc[past],
        changed.strategy_returns.loc[past],
    )
