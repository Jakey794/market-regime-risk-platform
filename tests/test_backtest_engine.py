from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_transaction_costs_reduce_net_returns() -> None:
    prices, features, config = backtest_inputs()
    free = run_backtest(
        prices,
        features,
        "SPY",
        BacktestConfig(rule=config.rule, transaction_cost_bps=0.0),
    )
    costly = run_backtest(
        prices,
        features,
        "SPY",
        BacktestConfig(rule=config.rule, transaction_cost_bps=50.0),
    )
    assert costly.strategy_returns.sum() < free.strategy_returns.sum()
    assert costly.metrics.transaction_cost_drag > free.metrics.transaction_cost_drag


def test_static_strategy_matches_weighted_asset_returns() -> None:
    prices, features, _ = backtest_inputs()
    weights = pd.Series({"A": 0.7, "B": 0.3})
    config = BacktestConfig(
        rule=RuleConfig(
            name="static_benchmark",
            base_weights=weights,
            defensive_weights=weights,
        ),
        transaction_cost_bps=0.0,
        rebalance_frequency="M",
    )
    result = run_backtest(prices, features, "SPY", config)
    asset_returns = prices.loc[:, ["A", "B"]].pct_change()
    expected = (
        asset_returns.mul(weights, axis=1)
        .sum(axis=1)
        .reindex(result.strategy_returns.index)
    )
    np.testing.assert_allclose(
        result.strategy_returns.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_monthly_rebalance_dates_are_month_end_sessions() -> None:
    prices, features, _ = backtest_inputs()
    result = run_backtest(
        prices,
        features,
        "SPY",
        BacktestConfig(
            rule=RuleConfig(
                name="static_benchmark",
                base_weights=pd.Series({"A": 0.7, "B": 0.3}),
                defensive_weights=pd.Series({"A": 0.7, "B": 0.3}),
            ),
            rebalance_frequency="M",
            transaction_cost_bps=0.0,
        ),
    )
    assert len(result.decision_dates) >= 3
    for stamp in result.decision_dates:
        month_sessions = prices.index[
            (prices.index.year == stamp.year) & (prices.index.month == stamp.month)
        ]
        assert stamp == month_sessions.max()


def test_signal_shift_rejects_same_close_execution() -> None:
    prices, features, config = backtest_inputs()
    with pytest.raises(ValueError, match="signal_shift"):
        run_backtest(
            prices,
            features,
            "SPY",
            BacktestConfig(rule=config.rule, signal_shift=0),
        )


def test_metrics_include_required_fields() -> None:
    prices, features, config = backtest_inputs()
    result = run_backtest(prices, features, "SPY", config)
    metrics = result.metrics.to_dict()
    required = {
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "tracking_error",
        "turnover",
        "transaction_cost_drag",
        "worst_month",
        "fraction_outperforming_months",
    }
    assert required.issubset(metrics)
    assert result.metrics.turnover == pytest.approx(float(result.turnover.sum()))
