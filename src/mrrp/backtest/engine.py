"""No-look-ahead backtest engine with explicit decision/execution timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mrrp.backtest.costs import transaction_cost, turnover
from mrrp.backtest.metrics import BacktestMetrics, compute_backtest_metrics
from mrrp.backtest.rules import (
    RuleConfig,
    compute_signal_scalar,
    target_weights_from_scalar,
    validate_weights,
)
from mrrp.portfolio.returns import compute_asset_returns

RebalanceFrequency = Literal["W", "M", "Q"]


@dataclass(frozen=True)
class BacktestConfig:
    """Engine configuration for leakage-safe portfolio backtests."""

    rule: RuleConfig
    rebalance_frequency: RebalanceFrequency = "M"
    transaction_cost_bps: float = 10.0
    signal_shift: int = 1
    periods_per_year: int = 252


@dataclass(frozen=True)
class BacktestResult:
    """Complete deterministic backtest output."""

    config_name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    exposure: pd.Series
    signal_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex
    execution_dates: pd.DatetimeIndex
    metrics: BacktestMetrics
    warnings: tuple[str, ...]


def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_ticker: str,
    config: BacktestConfig,
    *,
    config_name: str = "strategy",
) -> BacktestResult:
    """Run a no-look-ahead backtest.

    Timing contract
    ---------------
    - signal date ``t``: features available at close of ``t``
    - decision date ``t``: target weights formed from shifted signal
    - execution date ``t+signal_shift``: weights become active (default next day)
    - return realization: portfolio return on the execution/holding dates uses
      open-to-close/close-to-close returns realized after execution
    """
    if config.signal_shift < 1:
        raise ValueError("signal_shift must be >= 1 to forbid same-close execution")
    validate_weights(config.rule.base_weights)
    validate_weights(config.rule.defensive_weights)

    asset_tickers = list(
        config.rule.base_weights.index.union(config.rule.defensive_weights.index)
    )
    required = list(dict.fromkeys([*asset_tickers, benchmark_ticker]))
    missing = [ticker for ticker in required if ticker not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required tickers: {missing}")

    asset_returns = compute_asset_returns(prices.loc[:, asset_tickers], method="simple")
    benchmark_returns = compute_asset_returns(
        prices.loc[:, [benchmark_ticker]], method="simple"
    )[benchmark_ticker]

    scalar = compute_signal_scalar(features, config.rule)
    # Shift signals so decisions use only information available before execution.
    shifted_scalar = scalar.shift(config.signal_shift)
    common_index = asset_returns.index.intersection(shifted_scalar.dropna().index)
    if common_index.empty:
        raise ValueError("No overlapping dates after signal shift")

    rebalance_dates = _rebalance_dates(common_index, config.rebalance_frequency)
    weight_rows: list[pd.Series] = []
    turnover_rows: dict[pd.Timestamp, float] = {}
    cost_rows: dict[pd.Timestamp, float] = {}
    current_weights = validate_weights(config.rule.base_weights)
    previous_weights = current_weights.copy()
    warnings: list[str] = []

    for date in common_index:
        if date in rebalance_dates:
            target = target_weights_from_scalar(
                float(shifted_scalar.loc[date]), config.rule
            )
            turnover_rows[date] = turnover(previous_weights, target)
            cost_rows[date] = transaction_cost(
                previous_weights,
                target,
                config.transaction_cost_bps,
            )
            current_weights = target
            previous_weights = target
        weight_rows.append(current_weights.rename(date))

    weights = pd.DataFrame(weight_rows)
    weights.index = pd.DatetimeIndex(common_index)
    portfolio_returns = pd.Series(
        index=common_index, dtype=float, name="strategy_return"
    )
    for date in common_index:
        day_ret = asset_returns.loc[date].reindex(weights.columns).fillna(0.0)
        gross = float(day_ret.dot(weights.loc[date]))
        cost = cost_rows.get(date, 0.0)
        portfolio_returns.loc[date] = gross - cost

    exposure = weights.sum(axis=1).rename("gross_exposure")
    turnover_series = (
        pd.Series(turnover_rows, dtype=float).reindex(common_index).fillna(0.0)
    )
    cost_series = pd.Series(cost_rows, dtype=float).reindex(common_index).fillna(0.0)
    metrics = compute_backtest_metrics(
        portfolio_returns,
        benchmark_returns.reindex(common_index),
        turnover_series=turnover_series,
        cost_series=cost_series,
        periods_per_year=config.periods_per_year,
    )

    signal_dates = features.index.intersection(common_index)
    decision_dates = pd.DatetimeIndex(
        sorted(rebalance_dates.intersection(common_index))
    )
    execution_dates = decision_dates  # execution occurs on decision date after shift
    if config.signal_shift == 1:
        warnings.append(
            "Default signal_shift=1 means weights decided from prior close execute "
            "at the next close/return period; not same-close execution."
        )
    warnings.append(
        "Backtest results are historical simulations for research only and are "
        "not guarantees of future performance or alpha."
    )

    return BacktestResult(
        config_name=config_name,
        strategy_returns=portfolio_returns,
        benchmark_returns=benchmark_returns.reindex(common_index),
        weights=weights,
        turnover=turnover_series.rename("turnover"),
        costs=cost_series.rename("transaction_cost"),
        exposure=exposure,
        signal_dates=pd.DatetimeIndex(signal_dates),
        decision_dates=decision_dates,
        execution_dates=execution_dates,
        metrics=metrics,
        warnings=tuple(warnings),
    )


def _rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: RebalanceFrequency,
) -> pd.DatetimeIndex:
    if frequency == "W":
        rule = "W-FRI"
    elif frequency == "M":
        rule = "ME"
    elif frequency == "Q":
        rule = "QE"
    else:
        raise ValueError("rebalance_frequency must be one of 'W', 'M', 'Q'")
    period_ends = pd.Series(1, index=index).resample(rule).last().dropna().index
    # Map period-end labels to the last available session on/before each label.
    available = []
    for stamp in period_ends:
        candidates = index[index <= stamp]
        if len(candidates):
            available.append(candidates[-1])
    return pd.DatetimeIndex(sorted(set(available)))
