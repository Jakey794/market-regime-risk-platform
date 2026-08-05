"""Backtest performance and diagnostic metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mrrp.risk.drawdown import max_drawdown
from mrrp.risk.performance import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from mrrp.risk.volatility import annualized_return, annualized_volatility


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary metrics for a completed backtest."""

    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    tracking_error: float
    turnover: float
    transaction_cost_drag: float
    worst_month: float
    rolling_12m_return_last: float
    fraction_outperforming_months: float
    underperformance_duration_days: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "tracking_error": self.tracking_error,
            "turnover": self.turnover,
            "transaction_cost_drag": self.transaction_cost_drag,
            "worst_month": self.worst_month,
            "rolling_12m_return_last": self.rolling_12m_return_last,
            "fraction_outperforming_months": self.fraction_outperforming_months,
            "underperformance_duration_days": self.underperformance_duration_days,
        }


def compute_backtest_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    turnover_series: pd.Series,
    cost_series: pd.Series,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute recruiter-facing backtest diagnostics from realized returns."""
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    strategy = aligned["strategy"]
    benchmark = aligned["benchmark"]

    monthly_strategy = (1.0 + strategy).resample("ME").prod() - 1.0
    monthly_benchmark = (1.0 + benchmark).resample("ME").prod() - 1.0
    monthly_aligned = pd.concat(
        [monthly_strategy.rename("strategy"), monthly_benchmark.rename("benchmark")],
        axis=1,
    ).dropna()
    outperform = monthly_aligned["strategy"] > monthly_aligned["benchmark"]
    fraction = float(outperform.mean()) if len(monthly_aligned) else float("nan")
    worst_month = (
        float(monthly_strategy.min()) if len(monthly_strategy) else float("nan")
    )

    rolling_12m = (
        (1.0 + strategy)
        .rolling(periods_per_year)
        .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    )
    active = strategy - benchmark
    under_duration = _max_underperformance_duration(active)

    return BacktestMetrics(
        cagr=float(annualized_return(strategy, periods_per_year)),
        annualized_volatility=float(annualized_volatility(strategy, periods_per_year)),
        sharpe=float(sharpe_ratio(strategy, periods_per_year=periods_per_year)),
        sortino=float(sortino_ratio(strategy, periods_per_year=periods_per_year)),
        max_drawdown=float(max_drawdown(strategy)),
        calmar=float(calmar_ratio(strategy, periods_per_year=periods_per_year)),
        tracking_error=float(
            tracking_error(strategy, benchmark, periods_per_year=periods_per_year)
        ),
        turnover=float(turnover_series.sum()),
        transaction_cost_drag=float(cost_series.sum()),
        worst_month=worst_month,
        rolling_12m_return_last=float(rolling_12m.dropna().iloc[-1])
        if not rolling_12m.dropna().empty
        else float("nan"),
        fraction_outperforming_months=fraction,
        underperformance_duration_days=under_duration,
    )


def _max_underperformance_duration(active_returns: pd.Series) -> int:
    wealth = (1.0 + active_returns.fillna(0.0)).cumprod()
    relative = wealth / wealth.cummax() - 1.0
    duration = 0
    current = 0
    for value in relative:
        if value < 0:
            current += 1
            duration = max(duration, current)
        else:
            current = 0
    return int(duration)
