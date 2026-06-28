"""Integrated portfolio risk summary built from the metrics engine."""

from __future__ import annotations

import pandas as pd

from mrrp.risk.beta import beta
from mrrp.risk.drawdown import current_drawdown, max_drawdown
from mrrp.risk.performance import (
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from mrrp.risk.returns import portfolio_returns, simple_returns
from mrrp.risk.tail import historical_cvar, historical_var, worst_return
from mrrp.risk.tail import worst_rolling_return
from mrrp.risk.volatility import annualized_return, annualized_volatility


def portfolio_risk_summary(
    prices: pd.DataFrame,
    weights: dict[str, float],
    benchmark: str,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Return a numeric portfolio risk summary from asset and benchmark prices."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark column is missing: {benchmark}")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must be unique")

    returns = simple_returns(prices)
    asset_columns = [column for column in returns.columns if column != benchmark]
    weighted_returns = portfolio_returns(returns.loc[:, asset_columns], weights)
    benchmark_returns = returns[benchmark]

    rows = [
        (
            "Annualized return",
            annualized_return(weighted_returns, periods_per_year),
            "Compound annualized portfolio return.",
        ),
        (
            "Annualized volatility",
            annualized_volatility(weighted_returns, periods_per_year),
            "Annualized sample standard deviation of portfolio returns.",
        ),
        (
            "Sharpe ratio",
            sharpe_ratio(weighted_returns, periods_per_year=periods_per_year),
            "Annualized excess return per unit of total volatility.",
        ),
        (
            "Sortino ratio",
            sortino_ratio(weighted_returns, periods_per_year=periods_per_year),
            "Annualized excess return per unit of downside deviation.",
        ),
        (
            "Max drawdown",
            max_drawdown(weighted_returns),
            "Worst peak-to-trough portfolio decline.",
        ),
        (
            "Current drawdown",
            current_drawdown(weighted_returns),
            "Portfolio decline from its latest all-time high.",
        ),
        (
            "95% VaR",
            historical_var(weighted_returns, confidence=0.95),
            "Historical 5th-percentile return, expressed as a return.",
        ),
        (
            "99% VaR",
            historical_var(weighted_returns, confidence=0.99),
            "Historical 1st-percentile return, expressed as a return.",
        ),
        (
            "95% CVaR",
            historical_cvar(weighted_returns, confidence=0.95),
            "Mean return at or below the historical 95% VaR threshold.",
        ),
        (
            "99% CVaR",
            historical_cvar(weighted_returns, confidence=0.99),
            "Mean return at or below the historical 99% VaR threshold.",
        ),
        (
            "Beta vs benchmark",
            beta(weighted_returns, benchmark_returns),
            "Portfolio sensitivity to benchmark returns.",
        ),
        (
            "Tracking error",
            tracking_error(
                weighted_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "Annualized volatility of portfolio returns minus benchmark returns.",
        ),
        (
            "Information ratio",
            information_ratio(
                weighted_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "Annualized active return per unit of tracking error.",
        ),
        (
            "Worst day",
            worst_return(weighted_returns),
            "Worst observed single-period portfolio return.",
        ),
        (
            "Worst rolling 21D return",
            worst_rolling_return(weighted_returns, window=21),
            "Worst compounded portfolio return over 21 periods.",
        ),
    ]

    summary = pd.DataFrame(rows, columns=["metric", "value", "description"])
    summary["value"] = summary["value"].astype(float)
    return summary
