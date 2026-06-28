"""Risk-adjusted performance metrics for periodic return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.risk.drawdown import max_drawdown
from mrrp.risk.volatility import annualized_return, annualized_volatility


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Sharpe ratio using an annualized risk-free rate."""
    periodic_risk_free_rate = _periodic_risk_free_rate(risk_free_rate, periods_per_year)
    excess_returns = returns - periodic_risk_free_rate
    annualized_excess_return = annualized_return(excess_returns, periods_per_year)
    volatility = annualized_volatility(excess_returns, periods_per_year)

    if _is_zero_or_nan(volatility):
        return float(np.nan)
    return annualized_excess_return / volatility


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Sortino ratio using negative excess returns as downside."""
    periodic_risk_free_rate = _periodic_risk_free_rate(risk_free_rate, periods_per_year)
    excess_returns = returns - periodic_risk_free_rate
    annualized_excess_return = annualized_return(excess_returns, periods_per_year)
    negative_excess_returns = excess_returns.dropna()[lambda values: values < 0]

    if negative_excess_returns.empty:
        return float(np.nan)

    downside_deviation = float(
        np.sqrt(negative_excess_returns.pow(2).mean()) * np.sqrt(periods_per_year)
    )
    if _is_zero_or_nan(downside_deviation):
        return float(np.nan)
    return annualized_excess_return / downside_deviation


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized return divided by absolute maximum drawdown."""
    annual_return = annualized_return(returns, periods_per_year)
    maximum_drawdown = max_drawdown(returns)

    if _is_zero_or_nan(maximum_drawdown):
        return float(np.nan)
    return annual_return / abs(maximum_drawdown)


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility of index-aligned active returns."""
    active_returns = _aligned_active_returns(portfolio_returns, benchmark_returns)
    return annualized_volatility(active_returns, periods_per_year)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized active return divided by tracking error."""
    active_returns = _aligned_active_returns(portfolio_returns, benchmark_returns)
    annualized_active_return = annualized_return(active_returns, periods_per_year)
    active_tracking_error = annualized_volatility(active_returns, periods_per_year)

    if _is_zero_or_nan(active_tracking_error):
        return float(np.nan)
    return annualized_active_return / active_tracking_error


def _periodic_risk_free_rate(
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    try:
        finite_rate = np.isfinite(risk_free_rate)
    except TypeError as exc:
        raise ValueError("risk_free_rate must be numeric") from exc

    if not finite_rate or risk_free_rate < -1:
        raise ValueError("risk_free_rate must be finite and at least -1")

    return float((1 + risk_free_rate) ** (1 / periods_per_year) - 1)


def _aligned_active_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    if not isinstance(portfolio_returns, pd.Series) or not isinstance(
        benchmark_returns, pd.Series
    ):
        raise ValueError("Portfolio and benchmark returns must be pandas Series")

    aligned = pd.concat(
        {
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        },
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("Portfolio and benchmark have no overlapping returns")

    return aligned["portfolio"] - aligned["benchmark"]


def _is_zero_or_nan(value: float) -> bool:
    return bool(
        np.isnan(value) or np.isclose(value, 0.0, rtol=0.0, atol=np.finfo(float).eps)
    )
