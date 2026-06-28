"""Drawdown metrics calculated from periodic return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.risk.returns import wealth_index


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate the drawdown path from periodic fractional returns.

    The input is a return series, not a price or wealth series. Missing return
    observations remain missing in the output and do not reset compounding.
    """
    _validate_returns(returns)

    wealth = wealth_index(returns, start_value=1.0)
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Return the worst drawdown produced by periodic fractional returns."""
    return float(drawdown_series(returns).min())


def current_drawdown(returns: pd.Series) -> float:
    """Return the final drawdown produced by periodic fractional returns."""
    return float(drawdown_series(returns).iloc[-1])


def drawdown_duration(returns: pd.Series) -> int:
    """Return the number of valid periods since the last all-time high."""
    drawdowns = drawdown_series(returns).dropna()
    at_high = np.isclose(drawdowns.to_numpy(), 0.0, rtol=0.0, atol=1e-12)

    if at_high[-1]:
        return 0

    high_positions = np.flatnonzero(at_high)
    last_high_position = int(high_positions[-1]) if high_positions.size else -1
    return len(drawdowns) - last_high_position - 1


def rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Calculate max drawdown independently within each rolling window."""
    _validate_returns(returns)
    if window <= 1:
        raise ValueError("window must be greater than 1")

    return returns.rolling(window=window).apply(_window_max_drawdown, raw=True)


def _window_max_drawdown(window_returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + window_returns)
    running_peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return float(np.min(wealth / running_peak - 1))


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")
    if returns.empty or returns.dropna().empty:
        raise ValueError("Returns must contain at least one valid observation")

    valid_returns = returns.dropna()
    try:
        all_finite = np.isfinite(valid_returns.to_numpy()).all()
    except TypeError as exc:
        raise ValueError("Returns must contain numeric values") from exc

    if not all_finite:
        raise ValueError("Returns must contain finite values")
    if valid_returns.lt(-1).any():
        raise ValueError("Returns must not be less than -1")
