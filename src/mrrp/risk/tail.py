"""Historical tail-risk metrics expressed as returns.

VaR and CVaR are returned in return space, not as positive loss magnitudes. For
example, a 95% VaR of ``-0.04`` represents a 5th-percentile return of -4%.
Empty or all-missing input series return ``np.nan`` for every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Return historical VaR as the lower-tail return quantile."""
    _validate_confidence(confidence)
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    return float(valid_returns.quantile(1 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Return mean tail return at or below the historical VaR threshold."""
    _validate_confidence(confidence)
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    var_threshold = historical_var(valid_returns, confidence)
    return float(valid_returns[valid_returns <= var_threshold].mean())


def worst_return(returns: pd.Series) -> float:
    """Return the worst observed periodic return."""
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    return float(valid_returns.min())


def worst_rolling_return(returns: pd.Series, window: int) -> float:
    """Return the worst compounded return across complete rolling windows."""
    _validate_window(window)
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    rolling_growth = (1 + returns).rolling(window).apply(np.prod, raw=True)
    return float((rolling_growth - 1).min())


def skewness(returns: pd.Series) -> float:
    """Return pandas' unbiased sample skewness estimate."""
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    return float(valid_returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Return pandas' unbiased Fisher sample kurtosis estimate."""
    valid_returns = _valid_returns(returns)
    if valid_returns.empty:
        return float(np.nan)

    return float(valid_returns.kurt())


def _valid_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")

    valid_returns = returns.dropna()
    if valid_returns.empty:
        return valid_returns

    try:
        all_finite = np.isfinite(valid_returns.to_numpy()).all()
    except TypeError as exc:
        raise ValueError("Returns must contain numeric values") from exc

    if not all_finite:
        raise ValueError("Returns must contain finite values")
    return valid_returns


def _validate_confidence(confidence: float) -> None:
    try:
        valid_confidence = np.isfinite(confidence) and 0 < confidence < 1
    except TypeError as exc:
        raise ValueError("confidence must be numeric") from exc

    if not valid_confidence:
        raise ValueError("confidence must be between 0 and 1")


def _validate_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
