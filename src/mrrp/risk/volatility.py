"""Annualized return and volatility metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate the compound annualized return from periodic returns.

    Missing observations are excluded from both compounding and the period
    count. Returns below -100% are rejected because they imply negative wealth.
    """
    _validate_periods_per_year(periods_per_year)
    valid_returns = _drop_missing_returns(returns)

    if valid_returns.lt(-1).any():
        raise ValueError("Returns must not be less than -1")

    ending_wealth = (1 + valid_returns).prod()
    n_periods = len(valid_returns)
    return float(ending_wealth ** (periods_per_year / n_periods) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility using sample standard deviation."""
    _validate_periods_per_year(periods_per_year)
    valid_returns = _drop_missing_returns(returns)

    return float(valid_returns.std(ddof=1) * np.sqrt(periods_per_year))


def rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate annualized rolling sample volatility.

    Missing observations are removed before calculating the rolling statistic,
    then restored as missing values on the original index.
    """
    _validate_periods_per_year(periods_per_year)
    if window <= 1:
        raise ValueError("window must be greater than 1")

    valid_returns = _drop_missing_returns(returns)
    volatility = valid_returns.rolling(window=window).std(ddof=1)
    volatility *= np.sqrt(periods_per_year)
    return _restore_original_index(volatility, returns)


def ewma_volatility(
    returns: pd.Series,
    span: int = 21,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate annualized exponentially weighted sample volatility.

    Missing observations are removed before calculating the EWMA statistic,
    then restored as missing values on the original index.
    """
    _validate_periods_per_year(periods_per_year)
    if span <= 0:
        raise ValueError("span must be positive")

    valid_returns = _drop_missing_returns(returns)
    volatility = valid_returns.ewm(span=span, adjust=False).std()
    volatility *= np.sqrt(periods_per_year)
    return _restore_original_index(volatility, returns)


def _drop_missing_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")

    valid_returns = returns.dropna()
    if valid_returns.empty:
        raise ValueError("Returns must contain at least one valid observation")

    try:
        all_finite = np.isfinite(valid_returns.to_numpy()).all()
    except TypeError as exc:
        raise ValueError("Returns must contain numeric values") from exc

    if not all_finite:
        raise ValueError("Returns must contain finite values")

    return valid_returns


def _validate_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")


def _restore_original_index(
    values: pd.Series,
    original: pd.Series,
) -> pd.Series:
    result = pd.Series(np.nan, index=original.index, name=original.name)
    result.iloc[np.flatnonzero(original.notna().to_numpy())] = values.to_numpy()
    return result
