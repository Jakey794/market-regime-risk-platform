"""Return calculations for assets and portfolios."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PriceData = pd.Series | pd.DataFrame
WEIGHT_SUM_TOLERANCE = 1e-8


def simple_returns(prices: PriceData) -> PriceData:
    """Calculate fractional simple returns while preserving the input shape.

    The first row is retained as ``NaN`` because it has no preceding price.
    Missing prices are not filled, so returns that need a missing observation
    are also ``NaN``.
    """
    _validate_prices(prices)
    return prices / prices.shift(1) - 1


def log_returns(prices: PriceData) -> PriceData:
    """Calculate log returns while preserving the input shape.

    The first row is retained as ``NaN`` because it has no preceding price.
    Missing prices are not filled.
    """
    _validate_prices(prices)
    return np.log(prices / prices.shift(1))


def validate_weights(
    weights: dict[str, float],
    columns: Iterable[str],
) -> pd.Series:
    """Validate and align portfolio weights to a return column sequence."""
    column_list = list(columns)

    if not column_list:
        raise ValueError("Return columns must not be empty")
    if len(column_list) != len(set(column_list)):
        raise ValueError("Return columns must be unique")

    column_set = set(column_list)
    weight_set = set(weights)

    unknown_tickers = sorted(weight_set - column_set)
    if unknown_tickers:
        raise ValueError(f"Weights contain unknown tickers: {unknown_tickers}")

    missing_tickers = sorted(column_set - weight_set)
    if missing_tickers:
        raise ValueError(f"Weights are missing tickers: {missing_tickers}")

    try:
        aligned_weights = pd.Series(weights, dtype=float).reindex(column_list)
    except (TypeError, ValueError) as exc:
        raise ValueError("Weights must be numeric") from exc

    if aligned_weights.isna().any():
        raise ValueError("Weights must not contain NaN values")
    if not np.isfinite(aligned_weights.to_numpy()).all():
        raise ValueError("Weights must be finite")
    if not np.isclose(
        aligned_weights.sum(),
        1.0,
        rtol=0.0,
        atol=WEIGHT_SUM_TOLERANCE,
    ):
        raise ValueError("Weights must sum to 1")

    return aligned_weights


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Calculate portfolio returns from column-aligned asset weights."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")

    aligned_weights = validate_weights(weights, returns.columns)
    return returns.dot(aligned_weights)


def cumulative_returns(returns: PriceData) -> PriceData:
    """Compound periodic fractional returns into cumulative returns."""
    _validate_series_or_frame(returns, name="Returns")
    return (1 + returns).cumprod() - 1


def wealth_index(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Build a wealth index from periodic fractional returns."""
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")
    if not np.isfinite(start_value) or start_value <= 0:
        raise ValueError("start_value must be positive and finite")

    return start_value * (1 + returns).cumprod()


def _validate_prices(prices: PriceData) -> None:
    _validate_series_or_frame(prices, name="Prices")

    try:
        non_positive = prices.le(0)
    except TypeError as exc:
        raise ValueError("Prices must contain numeric values") from exc

    if non_positive.to_numpy().any():
        raise ValueError("Prices must be positive")

    try:
        finite_or_missing = np.isfinite(prices) | prices.isna()
    except TypeError as exc:
        raise ValueError("Prices must contain numeric values") from exc

    if not finite_or_missing.to_numpy().all():
        raise ValueError("Prices must be finite or NaN")


def _validate_series_or_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.Series | pd.DataFrame):
        raise ValueError(f"{name} must be a pandas Series or DataFrame")
