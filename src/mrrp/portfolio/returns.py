"""Portfolio-level return calculations built on asset return metrics."""

from __future__ import annotations

import pandas as pd

from mrrp.risk.returns import cumulative_returns, log_returns, simple_returns


def compute_asset_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """Compute asset returns and remove the initial undefined-return row."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")

    if method == "simple":
        asset_returns = simple_returns(prices)
    elif method == "log":
        asset_returns = log_returns(prices)
    else:
        raise ValueError("method must be either 'simple' or 'log'")

    return asset_returns.iloc[1:]


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute weighted portfolio returns without normalizing weights."""
    aligned_returns, aligned_weights = align_returns_and_weights(returns, weights)
    return aligned_returns.dot(aligned_weights)


def compute_cumulative_returns(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Compound periodic returns into a cumulative return path."""
    return cumulative_returns(returns)


def align_returns_and_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select and reorder return columns to match the weight index."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
    if not isinstance(weights, pd.Series):
        raise ValueError("Weights must be a pandas Series")
    if weights.empty:
        raise ValueError("Weights must not be empty")
    if weights.index.has_duplicates:
        raise ValueError("Weight tickers must be unique")
    if returns.columns.has_duplicates:
        raise ValueError("Return columns must be unique")

    missing_tickers = [
        str(ticker) for ticker in weights.index if ticker not in returns.columns
    ]
    if missing_tickers:
        raise ValueError(
            f"Weights reference missing return columns: {missing_tickers}"
        )

    ordered_tickers = weights.index.tolist()
    return returns.loc[:, ordered_tickers], weights.reindex(ordered_tickers)
