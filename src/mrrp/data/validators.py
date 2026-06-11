"""Validation helpers for price data."""

from __future__ import annotations

import pandas as pd


def validate_price_frame(
    prices: pd.DataFrame,
    max_missing_fraction: float = 0.25,
    min_observations: int = 252,
) -> None:
    """Validate the structure and completeness of a price DataFrame."""
    _validate_dataframe(prices)

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price data index must be a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price data index must be monotonic increasing")
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates")
    if prices.columns.has_duplicates:
        raise ValueError("Price data contains duplicate columns")

    all_null_tickers = prices.columns[prices.isna().all()].tolist()
    if all_null_tickers:
        raise ValueError(f"Ticker columns contain only null prices: {all_null_tickers}")

    try:
        non_positive = prices.le(0)
    except TypeError as exc:
        raise ValueError("Price data must contain numeric values") from exc

    if non_positive.any().any():
        invalid_tickers = non_positive.any()[lambda values: values].index.tolist()
        raise ValueError(f"Prices must be positive for tickers: {invalid_tickers}")

    validate_min_history(prices, min_observations=min_observations)

    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    missing_fractions = prices.isna().mean()
    excessive_missing = missing_fractions[missing_fractions > max_missing_fraction]
    if not excessive_missing.empty:
        tickers = excessive_missing.index.tolist()
        raise ValueError(
            f"Tickers exceed maximum missing-data fraction "
            f"({max_missing_fraction:.0%}): {tickers}"
        )


def report_missing_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per-ticker missing-data statistics."""
    _validate_dataframe(prices)

    rows = []
    for ticker in prices.columns:
        series = prices[ticker]
        observation_count = int(series.count())
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "ticker": ticker,
                "missing_count": missing_count,
                "missing_fraction": missing_count / len(prices) if len(prices) else 0.0,
                "first_valid_date": series.first_valid_index(),
                "last_valid_date": series.last_valid_index(),
                "observation_count": observation_count,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "missing_count",
            "missing_fraction",
            "first_valid_date",
            "last_valid_date",
            "observation_count",
        ],
    )


def validate_min_history(
    prices: pd.DataFrame,
    min_observations: int = 252,
) -> None:
    """Validate that every ticker has the required number of observations."""
    _validate_dataframe(prices)

    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")

    observation_counts = prices.count()
    insufficient = observation_counts[observation_counts < min_observations]
    if not insufficient.empty:
        counts = insufficient.to_dict()
        raise ValueError(
            f"Tickers have fewer than {min_observations} observations: {counts}"
        )


def _validate_dataframe(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Price data must be a pandas DataFrame")
