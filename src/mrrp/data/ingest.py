"""Price ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Yahoo Finance returns adjusted OHLC data when ``auto_adjust=True``. In that
    mode the ``Close`` field is the adjusted close price series.
    """
    if not tickers:
        raise ValueError("At least one ticker is required")

    normalized_tickers = [ticker.strip().upper() for ticker in tickers]
    if any(not ticker for ticker in normalized_tickers):
        raise ValueError("Ticker symbols must be non-empty strings")

    raw_prices = yf.download(
        normalized_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    close_prices = _extract_close_prices(raw_prices, normalized_tickers)

    return _validate_prices(close_prices)


def save_prices(
    prices: pd.DataFrame,
    raw_path: str | Path,
    processed_path: str | Path,
) -> None:
    """Save prices to raw and processed parquet paths."""
    validated_prices = _validate_prices(prices)

    raw_output = Path(raw_path)
    processed_output = Path(processed_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    processed_output.parent.mkdir(parents=True, exist_ok=True)

    validated_prices.to_parquet(raw_output)
    validated_prices.to_parquet(processed_output)


def _extract_close_prices(raw_prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw_prices.empty:
        raise ValueError("No price data returned from yfinance")

    if isinstance(raw_prices.columns, pd.MultiIndex):
        if "Close" in raw_prices.columns.get_level_values(0):
            close_prices = raw_prices["Close"]
        elif "Close" in raw_prices.columns.get_level_values(-1):
            close_prices = raw_prices.xs("Close", level=-1, axis=1)
        else:
            raise ValueError("Downloaded data does not contain Close prices")
    elif "Close" in raw_prices.columns:
        close_prices = raw_prices[["Close"]]
        if len(tickers) == 1:
            close_prices = close_prices.rename(columns={"Close": tickers[0]})
    else:
        raise ValueError("Downloaded data does not contain Close prices")

    if isinstance(close_prices, pd.Series):
        ticker = tickers[0] if len(tickers) == 1 else close_prices.name
        close_prices = close_prices.to_frame(name=ticker)

    missing_tickers = [
        ticker for ticker in tickers if ticker not in close_prices.columns
    ]
    if missing_tickers:
        raise ValueError(f"Downloaded data missing tickers: {missing_tickers}")

    return close_prices.loc[:, tickers]


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates")

    prices = prices.sort_index()

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    all_null_tickers = numeric_prices.columns[numeric_prices.isna().all()].tolist()
    if all_null_tickers:
        raise ValueError(f"Ticker columns contain only null prices: {all_null_tickers}")

    return numeric_prices
