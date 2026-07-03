"""Selection-aware data loading for dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from mrrp.data.cache import load_parquet
from mrrp.portfolio import (
    PortfolioConfig,
    compute_asset_returns,
    compute_portfolio_returns,
    load_portfolio_config,
    validate_weights,
)


DateSelection = date | pd.Timestamp
MINIMUM_COMPLETE_RETURNS = 63


class InsufficientObservationsError(ValueError):
    """Raised when a selected range cannot support dashboard risk metrics."""


@dataclass(frozen=True)
class DashboardOptions:
    """Available values and date bounds for dashboard controls."""

    portfolios: tuple[str, ...]
    benchmarks: tuple[str, ...]
    minimum_date: pd.Timestamp
    maximum_date: pd.Timestamp


@dataclass(frozen=True)
class DashboardData:
    """Validated prices and returns for the current dashboard selection."""

    prices: pd.DataFrame
    portfolio_config: PortfolioConfig
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    complete_observations: int


@st.cache_data(show_spinner=False)
def load_dashboard_options(
    prices_path: str,
    portfolio_path: str,
) -> DashboardOptions:
    """Load cached sidebar options from processed data and portfolio config."""
    prices, portfolio = load_dashboard_data(prices_path, portfolio_path)
    return DashboardOptions(
        portfolios=(portfolio.name,),
        benchmarks=tuple(str(column) for column in prices.columns),
        minimum_date=prices.index.min(),
        maximum_date=prices.index.max(),
    )


@st.cache_data(show_spinner=False)
def load_dashboard_dataset(
    prices_path: str,
    portfolio_path: str,
    *,
    start_date: DateSelection,
    end_date: DateSelection,
    benchmark: str,
    minimum_complete_returns: int = MINIMUM_COMPLETE_RETURNS,
) -> DashboardData:
    """Load cached selected prices and derive returns through portfolio APIs."""
    if (
        isinstance(minimum_complete_returns, bool)
        or not isinstance(minimum_complete_returns, int)
        or minimum_complete_returns < 1
    ):
        raise ValueError("minimum_complete_returns must be a positive integer")

    prices, portfolio = load_dashboard_data(
        prices_path,
        portfolio_path,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
    )
    asset_prices = prices.loc[:, portfolio.holdings.index]
    asset_returns = compute_asset_returns(asset_prices, method="simple")
    portfolio_returns = compute_portfolio_returns(
        asset_returns,
        portfolio.holdings,
    )
    benchmark_returns = compute_asset_returns(
        prices.loc[:, [portfolio.benchmark]],
        method="simple",
    )[portfolio.benchmark]
    complete_observations = len(
        pd.concat(
            [asset_returns, benchmark_returns.rename("__benchmark__")],
            axis=1,
        ).dropna()
    )
    if complete_observations < minimum_complete_returns:
        raise InsufficientObservationsError(
            "Selected date range must provide at least "
            f"{minimum_complete_returns} complete return observations; "
            f"found {complete_observations}."
        )

    return DashboardData(
        prices=prices,
        portfolio_config=portfolio,
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        complete_observations=complete_observations,
    )


def load_dashboard_data(
    prices_path: str | Path,
    portfolio_path: str | Path,
    *,
    start_date: DateSelection | None = None,
    end_date: DateSelection | None = None,
    benchmark: str | None = None,
) -> tuple[pd.DataFrame, PortfolioConfig]:
    """Load and validate selected prices and portfolio configuration.

    This adapter performs no return or risk calculations. Missing observations
    are preserved for the existing portfolio and risk APIs to handle.
    """
    prices = load_parquet(prices_path)
    portfolio = load_portfolio_config(portfolio_path)
    _validate_prices(prices)
    validate_weights(portfolio.holdings, allow_short=portfolio.allow_short)
    if portfolio.holdings.index.has_duplicates:
        raise ValueError("Portfolio tickers must be unique")

    if benchmark is not None:
        portfolio = replace(portfolio, benchmark=_normalize_benchmark(benchmark))

    start = _normalize_date(start_date, field="start_date")
    end = _normalize_date(end_date, field="end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date")

    selected_prices = prices.loc[start:end].copy()
    if selected_prices.empty:
        raise ValueError("Selected date range contains no price observations")

    required_tickers = list(portfolio.holdings.index)
    if portfolio.benchmark not in required_tickers:
        required_tickers.append(portfolio.benchmark)
    missing_tickers = [
        ticker for ticker in required_tickers if ticker not in selected_prices.columns
    ]
    if missing_tickers:
        raise ValueError(f"Prices are missing required tickers: {missing_tickers}")

    return selected_prices, portfolio


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("Processed prices must not be empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Processed prices must have a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("Processed price dates must be unique")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Processed prices must be ordered from oldest to newest")
    if prices.columns.has_duplicates:
        raise ValueError("Processed price columns must be unique")


def _normalize_benchmark(benchmark: str) -> str:
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("benchmark must be a non-empty string")
    return benchmark.strip().upper()


def _normalize_date(
    value: DateSelection | None,
    *,
    field: str,
) -> pd.Timestamp | None:
    if value is None:
        return None
    if not isinstance(value, (date, pd.Timestamp)):
        raise ValueError(f"{field} must be a date or pandas Timestamp")
    return pd.Timestamp(value)
