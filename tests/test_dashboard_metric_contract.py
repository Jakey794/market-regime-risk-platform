"""Contract between the risk engine's summary output and dashboard rendering.

``test_portfolio_summary.py`` already asserts ``PortfolioRiskSummary``'s full
field list. This file instead checks the narrower, previously-untested slice
that actually matters to the dashboard: the exact fields
``app/pages/1_Portfolio_Overview.py`` reads can be built and then formatted
by the dashboard's own formatting helpers without raising, plus the direct
value-correctness of those formatting helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.dashboard.formatting import format_decimal, format_percentage
from mrrp.portfolio import PortfolioConfig, build_portfolio_risk_summary


# The exact PortfolioRiskSummary fields read by app/pages/1_Portfolio_Overview.py.
PAGE_ONE_SUMMARY_FIELDS = [
    "annualized_return",
    "annualized_volatility",
    "current_drawdown",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "effective_holdings",
]


@pytest.fixture
def portfolio_config() -> PortfolioConfig:
    return PortfolioConfig(
        name="test_portfolio",
        benchmark="SPY",
        currency="CAD",
        allow_short=False,
        holdings=pd.Series({"ASSET_A": 0.60, "ASSET_B": 0.40}, name="weight"),
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=81, freq="B")
    benchmark_returns = np.resize(
        np.array([0.008, -0.006, 0.004, -0.010, 0.012, 0.002]),
        80,
    )
    asset_a_returns = 1.2 * benchmark_returns + np.resize(
        np.array([0.002, -0.001, 0.001]),
        80,
    )
    asset_b_returns = 0.5 * benchmark_returns + np.resize(
        np.array([-0.001, 0.002, -0.002, 0.001]),
        80,
    )

    def price_path(period_returns: np.ndarray) -> np.ndarray:
        return np.concatenate(([100.0], 100.0 * np.cumprod(1 + period_returns)))

    return pd.DataFrame(
        {
            "ASSET_A": price_path(asset_a_returns),
            "ASSET_B": price_path(asset_b_returns),
            "SPY": price_path(benchmark_returns),
        },
        index=index,
    )


def test_summary_has_fields_the_dashboard_reads(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    summary = build_portfolio_risk_summary(synthetic_prices, portfolio_config)

    for field in PAGE_ONE_SUMMARY_FIELDS:
        assert hasattr(summary, field)


def test_summary_fields_are_formattable_by_dashboard_helpers(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    summary = build_portfolio_risk_summary(synthetic_prices, portfolio_config)

    percentage_fields = [
        "annualized_return",
        "annualized_volatility",
        "current_drawdown",
        "max_drawdown",
        "var_95",
        "cvar_95",
    ]
    for field in percentage_fields:
        formatted = format_percentage(getattr(summary, field))
        assert isinstance(formatted, str)
        assert formatted.endswith("%")

    formatted_holdings = format_decimal(summary.effective_holdings)
    assert isinstance(formatted_holdings, str)
    assert formatted_holdings


def test_format_percentage_values() -> None:
    assert format_percentage(0.1234) == "12.34%"
    assert format_percentage(-0.05) == "-5.00%"
    assert format_percentage(0.5, decimals=0) == "50%"


def test_format_decimal_values() -> None:
    assert format_decimal(1.5) == "1.50"
    assert format_decimal(2) == "2.00"
    assert format_decimal(3.14159, decimals=3) == "3.142"
