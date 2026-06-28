from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.performance import (
    calmar_ratio,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)


def test_sharpe_is_nan_for_zero_volatility_returns() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    assert np.isnan(sharpe_ratio(returns))


def test_sortino_is_nan_without_downside_deviation() -> None:
    returns = pd.Series([0.0, 0.01, 0.02, 0.01])

    assert np.isnan(sortino_ratio(returns))


def test_calmar_is_nan_without_drawdown() -> None:
    returns = pd.Series([0.0, 0.01, 0.02, 0.01])

    assert np.isnan(calmar_ratio(returns))


def test_tracking_error_is_zero_for_identical_returns() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])

    assert tracking_error(returns, returns) == pytest.approx(0.0)


def test_information_ratio_is_nan_for_identical_returns() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])

    assert np.isnan(information_ratio(returns, returns))


def test_benchmark_metrics_align_mismatched_indexes() -> None:
    portfolio_index = pd.date_range("2025-01-01", periods=3, freq="D")
    benchmark_index = pd.date_range("2025-01-02", periods=3, freq="D")
    portfolio = pd.Series([0.50, 0.02, 0.00], index=portfolio_index)
    benchmark = pd.Series([0.01, 0.01, -0.50], index=benchmark_index)

    active_tracking_error = tracking_error(portfolio, benchmark, periods_per_year=2)
    ratio = information_ratio(portfolio, benchmark, periods_per_year=2)

    assert active_tracking_error == pytest.approx(0.02)
    assert ratio == pytest.approx(-0.005)


def test_known_active_return_toy_case() -> None:
    portfolio = pd.Series([0.00, 0.02])
    benchmark = pd.Series([0.01, 0.01])

    active_tracking_error = tracking_error(portfolio, benchmark, periods_per_year=2)
    ratio = information_ratio(portfolio, benchmark, periods_per_year=2)

    assert active_tracking_error == pytest.approx(0.02)
    assert ratio == pytest.approx(-0.005)


def test_sharpe_converts_annual_risk_free_rate_to_periodic() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.00])
    periods_per_year = 4
    annual_risk_free_rate = 0.04
    periodic_rate = (1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - periodic_rate
    expected_annual_return = (1 + excess_returns).prod() - 1
    expected_volatility = excess_returns.std(ddof=1) * np.sqrt(periods_per_year)

    result = sharpe_ratio(
        returns,
        risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    assert result == pytest.approx(expected_annual_return / expected_volatility)
