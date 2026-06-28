from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.volatility import (
    annualized_return,
    annualized_volatility,
    ewma_volatility,
    rolling_volatility,
)


def test_zero_returns_produce_zero_volatility() -> None:
    returns = pd.Series([0.0, 0.0, 0.0])

    result = annualized_volatility(returns)

    assert result == pytest.approx(0.0)


def test_constant_positive_returns_produce_zero_volatility() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    result = annualized_volatility(returns)

    assert result == pytest.approx(0.0)


def test_known_annualized_volatility() -> None:
    returns = pd.Series([0.0, 0.01, 0.02])

    result = annualized_volatility(returns, periods_per_year=4)

    assert result == pytest.approx(0.02)


def test_known_compound_annualized_return() -> None:
    returns = pd.Series([0.10, 0.10])

    result = annualized_return(returns, periods_per_year=4)

    assert result == pytest.approx(0.4641)


def test_rolling_volatility_preserves_index_and_length() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01], index=index)

    result = rolling_volatility(returns, window=3)

    assert result.index.equals(index)
    assert len(result) == len(returns)


def test_rolling_volatility_has_expected_initial_nans() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])

    result = rolling_volatility(returns, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_ewma_volatility_aligns_with_input_index() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, -0.01, 0.02, 0.0], index=index)

    result = ewma_volatility(returns, span=3)

    assert result.index.equals(index)
    assert len(result) == len(returns)


def test_metrics_drop_nan_observations() -> None:
    returns = pd.Series([0.0, np.nan, 0.01, 0.02])

    annualized = annualized_volatility(returns, periods_per_year=4)
    rolling = rolling_volatility(returns, window=2, periods_per_year=4)
    ewma = ewma_volatility(returns, span=2, periods_per_year=4)

    assert annualized == pytest.approx(0.02)
    assert np.isnan(rolling.iloc[1])
    assert np.isnan(ewma.iloc[1])
    assert rolling.index.equals(returns.index)
    assert ewma.index.equals(returns.index)


@pytest.mark.parametrize(
    "metric",
    [
        lambda: annualized_return(pd.Series([0.01]), periods_per_year=0),
        lambda: annualized_volatility(pd.Series([0.01]), periods_per_year=0),
        lambda: rolling_volatility(
            pd.Series([0.01, 0.02]), window=2, periods_per_year=0
        ),
        lambda: ewma_volatility(pd.Series([0.01]), periods_per_year=0),
    ],
)
def test_invalid_periods_per_year_raises(metric) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        metric()


@pytest.mark.parametrize("window", [0, 1])
def test_invalid_window_raises(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_volatility(pd.Series([0.01, 0.02]), window=window)


@pytest.mark.parametrize("span", [0, -1])
def test_invalid_span_raises(span: int) -> None:
    with pytest.raises(ValueError, match="span"):
        ewma_volatility(pd.Series([0.01, 0.02]), span=span)


@pytest.mark.parametrize(
    "metric",
    [
        annualized_return,
        annualized_volatility,
        lambda values: rolling_volatility(values, window=2),
        ewma_volatility,
    ],
)
def test_all_nan_input_raises(metric) -> None:
    with pytest.raises(ValueError, match="valid observation"):
        metric(pd.Series([np.nan, np.nan]))
