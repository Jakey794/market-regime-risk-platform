from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.returns import (
    cumulative_returns,
    log_returns,
    portfolio_returns,
    simple_returns,
    validate_weights,
    wealth_index,
)


def test_simple_returns_on_known_prices() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])

    result = simple_returns(prices)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.1)
    assert result.iloc[2] == pytest.approx(-0.1)


def test_log_returns_on_known_prices() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])

    result = log_returns(prices)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(1.1))
    assert result.iloc[2] == pytest.approx(np.log(0.9))


def test_portfolio_returns_with_fixed_weights() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.10, -0.02],
            "BND": [0.02, 0.01],
        }
    )

    result = portfolio_returns(returns, {"SPY": 0.6, "BND": 0.4})

    assert result.tolist() == pytest.approx([0.068, -0.008])


@pytest.mark.parametrize(
    "weights",
    [
        {"SPY": 0.7, "BND": 0.4},
        {"SPY": np.nan, "BND": np.nan},
    ],
)
def test_invalid_weights_raise_value_error(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        validate_weights(weights, ["SPY", "BND"])


def test_missing_weight_ticker_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing tickers"):
        validate_weights({"SPY": 1.0}, ["SPY", "BND"])


def test_cumulative_returns_on_known_path() -> None:
    returns = pd.Series([0.10, -0.10, 0.20])

    result = cumulative_returns(returns)

    assert result.tolist() == pytest.approx([0.10, -0.01, 0.188])


def test_wealth_index_on_known_path() -> None:
    returns = pd.Series([0.10, -0.10, 0.20])

    result = wealth_index(returns, start_value=100.0)

    assert result.tolist() == pytest.approx([110.0, 99.0, 118.8])


def test_returns_preserve_date_index() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {"SPY": [100.0, 102.0, 101.0], "BND": [50.0, 50.5, 51.0]},
        index=index,
    )

    simple = simple_returns(prices)
    logged = log_returns(prices)
    portfolio = portfolio_returns(simple, {"SPY": 0.5, "BND": 0.5})

    assert simple.index.equals(index)
    assert logged.index.equals(index)
    assert portfolio.index.equals(index)


@pytest.mark.parametrize("invalid_price", [0.0, -1.0])
def test_return_calculations_reject_non_positive_prices(
    invalid_price: float,
) -> None:
    prices = pd.Series([100.0, invalid_price])

    with pytest.raises(ValueError, match="positive"):
        simple_returns(prices)
    with pytest.raises(ValueError, match="positive"):
        log_returns(prices)


def test_wealth_index_rejects_non_positive_start_value() -> None:
    with pytest.raises(ValueError, match="positive"):
        wealth_index(pd.Series([0.1]), start_value=0.0)
