from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.portfolio import (
    align_returns_and_weights,
    compute_asset_returns,
    compute_cumulative_returns,
    compute_long_short_exposure,
    compute_portfolio_returns,
    compute_top_n_exposure,
    compute_weight_exposure,
)


def test_portfolio_return_weighted_sum() -> None:
    index = pd.date_range("2025-01-02", periods=2, freq="D")
    returns = pd.DataFrame(
        {"SPY": [0.10, -0.02], "BND": [0.02, 0.01]},
        index=index,
    )
    weights = pd.Series({"SPY": 0.6, "BND": 0.4})

    result = compute_portfolio_returns(returns, weights)

    assert result.tolist() == pytest.approx([0.068, -0.008])
    assert result.index.equals(index)


def test_portfolio_returns_align_columns() -> None:
    returns = pd.DataFrame({"BND": [0.02], "SPY": [0.10], "EFA": [0.05]})
    weights = pd.Series({"SPY": 0.75, "BND": 0.25})

    aligned_returns, aligned_weights = align_returns_and_weights(returns, weights)
    result = compute_portfolio_returns(returns, weights)

    assert aligned_returns.columns.tolist() == ["SPY", "BND"]
    assert aligned_weights.index.tolist() == ["SPY", "BND"]
    assert result.iloc[0] == pytest.approx(0.08)


def test_missing_weight_ticker_raises() -> None:
    returns = pd.DataFrame({"SPY": [0.01], "BND": [0.02]})
    weights = pd.Series({"SPY": 0.5, "QQQ": 0.5})

    with pytest.raises(ValueError, match="missing return columns.*QQQ"):
        compute_portfolio_returns(returns, weights)


def test_portfolio_returns_do_not_normalize_weights() -> None:
    returns = pd.DataFrame({"SPY": [0.10], "BND": [0.10]})
    weights = pd.Series({"SPY": 0.30, "BND": 0.20})

    result = compute_portfolio_returns(returns, weights)

    assert result.iloc[0] == pytest.approx(0.05)


def test_cumulative_returns_known_series() -> None:
    index = pd.date_range("2025-01-02", periods=3, freq="D")
    returns = pd.Series([0.10, -0.10, 0.20], index=index)

    result = compute_cumulative_returns(returns)

    assert result.tolist() == pytest.approx([0.10, -0.01, 0.188])
    assert result.index.equals(index)


def test_compute_asset_returns_simple() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 99.0]}, index=index)

    result = compute_asset_returns(prices, method="simple")

    assert result["SPY"].tolist() == pytest.approx([0.10, -0.10])
    assert result.index.equals(index[1:])


def test_compute_asset_returns_log() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 99.0]}, index=index)

    result = compute_asset_returns(prices, method="log")

    assert result["SPY"].tolist() == pytest.approx(
        [np.log(1.1), np.log(0.9)]
    )
    assert result.index.equals(index[1:])


def test_top_n_exposure() -> None:
    weights = pd.Series({"SPY": 0.50, "QQQ": 0.30, "EFA": 0.20})

    assert compute_top_n_exposure(weights, n=2) == pytest.approx(0.80)


def test_long_short_exposure_long_only() -> None:
    weights = pd.Series({"SPY": 0.60, "BND": 0.40})

    result = compute_long_short_exposure(weights)

    assert result == pytest.approx(
        {"gross": 1.0, "net": 1.0, "long": 1.0, "short": 0.0}
    )


def test_long_short_exposure_with_short() -> None:
    weights = pd.Series({"SPY": 0.80, "QQQ": 0.40, "EFA": -0.20})

    result = compute_long_short_exposure(weights)

    assert result == pytest.approx(
        {"gross": 1.40, "net": 1.0, "long": 1.20, "short": 0.20}
    )


def test_weight_exposure_sorted_by_absolute_weight() -> None:
    weights = pd.Series({"EFA": -0.10, "SPY": 0.60, "QQQ": 0.50})

    result = compute_weight_exposure(weights)

    assert result.index.tolist() == ["SPY", "QQQ", "EFA"]
    assert result.columns.tolist() == ["weight", "absolute_weight"]
    assert result["absolute_weight"].tolist() == pytest.approx([0.60, 0.50, 0.10])
