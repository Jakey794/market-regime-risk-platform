from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.beta import (
    beta,
    compute_asset_betas,
    compute_portfolio_beta,
    compute_rolling_portfolio_beta,
    compute_up_down_beta,
    portfolio_beta,
    rolling_beta,
)


def test_asset_twice_benchmark_has_beta_two() -> None:
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01, 0.02])
    asset = 2 * benchmark

    assert beta(asset, benchmark) == pytest.approx(2.0)


def test_asset_equal_to_benchmark_has_beta_one() -> None:
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01, 0.02])

    assert beta(benchmark, benchmark) == pytest.approx(1.0)


def test_zero_benchmark_variance_returns_nan() -> None:
    asset = pd.Series([0.01, 0.02, -0.01, 0.03])
    benchmark = pd.Series([0.01, 0.01, 0.01, 0.01])

    assert np.isnan(beta(asset, benchmark))


def test_insufficient_data_returns_nan() -> None:
    asset = pd.Series([0.01, np.nan])
    benchmark = pd.Series([0.02, 0.03])

    assert np.isnan(beta(asset, benchmark))


def test_rolling_beta_preserves_asset_index() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01, 0.02], index=index)
    asset = 2 * benchmark

    result = rolling_beta(asset, benchmark, window=3)

    assert result.index.equals(index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_rolling_beta_aligns_and_drops_nan_pairs() -> None:
    asset_index = pd.date_range("2025-01-01", periods=5, freq="D")
    benchmark_index = pd.date_range("2025-01-02", periods=5, freq="D")
    asset = pd.Series([0.10, 0.02, np.nan, 0.06, 0.08], index=asset_index)
    benchmark = pd.Series([0.01, 0.02, 0.03, 0.04, 0.50], index=benchmark_index)

    result = rolling_beta(asset, benchmark, window=2)

    assert result.index.equals(asset_index)
    assert result.notna().sum() == 2


def test_portfolio_beta_with_two_assets() -> None:
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01, 0.02])
    assets = pd.DataFrame({"HIGH": 2 * benchmark, "LOW": 0.5 * benchmark})

    result = portfolio_beta(
        assets,
        benchmark,
        weights={"HIGH": 0.5, "LOW": 0.5},
    )

    assert result == pytest.approx(1.25)


@pytest.mark.parametrize("window", [0, 1])
def test_rolling_beta_rejects_invalid_window(window: int) -> None:
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError, match="window"):
        rolling_beta(returns, returns, window=window)


def test_portfolio_beta_known_relationship() -> None:
    benchmark = pd.Series([-0.03, -0.01, 0.01, 0.02, 0.04])
    portfolio = 1.5 * benchmark

    assert compute_portfolio_beta(portfolio, benchmark) == pytest.approx(1.5)


def test_asset_betas_known_relationship() -> None:
    benchmark = pd.Series([-0.03, -0.01, 0.01, 0.02, 0.04])
    returns = pd.DataFrame({"DOUBLE": 2.0 * benchmark, "HALF": 0.5 * benchmark})

    result = compute_asset_betas(returns, benchmark)

    assert result.name == "beta"
    assert result.index.tolist() == ["DOUBLE", "HALF"]
    assert result.tolist() == pytest.approx([2.0, 0.5])


def test_rolling_beta_length() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    benchmark = pd.Series([-0.03, -0.01, 0.01, 0.02, -0.02, 0.04], index=index)
    portfolio = 1.25 * benchmark

    result = compute_rolling_portfolio_beta(portfolio, benchmark, window=3)

    assert len(result) == len(portfolio)
    assert result.index.equals(index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([1.25] * 4)


def test_up_down_beta_output_keys() -> None:
    benchmark = pd.Series([-0.04, -0.02, -0.01, 0.01, 0.02, 0.04])
    portfolio = 1.5 * benchmark

    result = compute_up_down_beta(portfolio, benchmark)

    assert set(result) == {"up_beta", "down_beta"}
    assert result == pytest.approx({"up_beta": 1.5, "down_beta": 1.5})


def test_zero_benchmark_variance_raises() -> None:
    portfolio = pd.Series([0.01, 0.02, -0.01, 0.03])
    benchmark = pd.Series([0.01, 0.01, 0.01, 0.01])

    with pytest.raises(ValueError, match="Benchmark variance"):
        compute_portfolio_beta(portfolio, benchmark)
    with pytest.raises(ValueError, match="Benchmark variance"):
        compute_rolling_portfolio_beta(portfolio, benchmark, window=3)


def test_compute_portfolio_beta_aligns_dates_and_drops_nan() -> None:
    portfolio_index = pd.date_range("2025-01-01", periods=5, freq="D")
    benchmark_index = pd.date_range("2025-01-02", periods=5, freq="D")
    benchmark = pd.Series([-0.02, 0.01, 0.03, -0.01, 0.50], index=benchmark_index)
    portfolio = pd.Series([0.90, -0.04, np.nan, 0.06, -0.02], index=portfolio_index)

    assert compute_portfolio_beta(portfolio, benchmark) == pytest.approx(2.0)
