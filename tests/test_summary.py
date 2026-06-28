from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.summary import portfolio_risk_summary


EXPECTED_METRICS = [
    "Annualized return",
    "Annualized volatility",
    "Sharpe ratio",
    "Sortino ratio",
    "Max drawdown",
    "Current drawdown",
    "95% VaR",
    "99% VaR",
    "95% CVaR",
    "99% CVaR",
    "Beta vs benchmark",
    "Tracking error",
    "Information ratio",
    "Worst day",
    "Worst rolling 21D return",
]


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=31, freq="B")
    benchmark_returns = np.resize(
        np.array([0.008, -0.006, 0.004, -0.010, 0.012, 0.002]),
        30,
    )
    asset_a_returns = 1.2 * benchmark_returns + np.resize(
        np.array([0.002, -0.001, 0.001]),
        30,
    )
    asset_b_returns = 0.5 * benchmark_returns + np.resize(
        np.array([-0.001, 0.002, -0.002, 0.001]),
        30,
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


def test_summary_returns_expected_metric_names(
    synthetic_prices: pd.DataFrame,
) -> None:
    summary = portfolio_risk_summary(
        synthetic_prices,
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        benchmark="SPY",
    )

    assert summary["metric"].tolist() == EXPECTED_METRICS
    assert list(summary.columns) == ["metric", "value", "description"]


def test_summary_values_are_numeric(synthetic_prices: pd.DataFrame) -> None:
    summary = portfolio_risk_summary(
        synthetic_prices,
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        benchmark="SPY",
    )

    assert pd.api.types.is_float_dtype(summary["value"])
    assert summary["value"].map(np.isscalar).all()


def test_missing_benchmark_raises(synthetic_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Benchmark column is missing"):
        portfolio_risk_summary(
            synthetic_prices,
            weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
            benchmark="MISSING",
        )


@pytest.mark.parametrize(
    "weights",
    [
        {"ASSET_A": 1.0},
        {"ASSET_A": 0.6, "ASSET_B": 0.3},
        {"ASSET_A": 0.6, "ASSET_B": 0.4, "SPY": 0.0},
    ],
)
def test_invalid_weights_raise(
    synthetic_prices: pd.DataFrame,
    weights: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        portfolio_risk_summary(
            synthetic_prices,
            weights=weights,
            benchmark="SPY",
        )


def test_summary_works_on_synthetic_prices(
    synthetic_prices: pd.DataFrame,
) -> None:
    summary = portfolio_risk_summary(
        synthetic_prices,
        weights={"ASSET_A": 0.6, "ASSET_B": 0.4},
        benchmark="SPY",
        periods_per_year=252,
    )

    assert summary.shape == (15, 3)
    assert summary["description"].str.len().gt(0).all()
    assert summary["value"].notna().all()
