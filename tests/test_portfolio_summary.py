from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from mrrp.portfolio import (
    PortfolioConfig,
    PortfolioRiskSummary,
    build_correlation_table,
    build_exposure_table,
    build_portfolio_risk_summary,
    build_risk_contribution_table,
    build_summary_cards,
)


SUMMARY_FIELDS = [
    "portfolio_name",
    "benchmark",
    "start_date",
    "end_date",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "current_drawdown",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "portfolio_beta",
    "concentration_label",
    "correlation_regime",
    "effective_holdings",
    "top_3_weight",
    "mean_pairwise_corr",
    "diversification_ratio",
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


def test_portfolio_summary_contract(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    result = build_portfolio_risk_summary(synthetic_prices, portfolio_config)

    assert isinstance(result, PortfolioRiskSummary)
    assert [field.name for field in fields(result)] == SUMMARY_FIELDS
    assert result.portfolio_name == "test_portfolio"
    assert result.benchmark == "SPY"
    assert result.start_date == synthetic_prices.index[0]
    assert result.end_date == synthetic_prices.index[-1]


def test_portfolio_summary_no_missing_fields(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    result = build_portfolio_risk_summary(synthetic_prices, portfolio_config)
    values = build_summary_cards(result)

    assert all(value is not None for value in values.values())
    numeric_values = [
        value
        for value in values.values()
        if isinstance(value, float)
    ]
    assert np.isfinite(numeric_values).all()


def test_portfolio_summary_handles_benchmark(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    internal = build_portfolio_risk_summary(synthetic_prices, portfolio_config)
    external = build_portfolio_risk_summary(
        synthetic_prices.drop(columns="SPY"),
        portfolio_config,
        benchmark_prices=synthetic_prices["SPY"],
    )

    assert internal.portfolio_beta == pytest.approx(external.portfolio_beta)


def test_portfolio_summary_rejects_bad_input(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    with pytest.raises(ValueError, match="missing portfolio tickers"):
        build_portfolio_risk_summary(
            synthetic_prices.drop(columns="ASSET_B"),
            portfolio_config,
        )
    with pytest.raises(ValueError, match="Benchmark prices are missing"):
        build_portfolio_risk_summary(
            synthetic_prices.drop(columns="SPY"),
            portfolio_config,
        )
    with pytest.raises(ValueError, match="at least 63 complete returns"):
        build_portfolio_risk_summary(synthetic_prices.iloc[:20], portfolio_config)

    invalid_config = PortfolioConfig(
        name="invalid",
        benchmark="SPY",
        currency="CAD",
        allow_short=False,
        holdings=pd.Series({"ASSET_A": 0.70, "ASSET_B": 0.40}),
    )
    with pytest.raises(ValueError, match="must sum to 1.0"):
        build_portfolio_risk_summary(synthetic_prices, invalid_config)


def test_summary_cards_are_dashboard_ready(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    summary = build_portfolio_risk_summary(synthetic_prices, portfolio_config)

    result = build_summary_cards(summary)

    assert isinstance(result, dict)
    assert list(result) == SUMMARY_FIELDS
    assert result["portfolio_name"] == "test_portfolio"


def test_exposure_table_columns(portfolio_config: PortfolioConfig) -> None:
    metadata = {
        "ASSET_A": {
            "asset_class": "Equity",
            "region": "Canada",
            "style": "Large Blend",
            "sector_proxy": "Broad Market",
        }
    }

    result = build_exposure_table(portfolio_config, metadata)

    assert result.columns.tolist() == [
        "ticker",
        "weight",
        "absolute_weight",
        "asset_class",
        "region",
        "style",
        "sector_proxy",
    ]
    assert result["ticker"].tolist() == ["ASSET_A", "ASSET_B"]
    assert result.loc[result["ticker"] == "ASSET_B", "region"].item() == "Unknown"


def test_correlation_table_columns(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    result = build_correlation_table(synthetic_prices, portfolio_config)

    assert result.index.tolist() == ["ASSET_A", "ASSET_B"]
    assert result.columns.tolist() == ["ASSET_A", "ASSET_B"]


def test_risk_contribution_table_columns(
    synthetic_prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> None:
    result = build_risk_contribution_table(synthetic_prices, portfolio_config)

    assert result.columns.tolist() == [
        "ticker",
        "weight",
        "asset_volatility",
        "risk_contribution",
        "risk_contribution_pct",
        "beta",
    ]
    assert result["ticker"].tolist() == ["ASSET_A", "ASSET_B"]
