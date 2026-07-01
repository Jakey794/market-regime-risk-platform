from __future__ import annotations

import pandas as pd
import pytest

from mrrp.risk.risk_contribution import (
    build_risk_contribution_table,
    compute_component_risk_contribution,
    compute_marginal_risk_contribution,
    compute_percent_risk_contribution,
    compute_portfolio_variance,
)


@pytest.fixture
def returns_and_weights() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "HIGH": [0.04, -0.04, 0.04, -0.04],
            "LOW": [0.01, 0.01, -0.01, -0.01],
        },
        index=index,
    )
    weights = pd.Series({"LOW": 0.30, "HIGH": 0.70})
    return returns, weights


def test_portfolio_variance_positive(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    assert compute_portfolio_variance(returns, weights) > 0


def test_risk_contributions_sum_to_one(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    result = compute_percent_risk_contribution(returns, weights)

    assert result.sum() == pytest.approx(1.0)
    assert result.index.tolist() == ["LOW", "HIGH"]


def test_component_contributions_sum_to_variance(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    component = compute_component_risk_contribution(returns, weights)
    variance = compute_portfolio_variance(returns, weights)

    assert component.sum() == pytest.approx(variance)


def test_marginal_risk_contribution_matches_covariance_product(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    result = compute_marginal_risk_contribution(returns, weights)
    aligned_returns = returns.loc[:, weights.index]
    expected = aligned_returns.cov().dot(weights)

    assert result.name == "marginal_risk_contribution"
    assert result.to_dict() == pytest.approx(expected.to_dict())


def test_high_weight_high_vol_asset_has_larger_contribution(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    component = compute_component_risk_contribution(returns, weights)

    assert component["HIGH"] > component["LOW"]


def test_short_hedge_can_have_negative_component_contribution() -> None:
    returns = pd.DataFrame(
        {
            "LONG": [0.01, -0.02, 0.03, -0.01],
            "HEDGE": [0.01, -0.02, 0.03, -0.01],
        }
    )
    weights = pd.Series({"LONG": 1.20, "HEDGE": -0.20})

    component = compute_component_risk_contribution(returns, weights)

    assert component["HEDGE"] < 0
    assert component.sum() == pytest.approx(
        compute_portfolio_variance(returns, weights)
    )


def test_percent_contribution_rejects_zero_variance() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02]})
    weights = pd.Series({"A": 0.50, "B": 0.50})

    with pytest.raises(ValueError, match="Portfolio variance"):
        compute_percent_risk_contribution(returns, weights)


def test_risk_contribution_table_columns(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights
    benchmark = pd.Series(
        [-0.03, -0.01, 0.01, 0.03],
        index=returns.index,
    )

    result = build_risk_contribution_table(returns, weights, benchmark)

    assert result.columns.tolist() == [
        "ticker",
        "weight",
        "asset_volatility",
        "risk_contribution",
        "risk_contribution_pct",
        "beta",
    ]
    assert result["ticker"].tolist() == ["LOW", "HIGH"]


def test_risk_contribution_table_omits_beta_without_benchmark(
    returns_and_weights: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, weights = returns_and_weights

    result = build_risk_contribution_table(returns, weights)

    assert "beta" not in result.columns
