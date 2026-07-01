"""Variance-based portfolio risk contribution metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.portfolio.returns import align_returns_and_weights
from mrrp.portfolio.weights import validate_weights
from mrrp.risk.beta import compute_asset_betas


def compute_portfolio_variance(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Return periodic portfolio variance using a sample covariance matrix."""
    _, _, aligned_weights, covariance = _risk_inputs(returns, weights)
    return _portfolio_variance(covariance, aligned_weights)


def compute_marginal_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Return marginal contribution to variance, ``covariance @ weights``."""
    _, _, aligned_weights, covariance = _risk_inputs(returns, weights)
    return _marginal_contribution(covariance, aligned_weights)


def compute_component_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Return each holding's component contribution to portfolio variance."""
    _, _, aligned_weights, covariance = _risk_inputs(returns, weights)
    marginal = _marginal_contribution(covariance, aligned_weights)
    component = aligned_weights.astype(float) * marginal
    component.name = "risk_contribution"
    return component


def compute_percent_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Return component variance contributions as fractions of total variance."""
    _, _, aligned_weights, covariance = _risk_inputs(returns, weights)
    variance = _portfolio_variance(covariance, aligned_weights)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Portfolio variance must be positive and finite")

    marginal = _marginal_contribution(covariance, aligned_weights)
    percent = aligned_weights.astype(float) * marginal / variance
    percent.name = "risk_contribution_pct"
    return percent


def build_risk_contribution_table(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build an ordered holding-level table of periodic variance contributions."""
    aligned_returns, complete_returns, aligned_weights, covariance = _risk_inputs(
        returns,
        weights,
    )
    variance = _portfolio_variance(covariance, aligned_weights)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Portfolio variance must be positive and finite")

    marginal = _marginal_contribution(covariance, aligned_weights)
    component = aligned_weights.astype(float) * marginal
    table = pd.DataFrame(
        {
            "ticker": aligned_weights.index.tolist(),
            "weight": aligned_weights.to_numpy(dtype=float),
            "asset_volatility": complete_returns.std(ddof=1).to_numpy(dtype=float),
            "risk_contribution": component.to_numpy(dtype=float),
            "risk_contribution_pct": (component / variance).to_numpy(dtype=float),
        }
    )

    if benchmark_returns is not None:
        asset_betas = compute_asset_betas(aligned_returns, benchmark_returns)
        table["beta"] = asset_betas.reindex(aligned_weights.index).to_numpy(dtype=float)

    return table


def _risk_inputs(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    aligned_returns, aligned_weights = align_returns_and_weights(returns, weights)
    validate_weights(aligned_weights, allow_short=True)
    complete_returns = aligned_returns.dropna()
    if len(complete_returns) < 2:
        raise ValueError("Returns must contain at least 2 complete observations")

    try:
        values = complete_returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Returns must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Returns must contain finite values")

    covariance = complete_returns.cov(ddof=1)
    if not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise ValueError("Covariance matrix must contain finite values")
    return (
        aligned_returns,
        complete_returns,
        aligned_weights.astype(float),
        covariance,
    )


def _portfolio_variance(covariance: pd.DataFrame, weights: pd.Series) -> float:
    weight_values = weights.to_numpy(dtype=float)
    return float(weight_values @ covariance.to_numpy(dtype=float) @ weight_values)


def _marginal_contribution(
    covariance: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    values = covariance.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    return pd.Series(
        values,
        index=weights.index,
        name="marginal_risk_contribution",
    )
