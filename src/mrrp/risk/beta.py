"""Beta metrics for assets and portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.risk.returns import portfolio_returns


def beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate asset beta from aligned, pairwise-complete returns."""
    aligned = _aligned_returns(asset_returns, benchmark_returns)
    if len(aligned) < 2:
        return float(np.nan)

    benchmark_variance = aligned["benchmark"].var(ddof=1)
    if pd.isna(benchmark_variance) or benchmark_variance == 0:
        return float(np.nan)

    covariance = aligned["asset"].cov(aligned["benchmark"], ddof=1)
    return float(covariance / benchmark_variance)


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Calculate rolling beta and restore the asset return index."""
    if window <= 1:
        raise ValueError("window must be greater than 1")

    aligned = _aligned_returns(asset_returns, benchmark_returns)
    rolling_covariance = aligned["asset"].rolling(window).cov(aligned["benchmark"])
    rolling_variance = aligned["benchmark"].rolling(window).var(ddof=1)
    values = (rolling_covariance / rolling_variance).where(rolling_variance.ne(0))

    result = values.reindex(asset_returns.index)
    result.name = "beta"
    return result


def portfolio_beta(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: dict[str, float],
) -> float:
    """Calculate beta for a weighted portfolio of asset returns."""
    weighted_returns = portfolio_returns(asset_returns, weights)
    return beta(weighted_returns, benchmark_returns)


def _aligned_returns(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.Series) or not isinstance(
        benchmark_returns, pd.Series
    ):
        raise ValueError("Asset and benchmark returns must be pandas Series")

    aligned = pd.concat(
        {"asset": asset_returns, "benchmark": benchmark_returns},
        axis=1,
        join="inner",
    ).dropna()

    if not aligned.empty:
        try:
            all_finite = np.isfinite(aligned.to_numpy()).all()
        except TypeError as exc:
            raise ValueError("Returns must contain numeric values") from exc
        if not all_finite:
            raise ValueError("Returns must contain finite values")

    return aligned
