"""Beta metrics for assets and portfolios."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from mrrp.risk.returns import portfolio_returns


def beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate asset beta from aligned, pairwise-complete returns."""
    aligned = _aligned_returns(asset_returns, benchmark_returns)
    return _beta_from_aligned(aligned, strict_variance=False)


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


def compute_asset_betas(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Calculate strict beta for each asset using pairwise-aligned returns."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("Returns must contain at least one asset")
    if returns.columns.has_duplicates:
        raise ValueError("Return columns must be unique")

    values = {
        column: compute_portfolio_beta(returns[column], benchmark_returns)
        for column in returns.columns
    }
    return pd.Series(values, index=returns.columns, dtype=float, name="beta")


def compute_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate beta and raise when aligned benchmark variance is zero."""
    aligned = _aligned_returns(portfolio_returns, benchmark_returns)
    return _beta_from_aligned(aligned, strict_variance=True)


def compute_rolling_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Calculate strict trailing beta using observations available through each date."""
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 1:
        raise ValueError("window must be an integer greater than 1")

    aligned = _aligned_returns(portfolio_returns, benchmark_returns)
    rolling_covariance = aligned["asset"].rolling(window).cov(aligned["benchmark"])
    rolling_variance = aligned["benchmark"].rolling(window).var(ddof=1)
    completed_variance = rolling_variance.dropna()
    if (
        not completed_variance.empty
        and (
            not np.isfinite(completed_variance.to_numpy()).all()
            or completed_variance.eq(0).any()
        )
    ):
        raise ValueError("Benchmark variance must be positive and finite")

    result = (rolling_covariance / rolling_variance).reindex(portfolio_returns.index)
    result.name = "beta"
    return result


def compute_up_down_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    """Calculate beta during strictly positive and negative benchmark periods."""
    aligned = _aligned_returns(portfolio_returns, benchmark_returns)
    up_returns = aligned.loc[aligned["benchmark"] > 0]
    down_returns = aligned.loc[aligned["benchmark"] < 0]
    return {
        "up_beta": _beta_from_aligned(up_returns, strict_variance=True),
        "down_beta": _beta_from_aligned(down_returns, strict_variance=True),
    }


def _beta_from_aligned(
    aligned: pd.DataFrame,
    *,
    strict_variance: bool,
) -> float:
    if len(aligned) < 2:
        return float(np.nan)

    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    if not np.isfinite(benchmark_variance) or benchmark_variance == 0:
        if strict_variance:
            raise ValueError("Benchmark variance must be positive and finite")
        return float(np.nan)

    covariance = aligned["asset"].cov(aligned["benchmark"], ddof=1)
    return float(covariance / benchmark_variance)


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
            values = aligned.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Returns must contain numeric values") from exc
        if not np.isfinite(values).all():
            raise ValueError("Returns must contain finite values")

    return aligned
