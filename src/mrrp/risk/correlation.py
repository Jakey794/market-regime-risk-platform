"""Static and rolling portfolio correlation risk metrics."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from mrrp.portfolio.returns import align_returns_and_weights
from mrrp.portfolio.weights import validate_weights


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for at least two assets."""
    _validate_returns(returns)
    return returns.corr()


def compute_rolling_correlation_matrices(
    returns: pd.DataFrame,
    window: int = 63,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Map each complete rolling-window end date to its correlation matrix.

    Each matrix at date ``t`` uses only complete observations available on or
    before ``t`` and therefore introduces no look-ahead.
    """
    _validate_rolling_inputs(returns, window)
    valid_returns = returns.dropna()
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}

    for end_position in range(window - 1, len(valid_returns)):
        window_returns = valid_returns.iloc[
            end_position - window + 1 : end_position + 1
        ]
        timestamp = pd.Timestamp(valid_returns.index[end_position])
        matrices[timestamp] = compute_correlation_matrix(window_returns)

    return matrices


def compute_mean_pairwise_correlation(returns: pd.DataFrame) -> float:
    """Return the mean finite off-diagonal asset correlation."""
    matrix = compute_correlation_matrix(returns)
    values = _finite_pairwise_values(matrix)
    return float(values.mean()) if values.size else float(np.nan)


def compute_rolling_mean_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Return mean pairwise correlation for each trailing window end date."""
    matrices = compute_rolling_correlation_matrices(returns, window)
    result = pd.Series(
        np.nan,
        index=returns.index,
        name="mean_pairwise_correlation",
    )

    for timestamp, matrix in matrices.items():
        values = _finite_pairwise_values(matrix)
        if values.size:
            result.loc[timestamp] = float(values.mean())

    return result


def compute_max_pairwise_correlation(returns: pd.DataFrame) -> float:
    """Return the maximum finite off-diagonal asset correlation."""
    matrix = compute_correlation_matrix(returns)
    values = _finite_pairwise_values(matrix)
    return float(values.max()) if values.size else float(np.nan)


def compute_diversification_ratio(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Return weighted asset volatility divided by portfolio volatility."""
    aligned_returns, aligned_weights = align_returns_and_weights(returns, weights)
    _validate_returns(aligned_returns)
    validate_weights(aligned_weights, allow_short=True)

    complete_returns = aligned_returns.dropna()
    if len(complete_returns) < 2:
        raise ValueError("Returns must contain at least 2 complete observations")

    asset_volatility = complete_returns.std(ddof=1)
    weighted_asset_volatility = float(asset_volatility.dot(aligned_weights))
    portfolio_volatility = float(complete_returns.dot(aligned_weights).std(ddof=1))
    if not np.isfinite(portfolio_volatility) or portfolio_volatility <= 0:
        raise ValueError("Portfolio volatility must be positive and finite")

    return weighted_asset_volatility / portfolio_volatility


def classify_correlation_regime(rolling_corr: pd.Series) -> str:
    """Classify the latest correlation against its history through that date."""
    valid_correlation = _valid_rolling_correlation(rolling_corr)
    latest = float(valid_correlation.iloc[-1])
    percentile_33 = float(valid_correlation.quantile(0.33))
    percentile_75 = float(valid_correlation.quantile(0.75))
    percentile_90 = float(valid_correlation.quantile(0.90))

    if latest < percentile_33:
        return "Low correlation"
    if latest <= percentile_75:
        return "Normal correlation"
    if latest <= percentile_90:
        return "High correlation"
    return "Crisis-like correlation"


def build_correlation_summary(
    returns: pd.DataFrame,
    weights: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """Build a one-row summary of current portfolio correlation risk."""
    aligned_returns, aligned_weights = align_returns_and_weights(returns, weights)
    rolling_correlation = compute_rolling_mean_pairwise_correlation(
        aligned_returns,
        window,
    )
    valid_rolling = _valid_rolling_correlation(rolling_correlation)
    current_rolling = float(valid_rolling.iloc[-1])
    correlation_percentile = float(valid_rolling.le(current_rolling).mean())

    return pd.DataFrame(
        [
            {
                "mean_pairwise_corr": compute_mean_pairwise_correlation(
                    aligned_returns
                ),
                "max_pairwise_corr": compute_max_pairwise_correlation(aligned_returns),
                "current_rolling_corr": current_rolling,
                "corr_percentile": correlation_percentile,
                "correlation_regime": classify_correlation_regime(rolling_correlation),
                "diversification_ratio": compute_diversification_ratio(
                    aligned_returns,
                    aligned_weights,
                ),
            }
        ]
    )


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 63,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Compatibility wrapper for :func:`compute_rolling_correlation_matrices`."""
    return compute_rolling_correlation_matrices(returns, window)


def mean_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Compatibility wrapper for rolling mean pairwise correlation."""
    return compute_rolling_mean_pairwise_correlation(returns, window)


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
    if returns.shape[1] < 2:
        raise ValueError("Returns must contain at least 2 assets")
    if returns.columns.has_duplicates:
        raise ValueError("Return columns must be unique")

    try:
        values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Returns must contain numeric values") from exc
    if not (np.isfinite(values) | pd.isna(values)).all():
        raise ValueError("Returns must contain finite values or NaN")


def _validate_rolling_inputs(returns: pd.DataFrame, window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 1:
        raise ValueError("window must be an integer greater than 1")
    _validate_returns(returns)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be a DatetimeIndex")


def _finite_pairwise_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    pairwise_values = values[np.triu_indices(len(matrix), k=1)]
    return pairwise_values[np.isfinite(pairwise_values)]


def _valid_rolling_correlation(rolling_corr: pd.Series) -> pd.Series:
    if not isinstance(rolling_corr, pd.Series):
        raise ValueError("rolling_corr must be a pandas Series")
    valid_correlation = rolling_corr.dropna()
    if valid_correlation.empty:
        raise ValueError("rolling_corr must contain at least one valid value")

    try:
        values = valid_correlation.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rolling_corr must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("rolling_corr must contain finite values")
    return valid_correlation
