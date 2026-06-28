"""Rolling correlation metrics for return DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 63,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Map valid rolling-window end timestamps to correlation matrices."""
    _validate_inputs(returns, window)
    valid_returns = returns.dropna()
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}

    for end_position in range(window - 1, len(valid_returns)):
        window_returns = valid_returns.iloc[
            end_position - window + 1 : end_position + 1
        ]
        timestamp = pd.Timestamp(valid_returns.index[end_position])
        matrices[timestamp] = window_returns.corr()

    return matrices


def mean_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Calculate mean off-diagonal correlation for each rolling window."""
    matrices = rolling_correlation_matrix(returns, window)
    result = pd.Series(
        np.nan,
        index=returns.index,
        name="mean_pairwise_correlation",
    )

    for timestamp, matrix in matrices.items():
        values = matrix.to_numpy()
        pairwise_values = values[np.triu_indices(len(matrix), k=1)]
        finite_values = pairwise_values[np.isfinite(pairwise_values)]
        if finite_values.size:
            result.loc[timestamp] = float(finite_values.mean())

    return result


def _validate_inputs(returns: pd.DataFrame, window: int) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns index must be a DatetimeIndex")
    if window <= 1:
        raise ValueError("window must be greater than 1")

    valid_returns = returns.dropna()
    if not valid_returns.empty:
        try:
            all_finite = np.isfinite(valid_returns.to_numpy()).all()
        except TypeError as exc:
            raise ValueError("Returns must contain numeric values") from exc
        if not all_finite:
            raise ValueError("Returns must contain finite values")
