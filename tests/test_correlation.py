from __future__ import annotations

import pandas as pd
import pytest

from mrrp.risk.correlation import (
    mean_pairwise_correlation,
    rolling_correlation_matrix,
)


def test_mean_pairwise_correlation_excludes_diagonal() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]},
        index=index,
    )

    result = mean_pairwise_correlation(returns, window=3)

    assert result.iloc[-1] == pytest.approx(-1.0)


def test_identical_assets_have_mean_pairwise_correlation_one() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    values = [0.01, -0.02, 0.03, 0.00]
    returns = pd.DataFrame({"A": values, "B": values}, index=index)

    result = mean_pairwise_correlation(returns, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([1.0, 1.0])


def test_correlation_matrix_keys_are_window_end_timestamps() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.03, 0.00], "B": [0.0, 0.01, 0.02, -0.01, 0.03]},
        index=index,
    )

    matrices = rolling_correlation_matrix(returns, window=3)

    assert list(matrices) == list(index[2:])
    assert all(isinstance(key, pd.Timestamp) for key in matrices)
    assert all(matrix.shape == (2, 2) for matrix in matrices.values())


def test_missing_rows_are_dropped_before_rolling_windows() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, None, 0.02, 0.03], "B": [0.02, 0.01, 0.03, 0.04]},
        index=index,
    )

    matrices = rolling_correlation_matrix(returns, window=3)
    mean_correlation = mean_pairwise_correlation(returns, window=3)

    assert list(matrices) == [index[-1]]
    assert mean_correlation.index.equals(index)
    assert mean_correlation.iloc[:3].isna().all()
    assert mean_correlation.iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    [rolling_correlation_matrix, mean_pairwise_correlation],
)
def test_correlation_metrics_reject_invalid_window(metric) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=index)

    with pytest.raises(ValueError, match="window"):
        metric(returns, window=1)
