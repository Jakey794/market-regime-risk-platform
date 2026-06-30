from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.correlation import (
    build_correlation_summary,
    classify_correlation_regime,
    compute_correlation_matrix,
    compute_diversification_ratio,
    compute_max_pairwise_correlation,
    compute_mean_pairwise_correlation,
    compute_rolling_mean_pairwise_correlation,
    mean_pairwise_correlation,
    rolling_correlation_matrix,
)


def _known_returns() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [2.0, 4.0, 6.0, 8.0],
            "C": [4.0, 3.0, 2.0, 1.0],
        },
        index=index,
    )


def test_correlation_matrix_shape() -> None:
    result = compute_correlation_matrix(_known_returns())

    assert result.shape == (3, 3)
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.columns.tolist() == ["A", "B", "C"]


def test_correlation_matrix_diagonal_is_one() -> None:
    result = compute_correlation_matrix(_known_returns())

    assert np.diag(result).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_mean_pairwise_correlation_known_matrix() -> None:
    assert compute_mean_pairwise_correlation(_known_returns()) == pytest.approx(
        -1 / 3
    )


def test_max_pairwise_correlation_known_matrix() -> None:
    assert compute_max_pairwise_correlation(_known_returns()) == pytest.approx(1.0)


def test_rolling_mean_pairwise_corr_length() -> None:
    returns = _known_returns()

    result = compute_rolling_mean_pairwise_correlation(returns, window=3)

    assert len(result) == len(returns)
    assert result.index.equals(returns.index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_diversification_ratio_basic_case() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.01, -0.01],
            "B": [0.02, 0.02, -0.02, -0.02],
        },
        index=index,
    )
    weights = pd.Series({"B": 0.25, "A": 0.75})

    result = compute_diversification_ratio(returns, weights)

    assert result == pytest.approx(1.25 / np.sqrt(0.8125))


def test_rolling_correlation_uses_no_future_observations() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, -0.02, 0.50],
            "B": [0.02, -0.01, 0.03, 0.01, -0.02, -0.50],
        },
        index=index,
    )

    full_result = compute_rolling_mean_pairwise_correlation(returns, window=3)
    prefix_result = compute_rolling_mean_pairwise_correlation(
        returns.iloc[:-1],
        window=3,
    )

    pd.testing.assert_series_equal(full_result.iloc[:-1], prefix_result)


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        (0.0, "Low correlation"),
        (50.0, "Normal correlation"),
        (80.0, "High correlation"),
        (100.0, "Crisis-like correlation"),
    ],
)
def test_correlation_regime_classification(latest: float, expected: str) -> None:
    rolling_correlation = pd.Series([*range(100), latest], dtype=float)

    assert classify_correlation_regime(rolling_correlation) == expected


def test_correlation_summary_columns() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="D")
    x = np.linspace(-0.03, 0.03, len(index))
    returns = pd.DataFrame(
        {
            "A": x,
            "B": np.sin(np.arange(len(index))) * 0.02,
            "C": np.cos(np.arange(len(index))) * 0.01,
        },
        index=index,
    )
    weights = pd.Series({"C": 0.2, "A": 0.5, "B": 0.3})

    result = build_correlation_summary(returns, weights, window=5)

    assert result.shape == (1, 6)
    assert result.columns.tolist() == [
        "mean_pairwise_corr",
        "max_pairwise_corr",
        "current_rolling_corr",
        "corr_percentile",
        "correlation_regime",
        "diversification_ratio",
    ]


def test_fewer_than_two_assets_raises() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, 0.02]},
        index=pd.date_range("2025-01-01", periods=2, freq="D"),
    )

    with pytest.raises(ValueError, match="at least 2 assets"):
        compute_correlation_matrix(returns)


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
