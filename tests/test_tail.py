from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.risk.tail import (
    historical_cvar,
    historical_var,
    kurtosis,
    skewness,
    worst_return,
    worst_rolling_return,
)


@pytest.fixture
def returns() -> pd.Series:
    return pd.Series([-0.10, -0.05, -0.02, 0.00, 0.01, 0.03, 0.04])


def test_historical_var_uses_lower_tail_percentile(returns: pd.Series) -> None:
    result = historical_var(returns, confidence=0.95)

    assert result == pytest.approx(returns.quantile(0.05))
    assert result < 0


def test_historical_cvar_averages_returns_at_or_below_var(
    returns: pd.Series,
) -> None:
    confidence = 0.75
    var_threshold = returns.quantile(1 - confidence)
    expected = returns[returns <= var_threshold].mean()

    result = historical_cvar(returns, confidence=confidence)

    assert result == pytest.approx(expected)
    assert result == pytest.approx(-0.075)


def test_worst_return(returns: pd.Series) -> None:
    assert worst_return(returns) == pytest.approx(-0.10)


def test_worst_rolling_return_uses_compounding(returns: pd.Series) -> None:
    result = worst_rolling_return(returns, window=2)

    assert result == pytest.approx((1 - 0.10) * (1 - 0.05) - 1)
    assert result == pytest.approx(-0.145)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_invalid_confidence_raises(
    returns: pd.Series,
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        historical_var(returns, confidence=confidence)
    with pytest.raises(ValueError, match="confidence"):
        historical_cvar(returns, confidence=confidence)


@pytest.mark.parametrize("window", [0, -1])
def test_invalid_window_raises(returns: pd.Series, window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        worst_rolling_return(returns, window=window)


def test_skewness_matches_pandas(returns: pd.Series) -> None:
    assert skewness(returns) == pytest.approx(returns.skew())


def test_kurtosis_matches_pandas(returns: pd.Series) -> None:
    assert kurtosis(returns) == pytest.approx(returns.kurt())


@pytest.mark.parametrize(
    "metric",
    [
        historical_var,
        historical_cvar,
        worst_return,
        lambda values: worst_rolling_return(values, window=2),
        skewness,
        kurtosis,
    ],
)
def test_empty_series_returns_nan(metric) -> None:
    assert np.isnan(metric(pd.Series(dtype=float)))


def test_nan_values_are_ignored_for_scalar_tail_metrics(
    returns: pd.Series,
) -> None:
    with_nan = pd.concat([returns, pd.Series([np.nan])], ignore_index=True)

    assert historical_var(with_nan) == pytest.approx(historical_var(returns))
    assert worst_return(with_nan) == pytest.approx(worst_return(returns))
