from __future__ import annotations

import pandas as pd
import pytest

from mrrp.risk.drawdown import (
    current_drawdown,
    drawdown_duration,
    drawdown_series,
    max_drawdown,
    rolling_max_drawdown,
)


@pytest.fixture
def known_wealth_returns() -> pd.Series:
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    wealth = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 130.0], index=index)
    return wealth.pct_change().fillna(0.0)


def test_known_drawdown_path(known_wealth_returns: pd.Series) -> None:
    result = drawdown_series(known_wealth_returns)

    assert result.iloc[2] == pytest.approx(-0.25)
    assert result.iloc[4] == pytest.approx(-1 / 3)
    assert result.index.equals(known_wealth_returns.index)


def test_max_drawdown_on_known_path(known_wealth_returns: pd.Series) -> None:
    assert max_drawdown(known_wealth_returns) == pytest.approx(-1 / 3)


def test_current_drawdown_at_new_high(known_wealth_returns: pd.Series) -> None:
    assert current_drawdown(known_wealth_returns) == pytest.approx(0.0)


def test_drawdown_duration_at_high_is_zero(
    known_wealth_returns: pd.Series,
) -> None:
    assert drawdown_duration(known_wealth_returns) == 0


def test_drawdown_duration_counts_underwater_periods(
    known_wealth_returns: pd.Series,
) -> None:
    assert drawdown_duration(known_wealth_returns.iloc[:-1]) == 3


def test_rising_only_series_has_zero_max_drawdown() -> None:
    returns = pd.Series([0.0, 0.10, 0.05, 0.02])

    assert max_drawdown(returns) == pytest.approx(0.0)


def test_falling_only_series_has_negative_max_drawdown() -> None:
    returns = pd.Series([0.0, -0.10, -0.10, -0.10])

    assert max_drawdown(returns) < 0.0


def test_flat_series_has_zero_max_drawdown() -> None:
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    assert max_drawdown(returns) == pytest.approx(0.0)


def test_drawdown_includes_initial_wealth_as_a_peak() -> None:
    returns = pd.Series([-0.10, 0.05])

    result = drawdown_series(returns)

    assert result.iloc[0] == pytest.approx(-0.10)
    assert max_drawdown(returns) == pytest.approx(-0.10)
    assert drawdown_duration(returns) == 2


def test_rolling_max_drawdown_preserves_index_and_initial_nans(
    known_wealth_returns: pd.Series,
) -> None:
    result = rolling_max_drawdown(known_wealth_returns, window=3)

    assert result.index.equals(known_wealth_returns.index)
    assert len(result) == len(known_wealth_returns)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(-0.25)
    assert result.iloc[2:].notna().all()


def test_rolling_max_drawdown_includes_window_starting_wealth() -> None:
    returns = pd.Series([-0.10, 0.05])

    result = rolling_max_drawdown(returns, window=2)

    assert result.iloc[-1] == pytest.approx(-0.10)


@pytest.mark.parametrize("window", [0, 1])
def test_rolling_max_drawdown_rejects_invalid_window(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_max_drawdown(pd.Series([0.0, -0.1]), window=window)


def test_empty_returns_raise_value_error() -> None:
    with pytest.raises(ValueError, match="valid observation"):
        drawdown_series(pd.Series(dtype=float))
