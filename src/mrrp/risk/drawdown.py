"""Drawdown metrics calculated from periodic return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.risk.returns import wealth_index


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate the drawdown path from periodic fractional returns.

    The input is a return series, not a price or wealth series. Missing return
    observations remain missing in the output and do not reset compounding.
    """
    _validate_returns(returns)

    wealth = wealth_index(returns, start_value=1.0)
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Return the worst drawdown produced by periodic fractional returns."""
    return float(drawdown_series(returns).min())


def current_drawdown(returns: pd.Series) -> float:
    """Return the final drawdown produced by periodic fractional returns."""
    return float(drawdown_series(returns).iloc[-1])


def drawdown_duration(returns: pd.Series) -> int:
    """Return the number of valid periods since the last all-time high."""
    drawdowns = drawdown_series(returns).dropna()
    at_high = np.isclose(drawdowns.to_numpy(), 0.0, rtol=0.0, atol=1e-12)

    if at_high[-1]:
        return 0

    high_positions = np.flatnonzero(at_high)
    last_high_position = int(high_positions[-1]) if high_positions.size else -1
    return len(drawdowns) - last_high_position - 1


def worst_drawdown_periods(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Identify the worst distinct drawdown episodes, ranked by depth.

    Each episode spans a contiguous run of periods below the prior all-time
    high. ``recovery`` is ``pd.NaT`` if the episode is still underwater at the
    end of the series. ``duration`` counts valid-observation periods between
    ``start`` and ``recovery`` (or the last observation, if unrecovered),
    matching :func:`drawdown_duration`'s existing period-counting convention.
    """
    _validate_top_n(top_n)
    drawdowns = drawdown_series(returns).dropna()
    values = drawdowns.to_numpy()
    at_high = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)

    episodes: list[dict[str, object]] = []
    n_periods = len(values)
    last_high_position = -1
    position = 0
    while position < n_periods:
        if at_high[position]:
            last_high_position = position
            position += 1
            continue

        run_start = position
        while position < n_periods and not at_high[position]:
            position += 1
        run_end = position  # exclusive; == n_periods if unrecovered

        run_values = values[run_start:run_end]
        trough_position = run_start + int(np.argmin(run_values))
        recovered = run_end < n_periods
        start_position = last_high_position if last_high_position >= 0 else 0

        episodes.append(
            {
                "start": drawdowns.index[start_position],
                "trough": drawdowns.index[trough_position],
                "recovery": drawdowns.index[run_end] if recovered else pd.NaT,
                "depth": float(values[trough_position]),
                "duration": int(
                    (run_end if recovered else n_periods) - last_high_position - 1
                ),
            }
        )

    columns = ["start", "trough", "recovery", "depth", "duration"]
    if not episodes:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(episodes, columns=columns)
    result = result.sort_values("depth", ascending=True, kind="stable").reset_index(
        drop=True
    )
    return result.head(top_n)


def _validate_top_n(top_n: int) -> None:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")


def rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Calculate max drawdown independently within each rolling window."""
    _validate_returns(returns)
    if window <= 1:
        raise ValueError("window must be greater than 1")

    return returns.rolling(window=window).apply(_window_max_drawdown, raw=True)


def _window_max_drawdown(window_returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + window_returns)
    running_peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return float(np.min(wealth / running_peak - 1))


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")
    if returns.empty or returns.dropna().empty:
        raise ValueError("Returns must contain at least one valid observation")

    valid_returns = returns.dropna()
    try:
        all_finite = np.isfinite(valid_returns.to_numpy()).all()
    except TypeError as exc:
        raise ValueError("Returns must contain numeric values") from exc

    if not all_finite:
        raise ValueError("Returns must contain finite values")
    if valid_returns.lt(-1).any():
        raise ValueError("Returns must not be less than -1")
