"""Typed regime feature configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrrp.portfolio.returns import align_returns_and_weights
from mrrp.portfolio.weights import validate_weights
from mrrp.risk.correlation import compute_rolling_mean_pairwise_correlation
from mrrp.risk.drawdown import drawdown_series
from mrrp.risk.volatility import rolling_volatility
from mrrp.utils.config import ConfigError, load_yaml


@dataclass(frozen=True)
class RegimeFeatureWindows:
    """Trailing lookback windows (in trading days) used for regime features."""

    short: int
    medium: int
    long: int


@dataclass(frozen=True)
class RegimeFeatureThresholds:
    """Percentile and drawdown thresholds used to derive regime flag features."""

    high_vol_percentile: float
    high_corr_percentile: float
    drawdown_warning: float


@dataclass(frozen=True)
class RegimeFeatureConfig:
    """Validated, reproducible regime feature configuration."""

    windows: RegimeFeatureWindows
    annualization_factor: int
    train_end: str
    thresholds: RegimeFeatureThresholds


def load_regime_feature_config(path: str | Path) -> RegimeFeatureConfig:
    """Load and validate a regime feature configuration from YAML."""
    data = load_yaml(path)
    required_fields = {"windows", "annualization_factor", "train_end", "thresholds"}
    missing = required_fields - set(data)
    if missing:
        raise ConfigError(
            f"Regime feature config missing required fields: {sorted(missing)}"
        )

    return RegimeFeatureConfig(
        windows=_windows(data["windows"]),
        annualization_factor=_positive_int(
            data["annualization_factor"], field="annualization_factor"
        ),
        train_end=_non_empty_string(data["train_end"], field="train_end"),
        thresholds=_thresholds(data["thresholds"]),
    )


def compute_portfolio_return(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    allow_short: bool = False,
) -> pd.Series:
    """Calculate weighted daily portfolio returns from validated portfolio weights."""
    validate_weights(weights, allow_short=allow_short)
    aligned_returns, aligned_weights = align_returns_and_weights(asset_returns, weights)
    portfolio_return = aligned_returns.dot(aligned_weights)
    portfolio_return.name = "portfolio_return"
    return portfolio_return


def compute_rolling_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> pd.Series:
    """Calculate leakage-safe trailing annualized rolling volatility."""
    volatility = rolling_volatility(
        returns, window=window, periods_per_year=annualization_factor
    )
    volatility.name = "volatility"
    return volatility


def compute_mean_pairwise_corr(asset_returns: pd.DataFrame, window: int) -> pd.Series:
    """Calculate leakage-safe trailing mean pairwise asset correlation."""
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame")
    if asset_returns.index.has_duplicates:
        raise ValueError("asset_returns index contains duplicate dates")

    correlation = compute_rolling_mean_pairwise_correlation(
        asset_returns, window=window
    )
    correlation.name = "mean_pairwise_corr"
    return correlation


def compute_momentum(returns: pd.Series, window: int) -> pd.Series:
    """Calculate trailing cumulative return using only the most recent window returns."""
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a pandas Series")
    if window <= 1:
        raise ValueError("window must be greater than 1")

    momentum = returns.rolling(window=window, min_periods=window).apply(
        _window_momentum, raw=True
    )
    momentum.name = "momentum"
    return momentum


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Calculate the drawdown path using the shared leakage-safe drawdown calculation."""
    drawdown = drawdown_series(returns)
    drawdown.name = "drawdown"
    return drawdown


def trailing_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate a z-score against the trailing window ending at each date."""
    _validate_trailing_inputs(series, window, minimum_window=2)

    rolling = series.rolling(window=window, min_periods=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=1).replace(0.0, np.nan)
    zscore = (series - rolling_mean) / rolling_std
    return zscore.replace([np.inf, -np.inf], np.nan)


def trailing_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rank each value within its trailing window without using future data."""
    _validate_trailing_inputs(series, window, minimum_window=1)

    return series.rolling(window=window, min_periods=window).apply(
        _last_value_percentile_rank,
        raw=True,
    )


def build_basic_regime_features(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns: pd.Series,
    windows: RegimeFeatureWindows,
    thresholds: RegimeFeatureThresholds,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Build the basic (non-cross-sectional) leakage-safe regime feature set."""
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError("asset_returns index must be a DatetimeIndex")
    if not isinstance(benchmark_returns, pd.Series):
        raise ValueError("benchmark_returns must be a pandas Series")

    portfolio_return = compute_portfolio_return(asset_returns, weights)
    if not benchmark_returns.index.equals(portfolio_return.index):
        raise ValueError("benchmark_returns index must match asset_returns index")

    aligned_returns, _ = align_returns_and_weights(asset_returns, weights)
    portfolio_vol_21d = compute_rolling_volatility(
        portfolio_return, windows.short, annualization_factor
    )
    portfolio_vol_63d = compute_rolling_volatility(
        portfolio_return, windows.medium, annualization_factor
    )
    portfolio_vol_252d = compute_rolling_volatility(
        portfolio_return, windows.long, annualization_factor
    )
    benchmark_vol_21d = compute_rolling_volatility(
        benchmark_returns, windows.short, annualization_factor
    )
    benchmark_vol_63d = compute_rolling_volatility(
        benchmark_returns, windows.medium, annualization_factor
    )
    mean_corr_21d = compute_mean_pairwise_corr(aligned_returns, windows.short)
    mean_corr_63d = compute_mean_pairwise_corr(aligned_returns, windows.medium)
    portfolio_drawdown = compute_drawdown(portfolio_return)

    vol_percentile = trailing_percentile_rank(portfolio_vol_63d, windows.long)
    corr_percentile = trailing_percentile_rank(mean_corr_63d, windows.long)

    return pd.DataFrame(
        {
            "portfolio_return_1d": portfolio_return,
            "benchmark_return_1d": benchmark_returns,
            "portfolio_vol_21d": portfolio_vol_21d,
            "portfolio_vol_63d": portfolio_vol_63d,
            "portfolio_vol_252d": portfolio_vol_252d,
            "benchmark_vol_21d": benchmark_vol_21d,
            "benchmark_vol_63d": benchmark_vol_63d,
            "mean_corr_21d": mean_corr_21d,
            "mean_corr_63d": mean_corr_63d,
            "portfolio_drawdown": portfolio_drawdown,
            "portfolio_drawdown_z_252d": trailing_zscore(
                portfolio_drawdown, windows.long
            ),
            "portfolio_momentum_21d": compute_momentum(portfolio_return, windows.short),
            "portfolio_momentum_63d": compute_momentum(
                portfolio_return, windows.medium
            ),
            "benchmark_momentum_63d": compute_momentum(
                benchmark_returns, windows.medium
            ),
            "vol_z_252d": trailing_zscore(portfolio_vol_63d, windows.long),
            "corr_z_252d": trailing_zscore(mean_corr_63d, windows.long),
            "drawdown_flag": portfolio_drawdown.le(thresholds.drawdown_warning),
            "high_vol_flag": vol_percentile.ge(thresholds.high_vol_percentile),
            "high_corr_flag": corr_percentile.ge(thresholds.high_corr_percentile),
        },
        index=portfolio_return.index,
    )


def _window_momentum(window_returns: np.ndarray) -> float:
    return float(np.prod(1 + window_returns) - 1)


def _last_value_percentile_rank(window_values: np.ndarray) -> float:
    return float(np.mean(window_values <= window_values[-1]))


def _validate_trailing_inputs(
    series: pd.Series,
    window: int,
    *,
    minimum_window: int,
) -> None:
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series")
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise ValueError("window must be an integer")
    if window < minimum_window:
        qualifier = "greater than 1" if minimum_window == 2 else "positive"
        raise ValueError(f"window must be {qualifier}")

    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("series must contain numeric values") from exc
    if not (np.isfinite(values) | pd.isna(values)).all():
        raise ValueError("series must contain finite values or NaN")


def _windows(value: Any) -> RegimeFeatureWindows:
    if not isinstance(value, dict):
        raise ConfigError("'windows' must be a mapping with 'short', 'medium', 'long'")
    required = {"short", "medium", "long"}
    missing = required - set(value)
    if missing:
        raise ConfigError(f"'windows' missing required fields: {sorted(missing)}")

    return RegimeFeatureWindows(
        short=_positive_int(value["short"], field="windows.short"),
        medium=_positive_int(value["medium"], field="windows.medium"),
        long=_positive_int(value["long"], field="windows.long"),
    )


def _thresholds(value: Any) -> RegimeFeatureThresholds:
    if not isinstance(value, dict):
        raise ConfigError(
            "'thresholds' must be a mapping with 'high_vol_percentile', "
            "'high_corr_percentile', 'drawdown_warning'"
        )
    required = {"high_vol_percentile", "high_corr_percentile", "drawdown_warning"}
    missing = required - set(value)
    if missing:
        raise ConfigError(f"'thresholds' missing required fields: {sorted(missing)}")

    drawdown_warning = value["drawdown_warning"]
    if isinstance(drawdown_warning, bool) or not isinstance(drawdown_warning, Real):
        raise ConfigError("'thresholds.drawdown_warning' must be numeric")
    drawdown_warning = float(drawdown_warning)
    if drawdown_warning >= 0:
        raise ConfigError("'thresholds.drawdown_warning' must be negative")

    return RegimeFeatureThresholds(
        high_vol_percentile=_unit_interval(
            value["high_vol_percentile"], field="thresholds.high_vol_percentile"
        ),
        high_corr_percentile=_unit_interval(
            value["high_corr_percentile"], field="thresholds.high_corr_percentile"
        ),
        drawdown_warning=drawdown_warning,
    )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"'{field}' must be a positive integer")
    return value


def _unit_interval(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"'{field}' must be numeric")
    numeric = float(value)
    if not 0 < numeric < 1:
        raise ConfigError(f"'{field}' must be between 0 and 1")
    return numeric


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string")
    return value.strip()
