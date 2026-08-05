"""Expanding-window walk-forward helpers for regime-aware backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from mrrp.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from mrrp.models.base import chronological_split
from mrrp.models.result import RegimeModelResult


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window train/test fold."""

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_folds(
    index: pd.DatetimeIndex,
    *,
    min_train_obs: int,
    test_horizon: int,
) -> list[WalkForwardFold]:
    """Build expanding train windows with fixed-length test horizons."""
    if min_train_obs < 2:
        raise ValueError("min_train_obs must be >= 2")
    if test_horizon < 1:
        raise ValueError("test_horizon must be >= 1")
    folds: list[WalkForwardFold] = []
    start = min_train_obs - 1
    while start + test_horizon < len(index):
        train_end = pd.Timestamp(index[start])
        test_start = pd.Timestamp(index[start + 1])
        test_end = pd.Timestamp(index[start + test_horizon])
        folds.append(
            WalkForwardFold(
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        start += test_horizon
    return folds


def walkforward_regime_labels(
    features: pd.DataFrame,
    *,
    train_end: str,
    fit_transform: Callable[[pd.DataFrame, pd.DataFrame], RegimeModelResult],
) -> pd.Series:
    """Fit a regime model on train data only and label the full sample.

    ``fit_transform(train, full)`` must fit exclusively on ``train`` and then
    infer labels for ``full``. Future folds must not be used to select states.
    """
    split = chronological_split(features, train_end=train_end)
    result = fit_transform(split.train, split.full)
    return result.labeled_states()


def run_walkforward_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_ticker: str,
    config: BacktestConfig,
    *,
    config_name: str = "walkforward",
) -> BacktestResult:
    """Run the standard engine; walk-forward model refits are caller-supplied.

    The engine itself never peeks at future prices for earlier weights. Callers
    that refresh regime features/models should regenerate ``features`` using only
    information available through each fold's train_end before invoking this.
    """
    return run_backtest(
        prices,
        features,
        benchmark_ticker,
        config,
        config_name=config_name,
    )
