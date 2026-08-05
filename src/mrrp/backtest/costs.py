"""Transaction cost helpers for no-look-ahead backtests."""

from __future__ import annotations

import pandas as pd


def turnover(previous_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way turnover as half the L1 distance between weight vectors."""
    aligned = pd.concat(
        [
            previous_weights.rename("prev"),
            new_weights.rename("new"),
        ],
        axis=1,
    ).fillna(0.0)
    return float(0.5 * (aligned["new"] - aligned["prev"]).abs().sum())


def transaction_cost(
    previous_weights: pd.Series,
    new_weights: pd.Series,
    cost_bps: float,
) -> float:
    """Proportional transaction cost from turnover and basis-point fee."""
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    return turnover(previous_weights, new_weights) * (float(cost_bps) / 10_000.0)
