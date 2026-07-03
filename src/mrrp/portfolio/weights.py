"""Portfolio weight validation and normalization."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from mrrp.portfolio.config import PortfolioConfig


def validate_weights(
    weights: pd.Series,
    allow_short: bool = False,
    tolerance: float = 1e-6,
) -> None:
    """Validate portfolio weights and raise ``ValueError`` when invalid."""
    _validate_weight_series(weights)
    if not isinstance(allow_short, bool):
        raise ValueError("allow_short must be a boolean")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be a non-negative finite number")
    if not allow_short and weights.lt(0).any():
        raise ValueError("Negative weights require allow_short=True")

    weight_sum = float(weights.sum())
    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"Portfolio weights must sum to 1.0; got {weight_sum:.10f}")


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Scale finite numeric weights so that they sum to one."""
    _validate_weight_series(weights)
    weight_sum = float(weights.sum())
    if np.isclose(weight_sum, 0.0, rtol=0.0, atol=np.finfo(float).eps):
        raise ValueError("Weights cannot be normalized when their sum is zero")
    return weights.astype(float) / weight_sum


def get_portfolio_tickers(config: PortfolioConfig) -> list[str]:
    """Return portfolio tickers in configured order."""
    from mrrp.portfolio.config import PortfolioConfig as PortfolioConfigType

    if not isinstance(config, PortfolioConfigType):
        raise ValueError("config must be a PortfolioConfig")
    return [str(ticker) for ticker in config.holdings.index]


def _validate_weight_series(weights: pd.Series) -> None:
    if not isinstance(weights, pd.Series):
        raise ValueError("Weights must be a pandas Series")
    if weights.empty:
        raise ValueError("Holdings cannot be empty")
    if weights.index.has_duplicates:
        raise ValueError("Holding tickers must be unique")
    if any(
        not isinstance(ticker, str) or not ticker.strip() for ticker in weights.index
    ):
        raise ValueError("Holding tickers must be non-empty strings")
    if is_bool_dtype(weights.dtype) or not is_numeric_dtype(weights.dtype):
        raise ValueError("Weights must be numeric")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Weights must be finite")
