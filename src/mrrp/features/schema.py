"""Regime feature schema and column contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

EXPECTED_REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "portfolio_return_1d",
    "benchmark_return_1d",
    "portfolio_vol_21d",
    "portfolio_vol_63d",
    "portfolio_vol_252d",
    "benchmark_vol_21d",
    "benchmark_vol_63d",
    "mean_corr_21d",
    "mean_corr_63d",
    "portfolio_drawdown",
    "portfolio_drawdown_z_252d",
    "portfolio_momentum_21d",
    "portfolio_momentum_63d",
    "benchmark_momentum_63d",
    "vol_z_252d",
    "corr_z_252d",
    "drawdown_flag",
    "high_vol_flag",
    "high_corr_flag",
)


@dataclass(frozen=True)
class RegimeFeatureSet:
    """Container for raw and optionally scaled regime feature data plus metadata."""

    raw_features: pd.DataFrame
    scaled_features: pd.DataFrame | None
    metadata: dict[str, Any]


def validate_regime_feature_columns(features: pd.DataFrame) -> None:
    """Validate that a feature DataFrame contains exactly the expected regime feature columns."""
    missing = [c for c in EXPECTED_REGIME_FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Feature data missing required columns: {missing}")

    extra = [c for c in features.columns if c not in EXPECTED_REGIME_FEATURE_COLUMNS]
    if extra:
        raise ValueError(f"Feature data contains unexpected columns: {extra}")
