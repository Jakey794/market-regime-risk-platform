"""Risk-aware allocation rules with intentional signal shifting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

RuleName = Literal[
    "static_benchmark",
    "volatility_target",
    "drawdown_guardrail",
    "high_vol_derisk",
    "high_corr_derisk",
    "combined_risk_aware",
]


@dataclass(frozen=True)
class RuleConfig:
    """Configuration for a single allocation rule."""

    name: RuleName
    base_weights: pd.Series
    defensive_weights: pd.Series
    vol_target: float = 0.10
    vol_column: str = "portfolio_vol_63d"
    corr_column: str = "mean_corr_63d"
    drawdown_column: str = "portfolio_drawdown"
    high_vol_threshold: float = 0.20
    high_corr_threshold: float = 0.60
    drawdown_limit: float = -0.10
    min_risk_scalar: float = 0.25


def validate_weights(weights: pd.Series, *, tol: float = 1e-8) -> pd.Series:
    """Validate finite weights that sum to approximately one."""
    if not isinstance(weights, pd.Series) or weights.empty:
        raise ValueError("weights must be a non-empty Series")
    if weights.index.has_duplicates:
        raise ValueError("weight tickers must be unique")
    values = weights.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("weights must be finite")
    total = float(values.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"weights must sum to 1.0; got {total}")
    return values


def compute_signal_scalar(
    features: pd.DataFrame,
    config: RuleConfig,
) -> pd.Series:
    """Compute a risk scalar in (min_risk_scalar, 1] from available features.

    The scalar is computed from information on date t and is intended to be
    shifted before execution so same-close trading cannot occur.
    """
    if config.name == "static_benchmark":
        return pd.Series(1.0, index=features.index, name="risk_scalar")

    scalar = pd.Series(1.0, index=features.index, name="risk_scalar")
    if config.name in {"volatility_target", "combined_risk_aware"}:
        if config.vol_column not in features.columns:
            raise ValueError(f"missing vol column {config.vol_column}")
        vol = features[config.vol_column].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_scalar = (config.vol_target / vol).clip(upper=1.0)
        scalar = scalar.combine(vol_scalar, min)

    if config.name in {"drawdown_guardrail", "combined_risk_aware"}:
        if config.drawdown_column not in features.columns:
            raise ValueError(f"missing drawdown column {config.drawdown_column}")
        dd = features[config.drawdown_column].astype(float)
        dd_scalar = pd.Series(1.0, index=features.index)
        dd_scalar = dd_scalar.where(dd > config.drawdown_limit, config.min_risk_scalar)
        scalar = scalar.combine(dd_scalar, min)

    if config.name in {"high_vol_derisk", "combined_risk_aware"}:
        if config.vol_column not in features.columns:
            raise ValueError(f"missing vol column {config.vol_column}")
        high_vol = (
            features[config.vol_column].astype(float) >= config.high_vol_threshold
        )
        vol_derisk = pd.Series(1.0, index=features.index)
        vol_derisk = vol_derisk.where(~high_vol, config.min_risk_scalar)
        scalar = scalar.combine(vol_derisk, min)

    if config.name in {"high_corr_derisk", "combined_risk_aware"}:
        if config.corr_column not in features.columns:
            raise ValueError(f"missing corr column {config.corr_column}")
        high_corr = (
            features[config.corr_column].astype(float) >= config.high_corr_threshold
        )
        corr_derisk = pd.Series(1.0, index=features.index)
        corr_derisk = corr_derisk.where(~high_corr, config.min_risk_scalar)
        scalar = scalar.combine(corr_derisk, min)

    return scalar.clip(lower=config.min_risk_scalar, upper=1.0)


def target_weights_from_scalar(
    scalar: float,
    config: RuleConfig,
) -> pd.Series:
    """Blend base and defensive weights using a risk scalar in [0, 1]."""
    base = validate_weights(config.base_weights)
    defensive = validate_weights(config.defensive_weights)
    aligned = pd.concat([base.rename("base"), defensive.rename("def")], axis=1).fillna(
        0.0
    )
    # Re-normalize after alignment in case tickers differ.
    aligned["base"] = aligned["base"] / aligned["base"].sum()
    aligned["def"] = aligned["def"] / aligned["def"].sum()
    mixed = float(scalar) * aligned["base"] + (1.0 - float(scalar)) * aligned["def"]
    return validate_weights(mixed)
