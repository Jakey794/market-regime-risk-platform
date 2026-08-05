"""Shared memo/regime context helpers used by reporting and thin dashboard pages."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from mrrp.risk.beta import compute_portfolio_beta
from mrrp.risk.concentration import classify_concentration_risk
from mrrp.risk.correlation import (
    classify_correlation_regime,
    compute_rolling_mean_pairwise_correlation,
)
from mrrp.risk.drawdown import current_drawdown
from mrrp.risk.risk_contribution import build_risk_contribution_table
from mrrp.risk.tail import historical_cvar, historical_var


def classify_volatility_regime_from_features(features: pd.DataFrame) -> str:
    """Classify the latest volatility regime from trailing feature flags/z-scores."""
    if features.empty:
        return "unavailable"
    latest = features.iloc[-1]
    if "high_vol_flag" in features.columns and bool(latest["high_vol_flag"]):
        return "High volatility"
    if "vol_z_252d" in features.columns:
        z_score = float(latest["vol_z_252d"])
        if z_score >= 1.0:
            return "Elevated volatility"
        if z_score <= -0.5:
            return "Low volatility"
        return "Normal volatility"
    if "portfolio_vol_63d" in features.columns:
        history = features["portfolio_vol_63d"].dropna()
        if history.empty:
            return "unavailable"
        latest_vol = float(history.iloc[-1])
        p75 = float(history.quantile(0.75))
        p33 = float(history.quantile(0.33))
        if latest_vol >= p75:
            return "High volatility"
        if latest_vol <= p33:
            return "Low volatility"
        return "Normal volatility"
    return "unavailable"


def classify_correlation_regime_from_features(features: pd.DataFrame) -> str:
    """Classify the latest correlation regime from trailing feature columns."""
    if features.empty:
        return "unavailable"
    latest = features.iloc[-1]
    if "high_corr_flag" in features.columns and bool(latest["high_corr_flag"]):
        return "High correlation"
    if "mean_corr_63d" in features.columns:
        series = features["mean_corr_63d"].dropna()
        if series.empty:
            return "unavailable"
        return classify_correlation_regime(series)
    return "unavailable"


def classify_correlation_regime_from_returns(
    asset_returns: pd.DataFrame,
    *,
    window: int = 63,
) -> str:
    """Classify correlation regime from asset returns when features are absent."""
    rolling = compute_rolling_mean_pairwise_correlation(asset_returns, window)
    if rolling.dropna().empty:
        return "unavailable"
    return classify_correlation_regime(rolling)


def build_memo_summary_cards(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: pd.Series,
    asset_returns: pd.DataFrame,
) -> dict[str, Any]:
    """Build memo summary fields from portfolio risk primitives."""
    contrib = build_risk_contribution_table(asset_returns, weights)
    ordered = contrib.sort_values(
        "risk_contribution_pct", ascending=False, kind="stable"
    ).head(5)
    return {
        "drawdown": float(current_drawdown(portfolio_returns)),
        "var_95": float(historical_var(portfolio_returns, confidence=0.95)),
        "cvar_95": float(historical_cvar(portfolio_returns, confidence=0.95)),
        "beta": float(compute_portfolio_beta(portfolio_returns, benchmark_returns)),
        "concentration": classify_concentration_risk(weights),
        "top_risk_contributors": {
            str(row.ticker): float(row.risk_contribution_pct)
            for row in ordered.itertuples(index=False)
        },
    }


def regime_agreement_text(labels: Mapping[str, str]) -> str:
    """Summarise agreement across model labels without implying forecasts."""
    if not labels:
        return "unavailable"
    unique = sorted(set(str(value) for value in labels.values()))
    joined = ", ".join(f"{name}={label}" for name, label in labels.items())
    if len(unique) == 1:
        return f"Models currently agree on '{unique[0]}' ({joined})."
    return f"Models currently disagree ({joined})."
