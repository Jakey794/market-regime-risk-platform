"""Portfolio configuration and weight utilities."""

from importlib import import_module

from mrrp.portfolio.config import PortfolioConfig, load_portfolio_config
from mrrp.portfolio.exposures import (
    compute_long_short_exposure,
    compute_top_n_exposure,
    compute_weight_exposure,
)
from mrrp.portfolio.metadata import compute_group_exposure, load_asset_metadata
from mrrp.portfolio.returns import (
    align_returns_and_weights,
    compute_asset_returns,
    compute_cumulative_returns,
    compute_portfolio_returns,
)
from mrrp.portfolio.weights import (
    get_portfolio_tickers,
    normalize_weights,
    validate_weights,
)

__all__ = [
    "PortfolioConfig",
    "PortfolioRiskSummary",
    "align_returns_and_weights",
    "build_correlation_table",
    "build_exposure_table",
    "build_portfolio_risk_summary",
    "build_risk_contribution_table",
    "build_summary_cards",
    "compute_asset_returns",
    "compute_cumulative_returns",
    "compute_group_exposure",
    "compute_long_short_exposure",
    "compute_portfolio_returns",
    "compute_top_n_exposure",
    "compute_weight_exposure",
    "get_portfolio_tickers",
    "load_portfolio_config",
    "load_asset_metadata",
    "normalize_weights",
    "validate_weights",
]


_SUMMARY_EXPORTS = {
    "PortfolioRiskSummary",
    "build_correlation_table",
    "build_exposure_table",
    "build_portfolio_risk_summary",
    "build_risk_contribution_table",
    "build_summary_cards",
}


def __getattr__(name: str) -> object:
    """Load summary facade exports lazily to avoid risk-package import cycles."""
    if name not in _SUMMARY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    summary_module = import_module("mrrp.portfolio.summary")
    return getattr(summary_module, name)
