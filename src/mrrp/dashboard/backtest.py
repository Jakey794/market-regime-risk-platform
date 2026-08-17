"""Feature loading for the interactive Backtest Lab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrrp.data.cache import load_parquet
from mrrp.features import build_basic_regime_features, load_regime_feature_config
from mrrp.portfolio import PortfolioConfig, compute_asset_returns


def load_or_build_backtest_features(
    prices: pd.DataFrame,
    portfolio: PortfolioConfig,
    *,
    artifact_path: str | Path,
    feature_config_path: str | Path,
) -> tuple[pd.DataFrame, bool]:
    """Load persisted features or derive a session-only fallback.

    Streamlit deployments intentionally omit generated artifacts. The fallback
    derives the same trailing, leakage-safe features from the selected prices
    without writing deployment-local files.
    """
    try:
        return load_parquet(artifact_path).reindex(prices.index), False
    except (FileNotFoundError, OSError, ValueError):
        feature_config = load_regime_feature_config(feature_config_path)
        asset_tickers = list(portfolio.holdings.index)
        required_tickers = list(dict.fromkeys([*asset_tickers, portfolio.benchmark]))
        missing = [
            ticker for ticker in required_tickers if ticker not in prices.columns
        ]
        if missing:
            raise ValueError(f"Prices are missing required tickers: {missing}")

        returns = compute_asset_returns(prices.loc[:, required_tickers]).dropna(
            subset=required_tickers,
            how="any",
        )
        if returns.empty:
            raise ValueError("Prices have no complete return observations")

        features = build_basic_regime_features(
            returns.loc[:, asset_tickers],
            portfolio.holdings,
            returns[portfolio.benchmark],
            windows=feature_config.windows,
            thresholds=feature_config.thresholds,
            annualization_factor=feature_config.annualization_factor,
        )
        return features.reindex(prices.index), True
