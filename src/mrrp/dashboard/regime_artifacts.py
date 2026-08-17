"""Session-only regime-feature artifacts for the public dashboard demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mrrp.data.cache import load_parquet
from mrrp.features import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    build_basic_regime_features,
    clean_feature_matrix,
    date_train_test_split,
    fit_train_scaler,
    load_regime_feature_config,
    transform_features,
)
from mrrp.portfolio import PortfolioConfig, compute_asset_returns


def load_or_build_regime_artifacts(
    prices: pd.DataFrame,
    portfolio: PortfolioConfig,
    *,
    raw_path: str | Path,
    scaled_path: str | Path,
    metadata_path: str | Path,
    feature_config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    """Load artifacts or derive their equivalent without writing to disk.

    Generated artifacts are intentionally excluded from the repository. When a
    public Streamlit deployment has only the tracked synthetic prices, this
    preserves the same trailing features and train-only scaling contract.
    """
    try:
        raw = load_parquet(raw_path)
        scaled = load_parquet(scaled_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Feature metadata must be a JSON object")
        return raw, scaled, metadata, False
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
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

        raw_features = build_basic_regime_features(
            returns.loc[:, asset_tickers],
            portfolio.holdings,
            returns[portfolio.benchmark],
            windows=feature_config.windows,
            thresholds=feature_config.thresholds,
            annualization_factor=feature_config.annualization_factor,
        )
        raw = clean_feature_matrix(
            raw_features,
            list(EXPECTED_REGIME_FEATURE_COLUMNS),
        )
        train, _ = date_train_test_split(raw, feature_config.train_end)
        scaled = transform_features(fit_train_scaler(train), raw)
        metadata = {
            "asset_universe": asset_tickers,
            "benchmark": portfolio.benchmark,
            "feature_columns": list(EXPECTED_REGIME_FEATURE_COLUMNS),
            "raw_shape": list(raw_features.shape),
            "cleaned_shape": list(raw.shape),
            "dropped_rows": len(raw_features) - len(raw),
            "train_end": feature_config.train_end,
            "scaler_fit_start": train.index.min().isoformat(),
            "scaler_fit_end": train.index.max().isoformat(),
        }
        return raw, scaled, metadata, True
