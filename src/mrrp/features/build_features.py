"""Build reproducible raw and train-scaled regime feature artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mrrp.data.cache import load_parquet, save_parquet
from mrrp.data.validators import validate_price_frame
from mrrp.features.preprocessing import (
    clean_feature_matrix,
    date_train_test_split,
    fit_train_scaler,
    transform_features,
)
from mrrp.features.regime_features import (
    build_basic_regime_features,
    load_regime_feature_config,
)
from mrrp.features.schema import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    validate_regime_feature_columns,
)
from mrrp.portfolio.config import load_portfolio_config
from mrrp.portfolio.returns import compute_asset_returns

DEFAULT_FEATURE_CONFIG = Path("configs/regime_features.yaml")
DEFAULT_PORTFOLIO_CONFIG = Path("configs/sample_portfolio.yaml")
DEFAULT_PRICES = Path("data/processed/adjusted_close.parquet")
DEFAULT_RAW_OUTPUT = Path("data/processed/regime_features_raw.parquet")
DEFAULT_SCALED_OUTPUT = Path("data/processed/regime_features_scaled.parquet")
DEFAULT_METADATA_OUTPUT = Path("data/processed/regime_feature_metadata.json")


def build_feature_artifacts(
    *,
    feature_config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    portfolio_config_path: str | Path = DEFAULT_PORTFOLIO_CONFIG,
    prices_path: str | Path = DEFAULT_PRICES,
    raw_output_path: str | Path = DEFAULT_RAW_OUTPUT,
    scaled_output_path: str | Path = DEFAULT_SCALED_OUTPUT,
    metadata_output_path: str | Path = DEFAULT_METADATA_OUTPUT,
) -> dict[str, Any]:
    """Build, clean, train-scale, and persist regime feature artifacts."""
    feature_config = load_regime_feature_config(feature_config_path)
    portfolio_config = load_portfolio_config(portfolio_config_path)
    prices = load_parquet(prices_path)

    asset_universe = [str(ticker) for ticker in portfolio_config.holdings.index]
    required_tickers = list(
        dict.fromkeys([*asset_universe, portfolio_config.benchmark])
    )
    missing_tickers = [
        ticker for ticker in required_tickers if ticker not in prices.columns
    ]
    if missing_tickers:
        raise ValueError(f"Price data missing required tickers: {missing_tickers}")

    selected_prices = prices.loc[:, required_tickers]
    validate_price_frame(
        selected_prices,
        min_observations=feature_config.windows.long,
    )
    all_returns = compute_asset_returns(selected_prices).dropna(
        subset=required_tickers,
        how="any",
    )
    if all_returns.empty:
        raise ValueError("Price data has no complete common return dates")
    asset_returns = all_returns.loc[:, asset_universe]
    benchmark_returns = all_returns.loc[:, portfolio_config.benchmark].rename(
        "benchmark_return"
    )

    raw_features = build_basic_regime_features(
        asset_returns,
        portfolio_config.holdings,
        benchmark_returns,
        windows=feature_config.windows,
        thresholds=feature_config.thresholds,
        annualization_factor=feature_config.annualization_factor,
    )
    validate_regime_feature_columns(raw_features)

    feature_columns = list(EXPECTED_REGIME_FEATURE_COLUMNS)
    cleaned_features = clean_feature_matrix(raw_features, feature_columns)
    train_features, _ = date_train_test_split(
        cleaned_features,
        feature_config.train_end,
    )
    scaler = fit_train_scaler(train_features)
    scaled_features = transform_features(scaler, cleaned_features)

    save_parquet(cleaned_features, raw_output_path)
    save_parquet(scaled_features, scaled_output_path)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "asset_universe": asset_universe,
        "benchmark": portfolio_config.benchmark,
        "windows": asdict(feature_config.windows),
        "train_end": feature_config.train_end,
        "scaler_fit_start": _timestamp_string(train_features.index.min()),
        "scaler_fit_end": _timestamp_string(train_features.index.max()),
        "raw_shape": list(raw_features.shape),
        "cleaned_shape": list(cleaned_features.shape),
        "dropped_rows": len(raw_features) - len(cleaned_features),
        "feature_columns": feature_columns,
    }
    _save_metadata(metadata, metadata_output_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-config", default=str(DEFAULT_FEATURE_CONFIG))
    parser.add_argument("--portfolio-config", default=str(DEFAULT_PORTFOLIO_CONFIG))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--raw-out", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--scaled-out", default=str(DEFAULT_SCALED_OUTPUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_feature_artifacts(
        feature_config_path=args.feature_config,
        portfolio_config_path=args.portfolio_config,
        prices_path=args.prices,
        raw_output_path=args.raw_out,
        scaled_output_path=args.scaled_out,
        metadata_output_path=args.metadata_out,
    )
    print(
        "Built regime features: "
        f"raw={metadata['raw_shape']}, "
        f"cleaned={metadata['cleaned_shape']}, "
        f"dropped_rows={metadata['dropped_rows']}"
    )
    print(f"Raw output: {args.raw_out}")
    print(f"Scaled output: {args.scaled_out}")
    print(f"Metadata output: {args.metadata_out}")


def _save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _timestamp_string(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


if __name__ == "__main__":
    main()
