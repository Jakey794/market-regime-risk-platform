"""Download reproducible ETF adjusted close data."""

from __future__ import annotations

import argparse
from pathlib import Path

from mrrp.data.ingest import download_prices, save_prices
from mrrp.data.universe import flatten_asset_universe_preserve_order
from mrrp.utils.config import load_universe_config

RAW_PRICES_PATH = Path("data/raw/prices.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default_universe.yaml",
        help="Path to universe YAML config.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/adjusted_close.parquet",
        help="Path for processed adjusted close parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_universe_config(args.config)
    tickers = flatten_asset_universe_preserve_order(config)

    prices = download_prices(
        tickers=tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    processed_path = Path(args.out)
    save_prices(prices, raw_path=RAW_PRICES_PATH, processed_path=processed_path)

    start = prices.index.min().date().isoformat()
    end = prices.index.max().date().isoformat()

    print(f"Downloaded prices shape: {prices.shape}")
    print(f"Date range: {start} to {end}")
    print(f"Raw output: {RAW_PRICES_PATH}")
    print(f"Processed output: {processed_path}")


if __name__ == "__main__":
    main()
