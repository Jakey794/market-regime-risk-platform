"""Asset universe helpers."""

from __future__ import annotations

from mrrp.utils.config import UniverseConfig


def flatten_asset_universe(config: UniverseConfig) -> list[str]:
    """Flatten grouped asset universe into a sorted unique ticker list.

    The benchmark is always included even if it is not listed under assets.

    Args:
        config: Validated universe configuration.

    Returns:
        Sorted list of unique ticker symbols.
    """
    tickers: set[str] = set()

    for group_tickers in config.assets.values():
        tickers.update(ticker.strip().upper() for ticker in group_tickers)

    tickers.add(config.benchmark.strip().upper())

    return sorted(tickers)


def get_asset_groups(config: UniverseConfig) -> dict[str, list[str]]:
    """Return a defensive copy of asset groups."""
    return {group: list(tickers) for group, tickers in config.assets.items()}


def benchmark_exists_in_universe(config: UniverseConfig) -> bool:
    """Return whether the configured benchmark is present in the flattened universe."""
    return config.benchmark in flatten_asset_universe(config)
