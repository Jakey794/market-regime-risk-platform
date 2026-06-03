from __future__ import annotations

from mrrp.data.universe import (
    benchmark_exists_in_universe,
    flatten_asset_universe,
    get_asset_groups,
)
from mrrp.utils.config import UniverseConfig, load_universe_config


def test_flatten_asset_universe_includes_benchmark() -> None:
    config = load_universe_config("configs/default_universe.yaml")

    tickers = flatten_asset_universe(config)

    assert "SPY" in tickers
    assert "QQQ" in tickers
    assert "XIU.TO" in tickers


def test_flatten_asset_universe_removes_duplicates() -> None:
    config = UniverseConfig(
        assets={
            "group_a": ["SPY", "QQQ"],
            "group_b": ["SPY", "EEM"],
        },
        benchmark="SPY",
        start_date="2005-01-01",
        end_date=None,
    )

    tickers = flatten_asset_universe(config)

    assert tickers.count("SPY") == 1
    assert tickers == ["EEM", "QQQ", "SPY"]


def test_benchmark_exists_in_universe() -> None:
    config = load_universe_config("configs/default_universe.yaml")

    assert benchmark_exists_in_universe(config)


def test_get_asset_groups_returns_copy() -> None:
    config = load_universe_config("configs/default_universe.yaml")

    groups = get_asset_groups(config)
    groups["us_equity"].append("FAKE")

    assert "FAKE" not in config.assets["us_equity"]
