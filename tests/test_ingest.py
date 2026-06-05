from __future__ import annotations

import pandas as pd
import pytest

from mrrp.data.ingest import download_prices
from mrrp.data.universe import flatten_asset_universe_preserve_order
from mrrp.utils.config import UniverseConfig


def _yfinance_close_frame(index: list[str], data: dict[str, list[float | None]]):
    columns = pd.MultiIndex.from_product([["Close"], data.keys()])
    return pd.DataFrame(
        [[values[row] for values in data.values()] for row in range(len(index))],
        index=pd.to_datetime(index),
        columns=columns,
    )


def test_download_prices_rejects_duplicate_dates(monkeypatch) -> None:
    raw_prices = _yfinance_close_frame(
        ["2024-01-02", "2024-01-02"],
        {"SPY": [470.0, 471.0]},
    )

    monkeypatch.setattr("mrrp.data.ingest.yf.download", lambda *_, **__: raw_prices)

    with pytest.raises(ValueError, match="duplicate dates"):
        download_prices(["SPY"], start_date="2024-01-01")


def test_download_prices_rejects_all_null_ticker(monkeypatch) -> None:
    raw_prices = _yfinance_close_frame(
        ["2024-01-02", "2024-01-03"],
        {"SPY": [470.0, 471.0], "QQQ": [None, None]},
    )

    monkeypatch.setattr("mrrp.data.ingest.yf.download", lambda *_, **__: raw_prices)

    with pytest.raises(ValueError, match="only null prices"):
        download_prices(["SPY", "QQQ"], start_date="2024-01-01")


def test_flatten_asset_universe_preserves_order_and_includes_benchmark() -> None:
    config = UniverseConfig(
        assets={
            "group_a": ["SPY", "QQQ"],
            "group_b": ["SPY", "EEM"],
        },
        benchmark="XIU.TO",
        start_date="2005-01-01",
        end_date=None,
    )

    assert flatten_asset_universe_preserve_order(config) == [
        "SPY",
        "QQQ",
        "EEM",
        "XIU.TO",
    ]
