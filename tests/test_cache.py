from __future__ import annotations

import pandas as pd
import pytest

from mrrp.data.cache import cache_exists, load_parquet, save_parquet


def test_parquet_save_load_roundtrip(tmp_path) -> None:
    expected = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=pd.date_range("2024-01-01", periods=2, name="date"),
    )
    path = tmp_path / "nested" / "prices.parquet"

    save_parquet(expected, path)
    actual = load_parquet(path)

    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_cache_exists_returns_true_and_false(tmp_path) -> None:
    path = tmp_path / "prices.parquet"

    assert not cache_exists(path)

    save_parquet(pd.DataFrame({"SPY": [100.0]}), str(path))

    assert cache_exists(str(path))


def test_load_parquet_raises_for_missing_file(tmp_path) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_parquet(path)
