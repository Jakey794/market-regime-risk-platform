from __future__ import annotations

import pandas as pd
import pytest

from mrrp.data.validators import (
    report_missing_data,
    validate_min_history,
    validate_price_frame,
)


def _valid_prices(periods: int = 300) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "SPY": [100.0 + value for value in range(periods)],
            "QQQ": [200.0 + value for value in range(periods)],
        },
        index=index,
    )


def test_valid_price_frame_passes() -> None:
    validate_price_frame(_valid_prices())


def test_duplicate_dates_fail() -> None:
    prices = _valid_prices()
    prices.index = prices.index[:-1].append(prices.index[-2:-1])

    with pytest.raises(ValueError, match="duplicate dates"):
        validate_price_frame(prices)


def test_non_datetime_index_fails() -> None:
    prices = _valid_prices()
    prices.index = range(len(prices))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_price_frame(prices)


def test_non_monotonic_dates_fail() -> None:
    prices = _valid_prices()
    prices = prices.iloc[::-1]

    with pytest.raises(ValueError, match="monotonic increasing"):
        validate_price_frame(prices)


def test_duplicate_columns_fail() -> None:
    prices = _valid_prices()
    prices.columns = ["SPY", "SPY"]

    with pytest.raises(ValueError, match="duplicate columns"):
        validate_price_frame(prices)


@pytest.mark.parametrize("invalid_price", [-1.0, 0.0])
def test_non_positive_prices_fail(invalid_price: float) -> None:
    prices = _valid_prices()
    prices.iloc[0, 0] = invalid_price

    with pytest.raises(ValueError, match="Prices must be positive"):
        validate_price_frame(prices)


def test_all_null_ticker_fails() -> None:
    prices = _valid_prices()
    prices["EMPTY"] = None

    with pytest.raises(ValueError, match="only null prices"):
        validate_price_frame(prices)


def test_insufficient_history_fails() -> None:
    prices = _valid_prices(periods=251)

    with pytest.raises(ValueError, match="fewer than 252 observations"):
        validate_min_history(prices)


def test_excessive_missing_data_fails() -> None:
    prices = _valid_prices()
    prices.loc[prices.index[:76], "SPY"] = None

    with pytest.raises(ValueError, match="maximum missing-data fraction"):
        validate_price_frame(prices, min_observations=1)


def test_missing_data_report_has_expected_columns() -> None:
    prices = _valid_prices(periods=4)
    prices.loc[prices.index[0], "SPY"] = None
    prices["EMPTY"] = None

    report = report_missing_data(prices)

    assert list(report.columns) == [
        "ticker",
        "missing_count",
        "missing_fraction",
        "first_valid_date",
        "last_valid_date",
        "observation_count",
    ]
    spy = report.loc[report["ticker"] == "SPY"].iloc[0]
    assert spy["missing_count"] == 1
    assert spy["missing_fraction"] == 0.25
    assert spy["first_valid_date"] == prices.index[1]
    assert spy["last_valid_date"] == prices.index[-1]
    assert spy["observation_count"] == 3


def test_exact_maximum_missing_fraction_passes() -> None:
    prices = _valid_prices()
    prices.loc[prices.index[:75], "SPY"] = None

    validate_price_frame(prices, min_observations=1)
