from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from mrrp.dashboard import (
    InsufficientObservationsError,
    load_dashboard_data,
    load_dashboard_dataset,
)


def _write_portfolio(path) -> None:
    path.write_text(
        """
name: test_portfolio
benchmark: SPY
currency: CAD
allow_short: false
holdings:
  SPY: 0.6
  QQQ: 0.4
""".strip(),
        encoding="utf-8",
    )


def test_load_dashboard_data_uses_canonical_portfolio_config(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    prices.to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    actual_prices, portfolio = load_dashboard_data(prices_path, portfolio_path)

    pd.testing.assert_frame_equal(actual_prices, prices, check_freq=False)
    assert portfolio.benchmark == "SPY"
    assert portfolio.holdings.to_dict() == pytest.approx({"SPY": 0.6, "QQQ": 0.4})


def test_load_dashboard_data_applies_dates_and_benchmark_override(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "QQQ": [200.0, 202.0, 204.0],
            "XIU.TO": [30.0, 30.5, 31.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    prices.to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    selected, portfolio = load_dashboard_data(
        prices_path,
        portfolio_path,
        start_date=date(2024, 1, 2),
        end_date=pd.Timestamp("2024-01-03"),
        benchmark=" xiu.to ",
    )

    assert selected.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert portfolio.benchmark == "XIU.TO"


def test_load_dashboard_data_rejects_missing_required_ticker(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    ).to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    with pytest.raises(ValueError, match="missing required tickers.*QQQ"):
        load_dashboard_data(prices_path, portfolio_path)


def test_load_dashboard_data_rejects_unknown_benchmark(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=pd.date_range("2024-01-01", periods=2),
    ).to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    with pytest.raises(ValueError, match="missing required tickers.*IWM"):
        load_dashboard_data(prices_path, portfolio_path, benchmark="IWM")


def test_load_dashboard_data_rejects_invalid_or_empty_date_range(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=pd.date_range("2024-01-01", periods=2),
    ).to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    with pytest.raises(ValueError, match="must not be after"):
        load_dashboard_data(
            prices_path,
            portfolio_path,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 1),
        )

    with pytest.raises(ValueError, match="contains no price observations"):
        load_dashboard_data(
            prices_path,
            portfolio_path,
            start_date=date(2025, 1, 1),
        )


def test_load_dashboard_dataset_uses_existing_return_functions(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    pd.DataFrame(
        {
            "SPY": [100.0, 102.0, 101.0],
            "QQQ": [200.0, 204.0, 208.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    ).to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    result = load_dashboard_dataset(
        str(prices_path),
        str(portfolio_path),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        benchmark="SPY",
        minimum_complete_returns=2,
    )

    assert result.asset_returns.columns.tolist() == ["SPY", "QQQ"]
    assert result.portfolio_returns.tolist() == pytest.approx([0.02, 0.0019607843])
    assert result.benchmark_returns.tolist() == pytest.approx([0.02, -0.0098039216])
    assert result.complete_observations == 2


def test_load_dashboard_dataset_rejects_short_sample(tmp_path) -> None:
    prices_path = tmp_path / "prices.parquet"
    portfolio_path = tmp_path / "portfolio.yaml"
    pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=pd.date_range("2024-01-01", periods=2),
    ).to_parquet(prices_path)
    _write_portfolio(portfolio_path)

    with pytest.raises(InsufficientObservationsError, match="at least 63.*found 1"):
        load_dashboard_dataset(
            str(prices_path),
            str(portfolio_path),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            benchmark="SPY",
        )
