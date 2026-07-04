from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from mrrp.reporting import (
    build_histogram_figure,
    build_return_comparison_figure,
    build_time_series_figure,
    build_weights_figure,
)


def test_return_comparison_figure_contains_portfolio_and_benchmark() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    figure = build_return_comparison_figure(
        pd.Series([0.01, 0.02], index=index),
        pd.Series([0.00, 0.01], index=index),
        benchmark_name="SPY",
    )

    assert [trace.name for trace in figure.data] == ["Portfolio", "SPY"]
    assert figure.layout.yaxis.tickformat == ".0%"


def test_time_series_figure_preserves_values() -> None:
    values = pd.Series(
        [0.10, 0.12],
        index=pd.date_range("2024-01-01", periods=2),
        name="Rolling volatility",
    )

    figure = build_time_series_figure(
        values,
        title="Rolling volatility",
        yaxis_title="Volatility",
        tickformat=".0%",
    )

    assert list(figure.data[0].y) == [0.10, 0.12]
    assert figure.layout.title.text == "Rolling volatility"


def test_weights_figure_orders_largest_weight_first() -> None:
    figure = build_weights_figure(pd.Series({"QQQ": 0.3, "SPY": 0.7}))

    assert list(figure.data[0].x) == ["SPY", "QQQ"]
    assert list(figure.data[0].y) == [0.7, 0.3]


def test_histogram_figure_contains_return_values() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.01])

    figure = build_histogram_figure(
        returns,
        title="Daily return distribution",
        xaxis_title="Daily return",
        tickformat=".1%",
    )

    assert isinstance(figure.data[0], go.Histogram)
    assert list(figure.data[0].x) == list(returns)
    assert figure.layout.title.text == "Daily return distribution"
    assert figure.layout.xaxis.tickformat == ".1%"


def test_histogram_figure_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_histogram_figure(
            pd.Series(dtype=float),
            title="Daily return distribution",
            xaxis_title="Daily return",
            tickformat=".1%",
        )
