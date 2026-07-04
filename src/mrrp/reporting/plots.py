"""Plotly figure builders for portfolio risk reporting."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_return_comparison_figure(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    benchmark_name: str,
) -> go.Figure:
    """Build a cumulative-return comparison from precomputed return paths."""
    _validate_series(portfolio_returns, name="portfolio_returns")
    _validate_series(benchmark_returns, name="benchmark_returns")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio_returns.index,
            y=portfolio_returns,
            mode="lines",
            name="Portfolio",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark_returns.index,
            y=benchmark_returns,
            mode="lines",
            name=benchmark_name,
        )
    )
    return _style_time_series(
        figure,
        title="Cumulative return vs benchmark",
        yaxis_title="Cumulative return",
        tickformat=".0%",
    )


def build_time_series_figure(
    values: pd.Series,
    *,
    title: str,
    yaxis_title: str,
    tickformat: str,
) -> go.Figure:
    """Build a single-series time-series figure from precomputed values."""
    _validate_series(values, name="values")
    figure = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            mode="lines",
            name=values.name or yaxis_title,
        )
    )
    return _style_time_series(
        figure,
        title=title,
        yaxis_title=yaxis_title,
        tickformat=tickformat,
    )


def build_histogram_figure(
    returns: pd.Series,
    *,
    title: str,
    xaxis_title: str,
    tickformat: str,
) -> go.Figure:
    """Build a return-distribution histogram from precomputed periodic returns."""
    _validate_series(returns, name="returns")
    figure = go.Figure(go.Histogram(x=returns, name=xaxis_title))
    figure.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title=xaxis_title,
        xaxis_tickformat=tickformat,
        yaxis_title="Frequency",
        showlegend=False,
    )
    return figure


def build_weights_figure(weights: pd.Series) -> go.Figure:
    """Build an asset-weight bar chart from validated portfolio weights."""
    _validate_series(weights, name="weights")
    ordered = weights.sort_values(ascending=False, kind="stable")
    figure = go.Figure(
        go.Bar(
            x=ordered.index.astype(str),
            y=ordered,
            name="Weight",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Asset weights",
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title="Asset",
        yaxis_title="Portfolio weight",
        yaxis_tickformat=".0%",
    )
    return figure


def _style_time_series(
    figure: go.Figure,
    *,
    title: str,
    yaxis_title: str,
    tickformat: str,
) -> go.Figure:
    figure.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        yaxis_tickformat=tickformat,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure


def _validate_series(values: pd.Series, *, name: str) -> None:
    if not isinstance(values, pd.Series):
        raise ValueError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
