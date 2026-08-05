"""Plotly figure builders for portfolio risk reporting."""

from __future__ import annotations

from collections.abc import Sequence

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


def build_bar_figure(
    values: pd.Series,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    tickformat: str,
) -> go.Figure:
    """Build a single-series bar chart from precomputed, pre-ordered values."""
    _validate_series(values, name="values")
    figure = go.Figure(
        go.Bar(
            x=values.index.astype(str),
            y=values,
            name=values.name or yaxis_title,
        )
    )
    figure.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=tickformat,
        showlegend=False,
    )
    return figure


def build_correlation_heatmap_figure(
    correlation_matrix: pd.DataFrame,
    *,
    title: str = "Asset correlation",
) -> go.Figure:
    """Build a diverging heatmap from a precomputed asset correlation matrix."""
    _validate_matrix(correlation_matrix, name="correlation_matrix")
    labels = correlation_matrix.columns.astype(str)
    figure = go.Figure(
        go.Heatmap(
            z=correlation_matrix.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            texttemplate="%{z:.2f}",
        )
    )
    figure.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


def build_transition_heatmap_figure(
    transition_matrix: pd.DataFrame | list[list[float]],
    *,
    labels: Sequence[str] | None = None,
    title: str = "HMM transition matrix",
) -> go.Figure:
    """Build a heatmap for an HMM (or similar) transition matrix."""
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels is None:
        labels = [f"State {idx}" for idx in range(matrix.shape[0])]
    if len(labels) != matrix.shape[0]:
        raise ValueError("labels must match transition_matrix dimension")
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(labels),
            y=list(labels),
            zmin=0,
            zmax=1,
            colorscale="Blues",
            texttemplate="%{z:.2f}",
        )
    )
    figure.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title="To state",
        yaxis_title="From state",
    )
    return figure


def build_regime_shaded_wealth_figure(
    returns: pd.Series,
    regime_labels: pd.Series,
    *,
    title: str = "Regime-shaded wealth",
    change_points: Sequence[pd.Timestamp] | None = None,
) -> go.Figure:
    """Build a cumulative-wealth chart with regime markers and optional breaks."""
    _validate_series(returns, name="returns")
    _validate_series(regime_labels, name="regime_labels")
    aligned = pd.concat(
        [returns.rename("ret"), regime_labels.rename("regime")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("returns and regime_labels have no overlapping dates")
    wealth = (1.0 + aligned["ret"]).cumprod()
    figure = go.Figure(
        go.Scatter(
            x=wealth.index,
            y=wealth,
            mode="lines",
            name="Wealth",
            line={"color": "#334155"},
        )
    )
    for label in sorted(aligned["regime"].astype(str).unique()):
        mask = aligned["regime"].astype(str) == label
        figure.add_trace(
            go.Scatter(
                x=wealth.index[mask],
                y=wealth.loc[mask],
                mode="markers",
                name=str(label),
                marker={"size": 4},
            )
        )
    if change_points:
        for stamp in change_points:
            figure.add_vline(
                x=pd.Timestamp(stamp),
                line_width=1,
                line_dash="dot",
                line_color="#94a3b8",
            )
    return _style_time_series(
        figure,
        title=title,
        yaxis_title="Growth of $1",
        tickformat=".2f",
    )


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


def _validate_matrix(values: pd.DataFrame, *, name: str) -> None:
    if not isinstance(values, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
