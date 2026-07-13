"""Regime feature diagnostics dashboard page."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mrrp.dashboard.components import (
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.formatting import format_decimal
from mrrp.data.cache import load_parquet
from mrrp.data.validators import report_missing_data
from mrrp.reporting import build_correlation_heatmap_figure


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "regime_features_raw.parquet"
SCALED_FEATURES_PATH = (
    ROOT_DIR / "data" / "processed" / "regime_features_scaled.parquet"
)
METADATA_PATH = ROOT_DIR / "data" / "processed" / "regime_feature_metadata.json"
RUN_MAKE_FEATURES_MESSAGE = "Run `make features` to build the regime feature artifacts."
DEFAULT_FEATURE_COUNT = 4


@st.cache_data(show_spinner=False)
def load_feature_data(
    raw_path: str,
    scaled_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached raw and train-scaled regime feature artifacts."""
    return load_parquet(raw_path), load_parquet(scaled_path)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict[str, Any]:
    """Load the regime feature build metadata as a plain dictionary."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature metadata file does not exist: {metadata_path}"
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def filter_by_date(
    features: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Return the rows of a feature DataFrame within an inclusive date range."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    return features.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]


def build_feature_lines_figure(features: pd.DataFrame) -> go.Figure:
    """Build a multi-line time-series figure for the selected feature columns."""
    figure = go.Figure()
    for column in features.columns:
        figure.add_trace(
            go.Scatter(
                x=features.index,
                y=features[column],
                mode="lines",
                name=str(column),
            )
        )
    figure.update_layout(
        title="Selected feature history",
        height=420,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title="Date",
        yaxis_title="Feature value",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure


render_page_header(
    "Regime Feature Diagnostics",
    "Descriptive risk-state features for regime research — rolling "
    "volatility, correlation, drawdown, and momentum. These describe past "
    "and current market conditions; they are not forecasts, predictions, "
    "or trading signals.",
)

try:
    raw_features, scaled_features = load_feature_data(
        str(RAW_FEATURES_PATH),
        str(SCALED_FEATURES_PATH),
    )
    metadata = load_metadata(str(METADATA_PATH))
except FileNotFoundError as exc:
    st.warning(f"{exc} {RUN_MAKE_FEATURES_MESSAGE}")
    st.stop()
except (OSError, ValueError) as exc:
    st.error(
        f"Unable to load regime feature artifacts: {exc} {RUN_MAKE_FEATURES_MESSAGE}"
    )
    st.stop()

if raw_features.empty or scaled_features.empty:
    st.error(f"Regime feature artifacts are empty. {RUN_MAKE_FEATURES_MESSAGE}")
    st.stop()
if not isinstance(raw_features.index, pd.DatetimeIndex) or not isinstance(
    scaled_features.index, pd.DatetimeIndex
):
    st.error(
        f"Regime feature artifacts must have a DatetimeIndex. {RUN_MAKE_FEATURES_MESSAGE}"
    )
    st.stop()

st.sidebar.header("Feature diagnostics")
view_mode = st.sidebar.radio(
    "Feature values",
    ["Raw", "Scaled"],
    key="regime_feature_view_mode",
)
active_features = raw_features if view_mode == "Raw" else scaled_features

all_columns = list(active_features.columns)
selected_columns = st.sidebar.multiselect(
    "Features",
    all_columns,
    default=all_columns[:DEFAULT_FEATURE_COUNT],
    key="regime_feature_selected_columns",
)

min_date = active_features.index.min().date()
max_date = active_features.index.max().date()
selected_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="regime_feature_date_range",
)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

st.subheader("Dataset overview")
st.caption(f"Currently viewing: {view_mode} features")
render_metric_cards(
    [
        ("First date", str(active_features.index.min().date())),
        ("Last date", str(active_features.index.max().date())),
        ("Rows", format_decimal(len(active_features), decimals=0)),
        ("Columns", format_decimal(active_features.shape[1], decimals=0)),
    ]
)

if not selected_columns:
    st.info("Select at least one feature in the sidebar to see detailed diagnostics.")
    render_disclaimer()
    st.stop()

try:
    filtered_features = filter_by_date(
        active_features[selected_columns],
        start_date,
        end_date,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if filtered_features.empty:
    st.warning("No feature observations fall within the selected date range.")
    st.stop()

st.subheader("Latest feature values")
latest_values = filtered_features.iloc[-1].rename("value").to_frame()
latest_values.index.name = "feature"
st.caption(f"As of {filtered_features.index[-1].date()}")
st.dataframe(latest_values, width="stretch")

st.subheader("Feature history")
st.plotly_chart(build_feature_lines_figure(filtered_features), width="stretch")

st.subheader("Feature correlation")
if len(selected_columns) >= 2:
    st.plotly_chart(
        build_correlation_heatmap_figure(
            filtered_features.corr(),
            title="Selected feature correlation",
        ),
        width="stretch",
    )
else:
    st.caption("Select at least two features to see a correlation heatmap.")

st.subheader("Missing values")
missing_report = report_missing_data(filtered_features).rename(
    columns={"ticker": "feature"}
)
st.dataframe(missing_report, width="stretch")

st.subheader("Train / test split")
train_end = metadata.get("train_end")
scaler_fit_start = metadata.get("scaler_fit_start")
scaler_fit_end = metadata.get("scaler_fit_end")
test_start_label = "N/A"
test_end_label = "N/A"
if train_end:
    test_period = raw_features.index[raw_features.index > pd.Timestamp(train_end)]
    if not test_period.empty:
        test_start_label = str(test_period.min().date())
        test_end_label = str(test_period.max().date())

render_metric_cards(
    [
        ("Train end", str(train_end) if train_end else "N/A"),
        (
            "Scaler fit start",
            str(pd.Timestamp(scaler_fit_start).date()) if scaler_fit_start else "N/A",
        ),
        (
            "Scaler fit end",
            str(pd.Timestamp(scaler_fit_end).date()) if scaler_fit_end else "N/A",
        ),
        ("Test period start", test_start_label),
        ("Test period end", test_end_label),
    ]
)
st.caption(
    "The scaler is fit only on the training period and used to transform the "
    "test period; it is never refit on test data."
)
with st.expander("Full feature build metadata"):
    st.json(metadata)

render_disclaimer()
