"""Regime-model comparison dashboard page."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mrrp.dashboard.components import render_disclaimer, render_page_header
from mrrp.dashboard.paths import regime_feature_paths
from mrrp.data.cache import load_parquet
from mrrp.models import (
    ChangePointConfig,
    ChangePointDetector,
    GMMConfig,
    GMMRegimeModel,
    HMMConfig,
    HMMRegimeModel,
    KMeansConfig,
    KMeansRegimeModel,
    ThresholdConfig,
    ThresholdRegimeModel,
    build_comparison_table,
)


render_page_header(
    "Regime Detection",
    "Descriptive market-risk states estimated from trailing features. Regimes are "
    "not forecasts, trading signals, or guarantees of future behavior.",
)
paths = regime_feature_paths()
try:
    features = load_parquet(paths.scaled).dropna()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.warning(f"Regime model artifacts are unavailable: {exc}. Run `make features`.")
    render_disclaimer()
    st.stop()

model_name = st.selectbox("Model", ["Threshold", "KMeans", "GMM", "HMM"])
n_states = st.selectbox("State count", [2, 3, 4], index=1)
train_size = max(60, int(len(features) * 0.7))
train = features.iloc[:train_size]
if model_name == "Threshold":
    model = ThresholdRegimeModel(ThresholdConfig(n_states=n_states))
elif model_name == "KMeans":
    model = KMeansRegimeModel(
        KMeansConfig(n_states=n_states, feature_columns=tuple(features.columns))
    )
elif model_name == "GMM":
    model = GMMRegimeModel(
        GMMConfig(n_states=n_states, feature_columns=tuple(features.columns))
    )
else:
    model = HMMRegimeModel(
        HMMConfig(
            n_states=n_states,
            feature_columns=tuple(features.columns),
            n_inits=2,
        )
    )

try:
    model.fit(train)
    result = model.transform(features)
except (RuntimeError, ValueError) as exc:
    st.error(f"Unable to fit selected model: {exc}")
    render_disclaimer()
    st.stop()

labels = result.labeled_states()
st.metric("Current regime", str(labels.iloc[-1]))
st.caption(
    f"Selected model: {result.model_name}; {len(result.economic_labels)} states; "
    f"fit through {result.fit_end.date()}."
)
figure = go.Figure(
    go.Scatter(
        x=features.index,
        y=features.iloc[:, 0],
        mode="lines",
        name=str(features.columns[0]),
        line={"color": "#334155"},
    )
)
for state in sorted(set(result.states)):
    mask = result.states == state
    figure.add_trace(
        go.Scatter(
            x=features.index[mask],
            y=features.iloc[:, 0][mask],
            mode="markers",
            name=f"State {state}",
            marker={"size": 4},
        )
    )
figure.update_layout(title="Regime-shaded feature history", height=420)
st.plotly_chart(figure, width="stretch")
st.subheader("State summary")
st.dataframe(result.state_summary, width="stretch")

if result.state_probabilities is not None:
    probabilities = pd.DataFrame(
        result.state_probabilities,
        index=result.dates,
        columns=[f"State {idx}" for idx in range(result.state_probabilities.shape[1])],
    )
    st.subheader("State probabilities")
    st.line_chart(probabilities)
if model_name == "HMM":
    st.subheader("HMM transition matrix")
    st.dataframe(pd.DataFrame(result.diagnostics["transition_matrix"]), width="stretch")

detector = ChangePointDetector(
    ChangePointConfig(method="binseg", signal="multivariate", n_bkps=3)
)
change_points = detector.detect(features)
st.subheader("Change-point overlays")
st.write([date.date().isoformat() for date in change_points.break_dates])
st.subheader("Comparison table")
st.dataframe(build_comparison_table([result]), width="stretch")
st.info(
    "Limitations: state identities depend on sample and specification; HMM assumptions "
    "can be restrictive; change points are retrospective; model metrics are not always "
    "comparable across families."
)
render_disclaimer()
