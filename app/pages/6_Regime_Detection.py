"""Regime-model comparison dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mrrp.reporting.memo_context import (
    classify_correlation_regime_from_features,
    classify_volatility_regime_from_features,
    regime_agreement_text,
)
from mrrp.dashboard.components import (
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.paths import (
    DEFAULT_CONFIG_DIR,
    portfolio_config_path,
    prices_path,
    regime_feature_paths,
)
from mrrp.dashboard.regime_artifacts import load_or_build_regime_artifacts
from mrrp.dashboard.state import get_dashboard_data
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
from mrrp.portfolio import load_portfolio_config
from mrrp.reporting import (
    build_regime_shaded_wealth_figure,
    build_transition_heatmap_figure,
)


render_page_header(
    "Regime Detection",
    "Descriptive market-risk states estimated from trailing features. Regimes are "
    "not forecasts, trading signals, or guarantees of future behavior.",
)

try:
    data = get_dashboard_data(st.session_state)
except ValueError:
    data = None

paths = regime_feature_paths()
try:
    source_prices = load_parquet(prices_path())
    raw_features, scaled_features, _, used_session_fallback = (
        load_or_build_regime_artifacts(
            source_prices,
            load_portfolio_config(portfolio_config_path()),
            raw_path=paths.raw,
            scaled_path=paths.scaled,
            metadata_path=paths.metadata,
            feature_config_path=DEFAULT_CONFIG_DIR / "regime_features.yaml",
        )
    )
    features = scaled_features.dropna()
    raw_features = raw_features.reindex(features.index)
except (FileNotFoundError, OSError, ValueError) as exc:
    st.warning(f"Regime features are unavailable: {exc}")
    render_disclaimer()
    st.stop()

if used_session_fallback:
    st.info(
        "Regime feature artifacts are unavailable, so this session derived and "
        "train-scaled leakage-safe features from the tracked demo prices."
    )

vol_regime = classify_volatility_regime_from_features(raw_features)
corr_regime = classify_correlation_regime_from_features(raw_features)
render_metric_cards(
    [
        ("Current volatility regime", vol_regime),
        ("Current correlation regime", corr_regime),
        ("Feature rows", str(len(features))),
        ("Train fit share", "70% chronological"),
    ],
    columns=4,
)

model_name = st.selectbox("Model", ["Threshold", "KMeans", "GMM", "HMM"])
n_states = st.selectbox("State count", [2, 3, 4], index=1)
train_size = max(60, int(len(features) * 0.7))
train = features.iloc[:train_size]
feature_columns = tuple(features.columns)

if model_name == "Threshold":
    model = ThresholdRegimeModel(ThresholdConfig(n_states=n_states))
elif model_name == "KMeans":
    model = KMeansRegimeModel(
        KMeansConfig(n_states=n_states, feature_columns=feature_columns)
    )
elif model_name == "GMM":
    model = GMMRegimeModel(
        GMMConfig(n_states=n_states, feature_columns=feature_columns)
    )
else:
    model = HMMRegimeModel(
        HMMConfig(
            n_states=n_states,
            feature_columns=feature_columns,
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
st.metric("Selected model current regime", str(labels.iloc[-1]))
st.caption(
    f"Selected model: {result.model_name}; {len(result.economic_labels)} states; "
    f"fit through {result.fit_end.date()}."
)

if data is not None:
    wealth_returns = data.portfolio_returns.reindex(result.dates).fillna(0.0)
else:
    wealth_returns = (
        raw_features["portfolio_return_1d"].reindex(result.dates).fillna(0.0)
        if "portfolio_return_1d" in raw_features.columns
        else pd.Series(0.0, index=result.dates)
    )

detector = ChangePointDetector(
    ChangePointConfig(method="binseg", signal="multivariate", n_bkps=3)
)
change_points = detector.detect(features)
st.subheader("Regime-shaded wealth")
st.plotly_chart(
    build_regime_shaded_wealth_figure(
        wealth_returns,
        labels,
        change_points=change_points.break_dates,
    ),
    width="stretch",
)

st.subheader("Regime summary")
st.dataframe(result.state_summary, width="stretch")

if result.state_probabilities is not None:
    probabilities = pd.DataFrame(
        result.state_probabilities,
        index=result.dates,
        columns=list(result.economic_labels),
    )
    st.subheader("GMM / HMM state probabilities")
    st.line_chart(probabilities)

if model_name == "HMM" and "transition_matrix" in result.diagnostics:
    st.subheader("HMM transition-matrix heatmap")
    st.plotly_chart(
        build_transition_heatmap_figure(
            result.diagnostics["transition_matrix"],
            labels=list(result.economic_labels),
        ),
        width="stretch",
    )

st.subheader("Change-point dates")
st.write([date.date().isoformat() for date in change_points.break_dates])

st.subheader("Side-by-side model comparison")
comparison_results = [result]
comparison_labels: dict[str, str] = {result.model_name: str(labels.iloc[-1])}
for family, factory in (
    (
        "threshold",
        lambda: ThresholdRegimeModel(ThresholdConfig(n_states=n_states)),
    ),
    (
        "kmeans",
        lambda: KMeansRegimeModel(
            KMeansConfig(n_states=n_states, feature_columns=feature_columns)
        ),
    ),
    (
        "gmm",
        lambda: GMMRegimeModel(
            GMMConfig(n_states=n_states, feature_columns=feature_columns)
        ),
    ),
):
    if family == result.model_name:
        continue
    try:
        other = factory()
        other.fit(train)
        other_result = other.transform(features)
        comparison_results.append(other_result)
        comparison_labels[other_result.model_name] = str(
            other_result.labeled_states().iloc[-1]
        )
    except (RuntimeError, ValueError):
        continue

st.caption(regime_agreement_text(comparison_labels))
st.dataframe(build_comparison_table(comparison_results), width="stretch")
st.info(
    "Limitations: state identities depend on sample and specification; HMM assumptions "
    "can be restrictive; change points are retrospective; model metrics are not always "
    "comparable across families. These outputs describe historical risk conditions only."
)
render_disclaimer()
