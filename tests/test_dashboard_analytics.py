from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.reporting.memo_context import (
    classify_correlation_regime_from_features,
    classify_volatility_regime_from_features,
    regime_agreement_text,
)
from mrrp.reporting.plots import (
    build_regime_shaded_wealth_figure,
    build_transition_heatmap_figure,
)


def test_feature_regime_classifiers_are_deterministic() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    features = pd.DataFrame(
        {
            "high_vol_flag": [False, False, False, False, True],
            "high_corr_flag": [False, False, False, False, True],
            "vol_z_252d": [0.0, 0.1, 0.2, 0.3, 1.5],
            "mean_corr_63d": [0.2, 0.2, 0.3, 0.4, 0.8],
        },
        index=index,
    )
    assert classify_volatility_regime_from_features(features) == "High volatility"
    assert classify_correlation_regime_from_features(features) == "High correlation"
    assert "disagree" in regime_agreement_text({"a": "calm", "b": "elevated_risk"})


def test_regime_plot_helpers_build_figures() -> None:
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.Series(np.full(20, 0.001), index=index, name="ret")
    labels = pd.Series(["calm"] * 10 + ["elevated_risk"] * 10, index=index)
    figure = build_regime_shaded_wealth_figure(
        returns, labels, change_points=[index[10]]
    )
    assert len(figure.data) >= 2
    heat = build_transition_heatmap_figure([[0.8, 0.2], [0.1, 0.9]], labels=["a", "b"])
    assert heat.data[0].z[0][0] == 0.8
