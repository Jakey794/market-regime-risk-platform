from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.models.changepoint import ChangePointConfig, ChangePointDetector


def _features() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    return pd.DataFrame(
        {
            "portfolio_vol_63d": np.repeat([0.1, 0.4, 0.2], 40),
            "mean_corr_63d": np.repeat([0.2, 0.8, 0.3], 40),
        },
        index=index,
    )


def test_changepoints_are_deterministic_sorted_and_internal() -> None:
    features = _features()
    detector = ChangePointDetector(
        ChangePointConfig(method="binseg", signal="multivariate", n_bkps=2)
    )
    first = detector.detect(features)
    second = detector.detect(features)
    assert first.break_indices == second.break_indices
    assert first.break_dates == tuple(sorted(first.break_dates))
    assert all(
        features.index.min() < date < features.index.max() for date in first.break_dates
    )


def test_training_window_changepoints_ignore_future_mutation() -> None:
    features = _features()
    detector = ChangePointDetector(ChangePointConfig(method="pelt", penalty=1.0))
    baseline = detector.detect(features.iloc[:80])
    mutated = features.copy()
    mutated.iloc[80:] = 99.0
    changed = detector.detect(mutated.iloc[:80])
    assert baseline.break_indices == changed.break_indices
    assert baseline.break_dates == changed.break_dates
