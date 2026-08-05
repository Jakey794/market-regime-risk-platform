from __future__ import annotations

import numpy as np
import pandas as pd

from mrrp.models.hmm import HMMConfig, HMMRegimeModel


def _features() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.date_range("2019-01-01", periods=180, freq="B")
    state = np.repeat([-2.0, 0.0, 2.0], 60)
    return pd.DataFrame(
        {
            "vol": state + rng.normal(0, 0.25, len(index)),
            "corr": state * 0.5 + rng.normal(0, 0.2, len(index)),
        },
        index=index,
    )


def test_hmm_is_deterministic_and_train_only() -> None:
    features = _features()
    train = features.iloc[:120]
    config = HMMConfig(n_states=3, random_seed=7, n_inits=2, n_iter=100)
    first = HMMRegimeModel(config)
    second = HMMRegimeModel(config)
    first_fit = first.fit(train)
    second_fit = second.fit(train)
    assert (
        first_fit.fitted_parameters["selected_init_seed"]
        == second_fit.fitted_parameters["selected_init_seed"]
    )
    np.testing.assert_array_equal(first_fit.states, second_fit.states)
    assert first_fit.fit_end == train.index[-1]


def test_hmm_future_mutation_cannot_change_earlier_filtered_states() -> None:
    features = _features()
    model = HMMRegimeModel(HMMConfig(n_states=3, random_seed=9, n_inits=2))
    model.fit(features.iloc[:100])
    baseline = model.transform(features)
    mutated = features.copy()
    mutated.iloc[150:] = mutated.iloc[150:] * -100
    changed = model.transform(mutated)
    np.testing.assert_array_equal(baseline.states[:150], changed.states[:150])
    np.testing.assert_allclose(
        baseline.state_probabilities[:150],
        changed.state_probabilities[:150],
    )
