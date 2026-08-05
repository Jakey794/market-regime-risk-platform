from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.models.base import chronological_split, validate_n_states
from mrrp.models.compare import build_comparison_table
from mrrp.models.gmm import GMMConfig, GMMRegimeModel
from mrrp.models.kmeans import KMeansConfig, KMeansRegimeModel
from mrrp.models.labeling import (
    build_state_summary,
    map_states_to_economic_labels,
    ordered_economic_labels,
)
from mrrp.models.result import RegimeModelResult, validate_regime_model_result
from mrrp.models.threshold import ThresholdConfig, ThresholdRegimeModel


FEATURE_COLUMNS = (
    "portfolio_vol_63d",
    "mean_corr_63d",
    "portfolio_drawdown",
    "portfolio_momentum_63d",
)


def _synthetic_features(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-02", periods=n, freq="W-FRI")
    regime = np.repeat([0, 1, 2, 1], n // 4 + 1)[:n]
    vol = np.where(regime == 0, 0.08, np.where(regime == 2, 0.28, 0.16))
    corr = np.where(regime == 2, 0.75, np.where(regime == 0, 0.20, 0.45))
    drawdown = np.where(regime == 2, -0.20, np.where(regime == 0, -0.02, -0.08))
    momentum = np.where(regime == 0, 0.04, np.where(regime == 2, -0.08, 0.01))
    noise = rng.normal(0.0, 0.01, size=n)
    return pd.DataFrame(
        {
            "portfolio_vol_63d": vol + noise,
            "mean_corr_63d": np.clip(corr + noise, -0.99, 0.99),
            "portfolio_drawdown": drawdown + noise * 0.5,
            "portfolio_momentum_63d": momentum + noise,
        },
        index=index,
    )


def test_validate_n_states_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="n_states must be one of"):
        validate_n_states(5)
    with pytest.raises(ValueError, match="integer"):
        validate_n_states(True)  # type: ignore[arg-type]


def test_result_contract_and_probability_rows() -> None:
    features = _synthetic_features()
    split = chronological_split(features, train_end="2020-12-31")
    model = GMMRegimeModel(
        GMMConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=11)
    )
    result = model.fit(split.train)
    validate_regime_model_result(result)
    probs = result.state_probabilities
    assert probs is not None
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert list(result.state_series().index) == list(split.train.index)
    assert result.fit_start == split.train.index.min()
    assert result.fit_end == split.train.index.max()


def test_deterministic_seeds_for_kmeans_and_gmm() -> None:
    features = _synthetic_features()
    train = chronological_split(features, train_end="2020-12-31").train
    km_a = KMeansRegimeModel(
        KMeansConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=42)
    ).fit(train)
    km_b = KMeansRegimeModel(
        KMeansConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=42)
    ).fit(train)
    assert np.array_equal(km_a.states, km_b.states)

    gmm_a = GMMRegimeModel(
        GMMConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=42)
    ).fit(train)
    gmm_b = GMMRegimeModel(
        GMMConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=42)
    ).fit(train)
    assert np.array_equal(gmm_a.states, gmm_b.states)
    assert gmm_a.state_probabilities is not None
    assert gmm_b.state_probabilities is not None
    assert np.allclose(gmm_a.state_probabilities, gmm_b.state_probabilities)


def test_train_only_fitting_and_transform_alignment() -> None:
    features = _synthetic_features()
    split = chronological_split(
        features,
        train_end="2020-06-30",
        validation_end="2021-06-30",
    )
    model = KMeansRegimeModel(
        KMeansConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=3)
    )
    fitted = model.fit(split.train)
    assert fitted.fit_end == split.train.index.max()
    transformed = model.transform(split.full)
    assert list(transformed.dates) == list(split.full.index)
    assert len(transformed.states) == len(split.full)
    # Historical training assignments must remain identical after transform.
    assert np.array_equal(
        transformed.states[: len(split.train)],
        fitted.states,
    )


def test_state_label_mapping_is_deterministic_and_complete() -> None:
    features = _synthetic_features()
    states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    summary = build_state_summary(features.iloc[:8], states)
    label_map = map_states_to_economic_labels(summary)
    labels = ordered_economic_labels(label_map)
    assert set(labels) <= {
        "calm",
        "elevated_risk",
        "high_vol_high_corr_risk_off",
        "recovery",
    }
    assert len(labels) == 4
    assert "calm" in labels


def test_threshold_model_for_supported_state_counts() -> None:
    features = _synthetic_features()
    train = chronological_split(features, train_end="2020-12-31").train
    for n_states in (2, 3, 4):
        result = ThresholdRegimeModel(ThresholdConfig(n_states=n_states)).fit(train)
        assert len(result.economic_labels) == n_states
        assert set(np.unique(result.states)).issubset(set(range(n_states)))


def test_gmm_covariance_types_and_aic_bic() -> None:
    features = _synthetic_features()
    train = chronological_split(features, train_end="2020-12-31").train
    results = []
    for covariance_type in ("full", "diag"):
        for n_states in (2, 3, 4):
            result = GMMRegimeModel(
                GMMConfig(
                    n_states=n_states,
                    covariance_type=covariance_type,
                    feature_columns=FEATURE_COLUMNS,
                    random_seed=5,
                )
            ).fit(train)
            assert "aic" in result.diagnostics
            assert "bic" in result.diagnostics
            results.append(result)
    table = build_comparison_table(results)
    assert not table.empty
    assert "comparable_metric_groups" in table.columns


def test_insufficient_observations_and_constant_features() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    tiny = pd.DataFrame(
        {
            "portfolio_vol_63d": np.linspace(0.1, 0.2, 10),
            "mean_corr_63d": np.linspace(0.2, 0.3, 10),
            "portfolio_drawdown": -0.05,
            "portfolio_momentum_63d": 0.01,
        },
        index=index,
    )
    with pytest.raises(ValueError, match="at least"):
        KMeansRegimeModel(
            KMeansConfig(n_states=4, feature_columns=FEATURE_COLUMNS)
        ).fit(tiny)

    constant = _synthetic_features().copy()
    constant["portfolio_vol_63d"] = 0.15
    constant["mean_corr_63d"] = 0.3
    constant["portfolio_drawdown"] = -0.05
    constant["portfolio_momentum_63d"] = 0.0
    train = chronological_split(constant, train_end="2020-12-31").train
    with pytest.raises(ValueError, match="constant features"):
        GMMRegimeModel(GMMConfig(n_states=3, feature_columns=FEATURE_COLUMNS)).fit(
            train
        )


def test_future_data_mutation_does_not_change_historical_outputs() -> None:
    features = _synthetic_features()
    split = chronological_split(features, train_end="2020-12-31")
    model = GMMRegimeModel(
        GMMConfig(n_states=3, feature_columns=FEATURE_COLUMNS, random_seed=9)
    )
    model.fit(split.train)
    baseline = model.transform(split.full)

    mutated = split.full.copy()
    mutated.iloc[-20:] = mutated.iloc[-20:] * 3.0
    mutated_result = model.transform(mutated)
    hist = len(split.train)
    assert np.array_equal(baseline.states[:hist], mutated_result.states[:hist])
    assert baseline.state_probabilities is not None
    assert mutated_result.state_probabilities is not None
    assert np.allclose(
        baseline.state_probabilities[:hist],
        mutated_result.state_probabilities[:hist],
    )


def test_result_to_dict_round_trip_fields() -> None:
    features = _synthetic_features(n=120)
    train = chronological_split(features, train_end="2019-12-31").train
    result = ThresholdRegimeModel(ThresholdConfig(n_states=2)).fit(train)
    payload = result.to_dict()
    assert payload["model_name"] == "threshold"
    assert isinstance(payload["states"], list)
    restored = RegimeModelResult(
        model_name=result.model_name,
        model_version=result.model_version,
        fitted_parameters=result.fitted_parameters,
        feature_columns=result.feature_columns,
        fit_start=result.fit_start,
        fit_end=result.fit_end,
        dates=result.dates,
        states=result.states,
        economic_labels=result.economic_labels,
        state_probabilities=result.state_probabilities,
        state_summary=result.state_summary,
        diagnostics=result.diagnostics,
        warnings=result.warnings,
        random_seed=result.random_seed,
    )
    validate_regime_model_result(restored)
