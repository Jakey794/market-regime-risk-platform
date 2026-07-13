from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.features import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    RegimeFeatureConfig,
    RegimeFeatureSet,
    RegimeFeatureThresholds,
    RegimeFeatureWindows,
    build_basic_regime_features,
    compute_drawdown,
    compute_mean_pairwise_corr,
    compute_momentum,
    compute_portfolio_return,
    compute_rolling_volatility,
    load_regime_feature_config,
    trailing_percentile_rank,
    trailing_zscore,
    validate_regime_feature_columns,
)
from mrrp.utils.config import ConfigError


def _feature_frame(periods: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    data = {
        column: [float(i) for i in range(periods)]
        for column in EXPECTED_REGIME_FEATURE_COLUMNS
    }
    return pd.DataFrame(data, index=index)


def test_expected_regime_feature_columns_match_spec() -> None:
    assert list(EXPECTED_REGIME_FEATURE_COLUMNS) == [
        "portfolio_return_1d",
        "benchmark_return_1d",
        "portfolio_vol_21d",
        "portfolio_vol_63d",
        "portfolio_vol_252d",
        "benchmark_vol_21d",
        "benchmark_vol_63d",
        "mean_corr_21d",
        "mean_corr_63d",
        "portfolio_drawdown",
        "portfolio_drawdown_z_252d",
        "portfolio_momentum_21d",
        "portfolio_momentum_63d",
        "benchmark_momentum_63d",
        "vol_z_252d",
        "corr_z_252d",
        "drawdown_flag",
        "high_vol_flag",
        "high_corr_flag",
    ]


def test_expected_regime_feature_columns_has_nineteen_entries() -> None:
    assert len(EXPECTED_REGIME_FEATURE_COLUMNS) == 19


def test_regime_feature_set_stores_fields() -> None:
    raw = _feature_frame()
    scaled = _feature_frame()
    metadata = {"train_end": "2021-12-31"}

    feature_set = RegimeFeatureSet(
        raw_features=raw,
        scaled_features=scaled,
        metadata=metadata,
    )

    assert feature_set.raw_features is raw
    assert feature_set.scaled_features is scaled
    assert feature_set.metadata == metadata


def test_regime_feature_set_allows_none_scaled_features() -> None:
    feature_set = RegimeFeatureSet(
        raw_features=_feature_frame(),
        scaled_features=None,
        metadata={},
    )

    assert feature_set.scaled_features is None


def test_regime_feature_set_preserves_datetime_index() -> None:
    raw = _feature_frame()
    original_index = raw.index

    validate_regime_feature_columns(raw)
    feature_set = RegimeFeatureSet(raw_features=raw, scaled_features=None, metadata={})

    assert isinstance(feature_set.raw_features.index, pd.DatetimeIndex)
    assert feature_set.raw_features.index.equals(original_index)


def test_validate_regime_feature_columns_passes_for_complete_frame() -> None:
    validate_regime_feature_columns(_feature_frame())


def test_validate_regime_feature_columns_rejects_missing_column() -> None:
    features = _feature_frame().drop(columns=["portfolio_return_1d"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_regime_feature_columns(features)


def test_validate_regime_feature_columns_rejects_unexpected_column() -> None:
    features = _feature_frame()
    features["unexpected_column"] = 0.0

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_regime_feature_columns(features)


def test_regime_feature_config_loads_from_sample_yaml() -> None:
    config = load_regime_feature_config("configs/regime_features.yaml")

    assert isinstance(config, RegimeFeatureConfig)
    assert config.windows.short == 21
    assert config.windows.medium == 63
    assert config.windows.long == 252
    assert config.annualization_factor == 252
    assert config.train_end == "2021-12-31"
    assert config.thresholds.high_vol_percentile == pytest.approx(0.75)
    assert config.thresholds.high_corr_percentile == pytest.approx(0.75)
    assert config.thresholds.drawdown_warning == pytest.approx(-0.10)


def test_regime_feature_config_missing_field_raises(tmp_path) -> None:
    path = tmp_path / "missing_windows.yaml"
    path.write_text(
        """
annualization_factor: 252
train_end: "2021-12-31"
thresholds:
  high_vol_percentile: 0.75
  high_corr_percentile: 0.75
  drawdown_warning: -0.10
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="missing required fields.*windows"):
        load_regime_feature_config(path)


def test_regime_feature_config_invalid_threshold_raises(tmp_path) -> None:
    path = tmp_path / "invalid_threshold.yaml"
    path.write_text(
        """
windows:
  short: 21
  medium: 63
  long: 252
annualization_factor: 252
train_end: "2021-12-31"
thresholds:
  high_vol_percentile: 1.5
  high_corr_percentile: 0.75
  drawdown_warning: -0.10
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_regime_feature_config(path)


def _asset_returns(periods: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.015, 0.005][:periods],
            "B": [0.02, -0.01, 0.02, 0.01, -0.005, 0.015][:periods],
        },
        index=index,
    )


def test_compute_portfolio_return_matches_known_weighted_average() -> None:
    asset_returns = _asset_returns()
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = compute_portfolio_return(asset_returns, weights)

    expected = asset_returns["A"] * 0.6 + asset_returns["B"] * 0.4
    assert result.tolist() == pytest.approx(expected.tolist())
    assert result.index.equals(asset_returns.index)


def test_compute_portfolio_return_missing_weight_raises() -> None:
    asset_returns = _asset_returns()
    weights = pd.Series({"A": 0.5, "C": 0.5})

    with pytest.raises(ValueError, match="missing return columns"):
        compute_portfolio_return(asset_returns, weights)


def test_compute_portfolio_return_invalid_weight_sum_raises() -> None:
    asset_returns = _asset_returns()
    weights = pd.Series({"A": 0.3, "B": 0.2})

    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_portfolio_return(asset_returns, weights)


def test_compute_rolling_volatility_first_valid_after_full_window() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01, -0.02], index=index)

    result = compute_rolling_volatility(returns, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()
    assert result.index.equals(index)


def test_compute_momentum_matches_known_path() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = [0.01, 0.02, -0.01, 0.03, 0.015]
    returns = pd.Series(values, index=index)

    result = compute_momentum(returns, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(np.prod(1 + np.array(values[0:3])) - 1)
    assert result.iloc[3] == pytest.approx(np.prod(1 + np.array(values[1:4])) - 1)
    assert result.iloc[4] == pytest.approx(np.prod(1 + np.array(values[2:5])) - 1)
    assert result.index.equals(index)


def test_compute_momentum_rejects_invalid_window() -> None:
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError, match="window"):
        compute_momentum(returns, window=1)


def test_compute_drawdown_matches_known_path() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    wealth = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 130.0], index=index)
    returns = wealth.pct_change().fillna(0.0)

    result = compute_drawdown(returns)

    assert result.iloc[2] == pytest.approx(-0.25)
    assert result.iloc[4] == pytest.approx(-1 / 3)
    assert result.index.equals(index)


def test_compute_functions_preserve_datetime_index() -> None:
    asset_returns = _asset_returns()
    weights = pd.Series({"A": 0.6, "B": 0.4})
    index = asset_returns.index

    portfolio_return = compute_portfolio_return(asset_returns, weights)
    volatility = compute_rolling_volatility(portfolio_return, window=3)
    momentum = compute_momentum(portfolio_return, window=3)
    drawdown = compute_drawdown(portfolio_return)

    assert portfolio_return.index.equals(index)
    assert volatility.index.equals(index)
    assert momentum.index.equals(index)
    assert drawdown.index.equals(index)


def test_trailing_zscore_matches_known_series() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = trailing_zscore(series, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([1.0, 1.0])


def test_trailing_zscore_preserves_index() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    result = trailing_zscore(series, window=3)

    assert result.index.equals(index)


def test_trailing_zscore_zero_std_returns_nan_without_infinity() -> None:
    series = pd.Series([2.0, 2.0, 2.0, 3.0])

    result = trailing_zscore(series, window=3)

    assert pd.isna(result.iloc[2])
    assert not np.isinf(result.to_numpy()).any()


def test_trailing_percentile_rank_uses_only_trailing_window() -> None:
    series = pd.Series([1.0, 4.0, 2.0, 3.0])

    result = trailing_percentile_rank(series, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2 / 3)
    assert result.iloc[3] == pytest.approx(2 / 3)

    changed_future = series.copy()
    changed_future.iloc[-1] = 100.0
    changed_result = trailing_percentile_rank(changed_future, window=3)
    assert changed_result.iloc[2] == pytest.approx(result.iloc[2])


def _known_flag_feature_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    known_returns = pd.Series([0.0, 0.1, -0.2, 0.05, 0.3, -0.1], index=index)
    asset_returns = pd.DataFrame({"A": known_returns, "B": known_returns})
    return build_basic_regime_features(
        asset_returns,
        pd.Series({"A": 0.5, "B": 0.5}),
        known_returns.rename("benchmark"),
        windows=RegimeFeatureWindows(short=2, medium=2, long=3),
        thresholds=RegimeFeatureThresholds(
            high_vol_percentile=0.75,
            high_corr_percentile=0.60,
            drawdown_warning=-0.15,
        ),
    )


def test_high_vol_flag_matches_known_series() -> None:
    result = _known_flag_feature_frame()

    assert result["high_vol_flag"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_high_corr_flag_matches_known_series() -> None:
    result = _known_flag_feature_frame()

    assert result["high_corr_flag"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_drawdown_flag_matches_known_path() -> None:
    result = _known_flag_feature_frame()

    assert result["drawdown_flag"].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_build_basic_regime_features_returns_expected_columns() -> None:
    asset_returns = _asset_returns(periods=6)
    weights = pd.Series({"A": 0.6, "B": 0.4})
    benchmark_returns = pd.Series(
        [0.01, 0.0, -0.01, 0.02, 0.01, -0.005],
        index=asset_returns.index,
        name="benchmark",
    )
    windows = RegimeFeatureWindows(short=2, medium=3, long=4)
    thresholds = RegimeFeatureThresholds(
        high_vol_percentile=0.75,
        high_corr_percentile=0.75,
        drawdown_warning=-0.10,
    )

    result = build_basic_regime_features(
        asset_returns,
        weights,
        benchmark_returns,
        windows=windows,
        thresholds=thresholds,
    )

    assert list(result.columns) == [
        "portfolio_return_1d",
        "benchmark_return_1d",
        "portfolio_vol_21d",
        "portfolio_vol_63d",
        "portfolio_vol_252d",
        "benchmark_vol_21d",
        "benchmark_vol_63d",
        "mean_corr_21d",
        "mean_corr_63d",
        "portfolio_drawdown",
        "portfolio_drawdown_z_252d",
        "portfolio_momentum_21d",
        "portfolio_momentum_63d",
        "benchmark_momentum_63d",
        "vol_z_252d",
        "corr_z_252d",
        "drawdown_flag",
        "high_vol_flag",
        "high_corr_flag",
    ]
    assert result.index.equals(asset_returns.index)


def test_build_basic_regime_features_rejects_misaligned_benchmark() -> None:
    asset_returns = _asset_returns()
    weights = pd.Series({"A": 0.6, "B": 0.4})
    benchmark_returns = pd.Series(
        [0.01, 0.0, -0.01],
        index=pd.date_range("2030-01-01", periods=3, freq="D"),
    )
    windows = RegimeFeatureWindows(short=2, medium=3, long=4)
    thresholds = RegimeFeatureThresholds(
        high_vol_percentile=0.75,
        high_corr_percentile=0.75,
        drawdown_warning=-0.10,
    )

    with pytest.raises(ValueError, match="benchmark_returns index"):
        build_basic_regime_features(
            asset_returns,
            weights,
            benchmark_returns,
            windows=windows,
            thresholds=thresholds,
        )


def test_compute_mean_pairwise_corr_perfect_positive_correlation() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    a = pd.Series([0.01, 0.02, -0.01, 0.03], index=index)
    returns = pd.DataFrame({"A": a, "B": a})

    result = compute_mean_pairwise_corr(returns, window=3)

    assert result.iloc[2:].tolist() == pytest.approx([1.0, 1.0])


def test_compute_mean_pairwise_corr_perfect_negative_correlation() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    a = pd.Series([0.01, 0.02, -0.01, 0.03], index=index)
    returns = pd.DataFrame({"A": a, "B": -a})

    result = compute_mean_pairwise_corr(returns, window=3)

    assert result.iloc[2:].tolist() == pytest.approx([-1.0, -1.0])


def test_compute_mean_pairwise_corr_three_assets_ignores_diagonal() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    a = pd.Series([0.01, 0.02, -0.01, 0.03], index=index)
    returns = pd.DataFrame({"A": a, "B": a, "C": -a})

    result = compute_mean_pairwise_corr(returns, window=4)

    assert result.iloc[-1] == pytest.approx(-1 / 3)


def test_compute_mean_pairwise_corr_first_valid_at_window_minus_one() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.015],
            "B": [0.02, -0.01, 0.02, 0.01, -0.005],
        },
        index=index,
    )

    result = compute_mean_pairwise_corr(returns, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_compute_mean_pairwise_corr_preserves_datetime_index() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.015],
            "B": [0.02, -0.01, 0.02, 0.01, -0.005],
        },
        index=index,
    )

    result = compute_mean_pairwise_corr(returns, window=3)

    assert result.index.equals(index)


def test_compute_mean_pairwise_corr_rejects_duplicate_dates() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.015],
            "B": [0.02, -0.01, 0.02, 0.01, -0.005],
        },
        index=index,
    )
    returns.index = returns.index[:-1].append(returns.index[-2:-1])

    with pytest.raises(ValueError, match="duplicate dates"):
        compute_mean_pairwise_corr(returns, window=3)


def test_compute_mean_pairwise_corr_rejects_invalid_window() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01], "B": [0.02, -0.01, 0.02]},
        index=index,
    )

    with pytest.raises(ValueError, match="window"):
        compute_mean_pairwise_corr(returns, window=1)


def test_compute_mean_pairwise_corr_ignores_future_data() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    returns_a = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.015],
            "B": [0.02, -0.01, 0.02, 0.01, -0.005],
        },
        index=index,
    )
    returns_b = returns_a.copy()
    returns_b.iloc[-1] = [5.0, -5.0]

    result_a = compute_mean_pairwise_corr(returns_a, window=3)
    result_b = compute_mean_pairwise_corr(returns_b, window=3)

    assert result_a.iloc[:-1].equals(result_b.iloc[:-1])
