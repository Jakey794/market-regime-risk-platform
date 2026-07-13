"""Leakage-safe regime feature contracts and configuration."""

from mrrp.features.preprocessing import (
    clean_feature_matrix,
    date_train_test_split,
    fit_train_scaler,
    scale_train_test_features,
    transform_features,
    validate_feature_frame,
)
from mrrp.features.regime_features import (
    RegimeFeatureConfig,
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
)
from mrrp.features.schema import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    RegimeFeatureSet,
    validate_regime_feature_columns,
)

__all__ = [
    "EXPECTED_REGIME_FEATURE_COLUMNS",
    "RegimeFeatureConfig",
    "RegimeFeatureSet",
    "RegimeFeatureThresholds",
    "RegimeFeatureWindows",
    "build_basic_regime_features",
    "clean_feature_matrix",
    "compute_drawdown",
    "compute_mean_pairwise_corr",
    "compute_momentum",
    "compute_portfolio_return",
    "compute_rolling_volatility",
    "date_train_test_split",
    "fit_train_scaler",
    "load_regime_feature_config",
    "scale_train_test_features",
    "trailing_percentile_rank",
    "trailing_zscore",
    "transform_features",
    "validate_feature_frame",
    "validate_regime_feature_columns",
]
