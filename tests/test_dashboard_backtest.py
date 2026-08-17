from pathlib import Path

from mrrp.dashboard.backtest import load_or_build_backtest_features
from mrrp.dashboard.regime_artifacts import load_or_build_regime_artifacts
from mrrp.dashboard.paths import prices_path
from mrrp.data.cache import load_parquet
from mrrp.features.schema import EXPECTED_REGIME_FEATURE_COLUMNS
from mrrp.portfolio import load_portfolio_config


def test_backtest_features_fall_back_to_selected_prices(tmp_path: Path) -> None:
    prices = load_parquet(prices_path())
    portfolio = load_portfolio_config("configs/sample_portfolio.yaml")

    features, used_fallback = load_or_build_backtest_features(
        prices,
        portfolio,
        artifact_path=tmp_path / "missing.parquet",
        feature_config_path="configs/regime_features.yaml",
    )

    assert used_fallback
    assert features.index.equals(prices.index)
    assert set(EXPECTED_REGIME_FEATURE_COLUMNS).issubset(features.columns)


def test_regime_artifacts_fall_back_to_selected_prices(tmp_path: Path) -> None:
    prices = load_parquet(prices_path())
    portfolio = load_portfolio_config("configs/sample_portfolio.yaml")

    raw, scaled, metadata, used_fallback = load_or_build_regime_artifacts(
        prices,
        portfolio,
        raw_path=tmp_path / "missing-raw.parquet",
        scaled_path=tmp_path / "missing-scaled.parquet",
        metadata_path=tmp_path / "missing-metadata.json",
        feature_config_path="configs/regime_features.yaml",
    )

    assert used_fallback
    assert raw.index.equals(scaled.index)
    assert set(EXPECTED_REGIME_FEATURE_COLUMNS).issubset(raw.columns)
    assert metadata["train_end"] == "2021-12-31"
