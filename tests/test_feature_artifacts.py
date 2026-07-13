from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mrrp.data.cache import load_parquet, save_parquet
from mrrp.features.build_features import build_feature_artifacts
from mrrp.features.schema import EXPECTED_REGIME_FEATURE_COLUMNS
from mrrp.features.validate_features import validate_feature_artifacts


def test_build_and_validate_feature_artifacts(tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    positions = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1.005 + 0.02 * np.sin(positions)),
            "B": 100 * np.cumprod(1.003 + 0.015 * np.cos(positions * 0.7)),
            "SPY": 100 * np.cumprod(1.004 + 0.018 * np.sin(positions * 0.4)),
        },
        index=index,
    )
    prices_path = tmp_path / "prices.parquet"
    feature_config_path = tmp_path / "regime_features.yaml"
    portfolio_config_path = tmp_path / "portfolio.yaml"
    raw_path = tmp_path / "raw.parquet"
    scaled_path = tmp_path / "scaled.parquet"
    metadata_path = tmp_path / "metadata.json"
    save_parquet(prices, prices_path)
    feature_config_path.write_text(
        """
windows:
  short: 3
  medium: 5
  long: 8
annualization_factor: 252
train_end: "2020-02-14"
thresholds:
  high_vol_percentile: 0.75
  high_corr_percentile: 0.75
  drawdown_warning: -0.05
""",
        encoding="utf-8",
    )
    portfolio_config_path.write_text(
        """
name: synthetic_portfolio
benchmark: SPY
currency: USD
allow_short: false
holdings:
  A: 0.6
  B: 0.4
""",
        encoding="utf-8",
    )

    metadata = build_feature_artifacts(
        feature_config_path=feature_config_path,
        portfolio_config_path=portfolio_config_path,
        prices_path=prices_path,
        raw_output_path=raw_path,
        scaled_output_path=scaled_path,
        metadata_output_path=metadata_path,
    )
    report = validate_feature_artifacts(raw_path, scaled_path)
    raw = load_parquet(raw_path)
    scaled = load_parquet(scaled_path)
    scaled_train = scaled.loc[scaled.index <= pd.Timestamp("2020-02-14")]

    assert metadata["asset_universe"] == ["A", "B"]
    assert metadata["benchmark"] == "SPY"
    assert metadata["dropped_rows"] > 0
    assert metadata["scaler_fit_end"] == scaled_train.index.max().isoformat()
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert report["raw_shape"] == raw.shape == scaled.shape
    assert raw.index.equals(scaled.index)
    assert scaled_train.mean().to_numpy() == pytest.approx(
        np.zeros(scaled.shape[1]), abs=1e-12
    )


def test_validate_feature_artifacts_rejects_infinite_values(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    raw = pd.DataFrame(0.0, index=index, columns=EXPECTED_REGIME_FEATURE_COLUMNS)
    scaled = raw.copy()
    scaled.iloc[0, 0] = np.inf
    raw_path = tmp_path / "raw.parquet"
    scaled_path = tmp_path / "scaled.parquet"
    save_parquet(raw, raw_path)
    save_parquet(scaled, scaled_path)

    with pytest.raises(ValueError, match="infinite values"):
        validate_feature_artifacts(raw_path, scaled_path)
