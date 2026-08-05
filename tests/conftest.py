"""Shared pytest fixtures for deterministic offline tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mrrp.dashboard.paths import FEATURE_DIR_ENV
from mrrp.features.schema import EXPECTED_REGIME_FEATURE_COLUMNS


@pytest.fixture
def regime_feature_artifact_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Create synthetic regime feature artifacts and point the dashboard at them."""
    index = pd.date_range("2020-01-02", periods=120, freq="B")
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(index), len(EXPECTED_REGIME_FEATURE_COLUMNS)))
    raw = pd.DataFrame(
        values,
        index=index,
        columns=list(EXPECTED_REGIME_FEATURE_COLUMNS),
    )
    scaled = (raw - raw.mean()) / raw.std(ddof=0)

    raw_path = tmp_path / "regime_features_raw.parquet"
    scaled_path = tmp_path / "regime_features_scaled.parquet"
    metadata_path = tmp_path / "regime_feature_metadata.json"
    raw.to_parquet(raw_path)
    scaled.to_parquet(scaled_path)
    metadata_path.write_text(
        json.dumps(
            {
                "train_end": "2020-06-01",
                "scaler_fit_start": "2020-01-02",
                "scaler_fit_end": "2020-06-01",
                "feature_columns": list(EXPECTED_REGIME_FEATURE_COLUMNS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(FEATURE_DIR_ENV, str(tmp_path))
    return tmp_path
