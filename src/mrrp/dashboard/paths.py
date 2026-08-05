"""Configurable filesystem paths for dashboard artifact loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

FEATURE_DIR_ENV = "MRRP_FEATURE_DIR"
PROCESSED_DIR_ENV = "MRRP_PROCESSED_DIR"
PRICES_PATH_ENV = "MRRP_PRICES_PATH"
PORTFOLIO_CONFIG_ENV = "MRRP_PORTFOLIO_CONFIG"


@dataclass(frozen=True)
class RegimeFeaturePaths:
    """Filesystem locations for regime feature artifacts."""

    raw: Path
    scaled: Path
    metadata: Path


def processed_data_dir() -> Path:
    """Return the processed-data directory, honoring test/deploy overrides."""
    override = os.environ.get(PROCESSED_DIR_ENV)
    if override:
        return Path(override)
    return DEFAULT_PROCESSED_DIR


def regime_feature_paths() -> RegimeFeaturePaths:
    """Return regime feature artifact paths, honoring ``MRRP_FEATURE_DIR``."""
    override = os.environ.get(FEATURE_DIR_ENV)
    base = Path(override) if override else processed_data_dir()
    return RegimeFeaturePaths(
        raw=base / "regime_features_raw.parquet",
        scaled=base / "regime_features_scaled.parquet",
        metadata=base / "regime_feature_metadata.json",
    )


def prices_path() -> Path:
    """Return the adjusted-close prices path used by the dashboard shell."""
    override = os.environ.get(PRICES_PATH_ENV)
    if override:
        return Path(override)
    return processed_data_dir() / "adjusted_close.parquet"


def portfolio_config_path() -> Path:
    """Return the portfolio YAML path used by the dashboard shell."""
    override = os.environ.get(PORTFOLIO_CONFIG_ENV)
    if override:
        return Path(override)
    return DEFAULT_CONFIG_DIR / "sample_portfolio.yaml"
