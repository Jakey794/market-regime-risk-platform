from __future__ import annotations

import pytest

from mrrp.utils.config import (
    ConfigError,
    PortfolioConfig,
    UniverseConfig,
    load_portfolio_config,
    load_universe_config,
)


def test_universe_config_loads() -> None:
    config = load_universe_config("configs/default_universe.yaml")

    assert isinstance(config, UniverseConfig)
    assert config.benchmark == "SPY"
    assert config.start_date == "2005-01-01"
    assert config.end_date is None
    assert "us_equity" in config.assets
    assert "SPY" in config.assets["us_equity"]


def test_portfolio_config_loads() -> None:
    config = load_portfolio_config("configs/sample_portfolio.yaml")

    assert isinstance(config, PortfolioConfig)
    assert config.name == "sample_global_equity_portfolio"
    assert config.benchmark == "SPY"
    assert config.weights["SPY"] == 0.35
    assert abs(sum(config.weights.values()) - 1.0) < 1e-8


def test_invalid_universe_config_raises_clean_error(tmp_path) -> None:
    # pytest's tmp_path fixture gives each test a unique temporary pathlib.Path.
    # This avoids polluting the real configs/ directory.
    invalid_config = tmp_path / "invalid_universe.yaml"
    invalid_config.write_text(
        """
benchmark: SPY
start_date: "2005-01-01"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="missing required fields"):
        load_universe_config(invalid_config)


def test_invalid_portfolio_weights_raise_clean_error(tmp_path) -> None:
    invalid_config = tmp_path / "invalid_portfolio.yaml"
    invalid_config.write_text(
        """
name: invalid_portfolio
benchmark: SPY
weights:
  SPY: 0.60
  QQQ: 0.60
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must sum to 1.0"):
        load_portfolio_config(invalid_config)


def test_negative_portfolio_weight_raises_clean_error(tmp_path) -> None:
    invalid_config = tmp_path / "negative_weight_portfolio.yaml"
    invalid_config.write_text(
        """
name: invalid_portfolio
benchmark: SPY
weights:
  SPY: 1.10
  QQQ: -0.10
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Negative weights"):
        load_portfolio_config(invalid_config)
