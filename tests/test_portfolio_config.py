from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.portfolio import (
    PortfolioConfig,
    get_portfolio_tickers,
    load_portfolio_config,
    normalize_weights,
    validate_weights,
)
from mrrp.utils.config import ConfigError


def test_valid_portfolio_config_loads() -> None:
    config = load_portfolio_config("configs/sample_portfolio.yaml")

    assert isinstance(config, PortfolioConfig)
    assert config.name == "sample_global_equity_portfolio"
    assert config.benchmark == "SPY"
    assert config.currency == "CAD"
    assert config.allow_short is False
    assert config.holdings["SPY"] == pytest.approx(0.35)


def test_weights_sum_to_one() -> None:
    weights = pd.Series({"SPY": 0.6, "QQQ": 0.4})

    assert validate_weights(weights) is None


def test_invalid_weight_sum_raises() -> None:
    weights = pd.Series({"SPY": 0.7, "QQQ": 0.4})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_weights(weights)


def test_negative_weight_rejected() -> None:
    weights = pd.Series({"SPY": 1.1, "QQQ": -0.1})

    with pytest.raises(ValueError, match="Negative weights"):
        validate_weights(weights)


def test_negative_weight_allowed_when_allow_short_true() -> None:
    weights = pd.Series({"SPY": 1.1, "QQQ": -0.1})

    assert validate_weights(weights, allow_short=True) is None


def test_missing_benchmark_rejected(tmp_path) -> None:
    path = tmp_path / "missing_benchmark.yaml"
    path.write_text(
        """
name: missing_benchmark
currency: CAD
allow_short: false
holdings:
  SPY: 0.6
  QQQ: 0.4
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="missing required fields.*benchmark"):
        load_portfolio_config(path)


def test_normalize_weights() -> None:
    weights = pd.Series({"SPY": 2.0, "QQQ": 3.0}, name="weight")

    result = normalize_weights(weights)

    assert result.name == "weight"
    assert result.to_dict() == pytest.approx({"SPY": 0.4, "QQQ": 0.6})


def test_get_portfolio_tickers() -> None:
    config = load_portfolio_config("configs/sample_portfolio.yaml")

    assert get_portfolio_tickers(config) == [
        "SPY",
        "QQQ",
        "XIU.TO",
        "EFA",
        "EEM",
        "XLK",
    ]


def test_portfolio_requires_at_least_two_assets(tmp_path) -> None:
    path = tmp_path / "one_asset.yaml"
    path.write_text(
        """
name: one_asset
benchmark: SPY
currency: CAD
allow_short: false
holdings:
  SPY: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="at least 2 assets"):
        load_portfolio_config(path)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (pd.Series(dtype=float), "empty"),
        (pd.Series([0.5, 0.5], index=["SPY", "SPY"]), "unique"),
        (pd.Series([0.5, 0.5], index=["SPY", " "]), "non-empty"),
        (pd.Series({"SPY": "half", "QQQ": "half"}), "numeric"),
        (pd.Series({"SPY": np.inf, "QQQ": -np.inf}), "finite"),
    ],
)
def test_invalid_weight_values_raise(weights: pd.Series, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_weights(weights)
