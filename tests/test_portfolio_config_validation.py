"""End-to-end YAML-parsing validation for portfolio configuration.

These tests exercise ``load_portfolio_config``'s full file-parsing path, as
distinct from ``test_portfolio_config.py``'s tests of ``validate_weights``
called directly on a ``pd.Series``. "Unknown ticker" and "benchmark must
exist in loaded data" are deliberately not covered here: ``PortfolioConfig``
has no concept of a ticker universe or price data to validate against, and
those checks only exist once tickers are matched against real price columns
in ``mrrp.dashboard.loaders`` (see ``tests/test_dashboard_loaders.py``).
"""

from __future__ import annotations

import pytest

from mrrp.portfolio import PortfolioConfig, load_portfolio_config
from mrrp.utils.config import ConfigError


def test_sample_portfolio_config_loads() -> None:
    config = load_portfolio_config("configs/sample_portfolio.yaml")

    assert isinstance(config, PortfolioConfig)
    assert config.holdings.sum() == pytest.approx(1.0)


def test_invalid_weight_sum_rejected_via_yaml(tmp_path) -> None:
    path = tmp_path / "bad_sum.yaml"
    path.write_text(
        """
name: bad_sum
benchmark: SPY
currency: CAD
allow_short: false
holdings:
  SPY: 0.7
  QQQ: 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must sum to 1.0"):
        load_portfolio_config(path)


def test_duplicate_ticker_rejected_via_case_normalization(tmp_path) -> None:
    path = tmp_path / "duplicate_ticker.yaml"
    path.write_text(
        """
name: duplicate_ticker
benchmark: SPY
currency: CAD
allow_short: false
holdings:
  SPY: 0.5
  spy: 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Duplicate ticker"):
        load_portfolio_config(path)


def test_negative_weight_rejected_via_yaml(tmp_path) -> None:
    path = tmp_path / "negative_weight.yaml"
    path.write_text(
        """
name: negative_weight
benchmark: SPY
currency: CAD
allow_short: false
holdings:
  SPY: 1.1
  QQQ: -0.1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Negative weights"):
        load_portfolio_config(path)
