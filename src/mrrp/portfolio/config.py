"""Typed portfolio configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from mrrp.portfolio.weights import validate_weights
from mrrp.utils.config import ConfigError, load_yaml


@dataclass(frozen=True)
class PortfolioConfig:
    """Validated, reproducible portfolio configuration."""

    name: str
    benchmark: str
    currency: str
    allow_short: bool
    holdings: pd.Series


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load and validate a portfolio configuration from YAML."""
    data = load_yaml(path)
    required_fields = {"name", "benchmark", "currency", "allow_short", "holdings"}
    missing = required_fields - set(data)
    if missing:
        raise ConfigError(
            f"Portfolio config missing required fields: {sorted(missing)}"
        )

    name = _non_empty_string(data["name"], field="name")
    benchmark = _non_empty_string(data["benchmark"], field="benchmark").upper()
    currency = _non_empty_string(data["currency"], field="currency").upper()

    allow_short = data["allow_short"]
    if not isinstance(allow_short, bool):
        raise ConfigError("'allow_short' must be true or false")

    holdings = _holdings_series(data["holdings"])
    if len(holdings) < 2:
        raise ConfigError("Portfolio must contain at least 2 assets")

    try:
        validate_weights(holdings, allow_short=allow_short)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    return PortfolioConfig(
        name=name,
        benchmark=benchmark,
        currency=currency,
        allow_short=allow_short,
        holdings=holdings,
    )


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string")
    return value.strip()


def _holdings_series(value: Any) -> pd.Series:
    if not isinstance(value, dict) or not value:
        raise ConfigError("'holdings' must be a non-empty mapping of ticker to weight")

    normalized: dict[str, float] = {}
    for ticker, weight in value.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ConfigError(f"Invalid ticker in holdings: {ticker!r}")

        normalized_ticker = ticker.strip().upper()
        if normalized_ticker in normalized:
            raise ConfigError(f"Duplicate ticker in holdings: {normalized_ticker}")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ConfigError(f"Weight for {normalized_ticker} must be numeric")
        normalized[normalized_ticker] = float(weight)

    return pd.Series(normalized, dtype=float, name="weight")
