"""Configuration loading and validation utilities.

This module keeps project configuration centralized. The goal is to avoid
hardcoded ticker lists, benchmark symbols, and portfolio weights scattered
throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing required or valid fields."""


@dataclass(frozen=True)
class UniverseConfig:
    """Validated market universe configuration."""

    assets: dict[str, list[str]]
    benchmark: str
    start_date: str
    end_date: str | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    """Validated portfolio configuration."""

    name: str
    benchmark: str
    weights: dict[str, float]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If the file is empty or does not parse to a dictionary.
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {config_path}")

    return data


def load_universe_config(path: str | Path) -> UniverseConfig:
    """Load and validate the asset universe config."""
    data = load_yaml(path)

    required_fields = {"assets", "benchmark", "start_date"}
    missing = required_fields - set(data)

    if missing:
        raise ConfigError(f"Universe config missing required fields: {sorted(missing)}")

    assets = data["assets"]
    benchmark = data["benchmark"]
    start_date = data["start_date"]
    end_date = data.get("end_date")

    if not isinstance(assets, dict) or not assets:
        raise ConfigError("'assets' must be a non-empty mapping of groups to tickers")

    normalized_assets: dict[str, list[str]] = {}
    for group_name, tickers in assets.items():
        if not isinstance(group_name, str) or not group_name:
            raise ConfigError("Asset group names must be non-empty strings")

        if not isinstance(tickers, list) or not tickers:
            raise ConfigError(
                f"Asset group '{group_name}' must contain a non-empty list"
            )

        normalized_tickers: list[str] = []
        for ticker in tickers:
            if not isinstance(ticker, str) or not ticker.strip():
                raise ConfigError(f"Invalid ticker in group '{group_name}': {ticker!r}")
            normalized_tickers.append(ticker.strip().upper())

        normalized_assets[group_name] = normalized_tickers

    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ConfigError("'benchmark' must be a non-empty string")

    if not isinstance(start_date, str) or not start_date.strip():
        raise ConfigError("'start_date' must be a non-empty YYYY-MM-DD string")

    if end_date is not None and not isinstance(end_date, str):
        raise ConfigError("'end_date' must be null or a YYYY-MM-DD string")

    return UniverseConfig(
        assets=normalized_assets,
        benchmark=benchmark.strip().upper(),
        start_date=start_date.strip(),
        end_date=end_date.strip() if isinstance(end_date, str) else None,
    )


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load and validate a portfolio config."""
    data = load_yaml(path)

    required_fields = {"name", "benchmark"}
    missing = required_fields - set(data)

    if missing:
        raise ConfigError(
            f"Portfolio config missing required fields: {sorted(missing)}"
        )

    name = data["name"]
    benchmark = data["benchmark"]
    weights = data.get("weights", data.get("holdings"))

    if not isinstance(name, str) or not name.strip():
        raise ConfigError("'name' must be a non-empty string")

    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ConfigError("'benchmark' must be a non-empty string")

    if not isinstance(weights, dict) or not weights:
        raise ConfigError(
            "Portfolio config must contain a non-empty 'holdings' mapping"
        )

    normalized_weights: dict[str, float] = {}
    for ticker, weight in weights.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ConfigError(f"Invalid ticker in weights: {ticker!r}")

        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid weight for ticker {ticker!r}: {weight!r}"
            ) from exc

        if numeric_weight < 0:
            raise ConfigError(f"Negative weights are not supported yet: {ticker}")

        normalized_weights[ticker.strip().upper()] = numeric_weight

    weight_sum = sum(normalized_weights.values())
    if abs(weight_sum - 1.0) > 1e-8:
        raise ConfigError(f"Portfolio weights must sum to 1.0; got {weight_sum:.10f}")

    return PortfolioConfig(
        name=name.strip(),
        benchmark=benchmark.strip().upper(),
        weights=normalized_weights,
    )
