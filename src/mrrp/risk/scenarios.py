"""Historical and deterministic stress scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from mrrp.utils.config import ConfigError, load_yaml

ScenarioType = Literal[
    "historical_window",
    "worst_rolling",
    "deterministic_asset_shock",
    "benchmark_beta_shock",
    "volatility_shock",
    "correlation_shock",
    "sector_proxy_shock",
    "emerging_market_shock",
    "regime_conditioned",
]


@dataclass(frozen=True)
class HistoricalWindowScenario:
    """Exact observed calendar window for historical replay."""

    name: str
    start: str
    end: str
    description: str


@dataclass(frozen=True)
class DeterministicShockScenario:
    """User-defined deterministic shock specification."""

    name: str
    scenario_type: ScenarioType
    shocks: dict[str, float]
    description: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StressScenarioConfig:
    """Loaded stress-scenario configuration."""

    historical_windows: tuple[HistoricalWindowScenario, ...]
    worst_rolling_windows: tuple[int, ...]
    deterministic: tuple[DeterministicShockScenario, ...]
    regime_horizons: tuple[str, ...]


def load_stress_scenario_config(path: str) -> StressScenarioConfig:
    """Load stress scenario definitions from YAML."""
    data = load_yaml(path)
    historical = []
    for item in data.get("historical_windows", []):
        historical.append(
            HistoricalWindowScenario(
                name=str(item["name"]),
                start=str(item["start"]),
                end=str(item["end"]),
                description=str(item.get("description", "")),
            )
        )
    deterministic = []
    for item in data.get("deterministic", []):
        shocks = item.get("shocks", {})
        if not isinstance(shocks, dict):
            raise ConfigError(f"Scenario {item.get('name')} shocks must be a mapping")
        deterministic.append(
            DeterministicShockScenario(
                name=str(item["name"]),
                scenario_type=item.get("scenario_type", "deterministic_asset_shock"),
                shocks={str(k): float(v) for k, v in shocks.items()},
                description=str(item.get("description", "")),
                metadata=dict(item.get("metadata", {})),
            )
        )
    worst = tuple(int(x) for x in data.get("worst_rolling_windows", [21, 63, 126]))
    horizons = tuple(str(x) for x in data.get("regime_horizons", ["1d", "5d", "21d"]))
    return StressScenarioConfig(
        historical_windows=tuple(historical),
        worst_rolling_windows=worst,
        deterministic=tuple(deterministic),
        regime_horizons=horizons,
    )


def window_coverage(
    prices: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.Timestamp, pd.Timestamp, bool, str | None]:
    """Return clipped window bounds and a coverage warning when needed."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if prices.empty:
        raise ValueError("prices must not be empty")
    available_start = pd.Timestamp(prices.index.min())
    available_end = pd.Timestamp(prices.index.max())
    clipped_start = max(start_ts, available_start)
    clipped_end = min(end_ts, available_end)
    fully_covered = (
        clipped_start <= start_ts
        and clipped_end >= end_ts
        and clipped_start <= clipped_end
    )
    warning = None
    if clipped_start > clipped_end:
        warning = (
            f"Requested window {start}→{end} has no overlap with available data "
            f"{available_start.date()}→{available_end.date()}"
        )
    elif not fully_covered:
        warning = (
            f"ETF history only partially covers {start}→{end}; "
            f"using {clipped_start.date()}→{clipped_end.date()}"
        )
    return clipped_start, clipped_end, fully_covered, warning
