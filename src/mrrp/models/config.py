"""Load and validate regime-model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mrrp.utils.config import ConfigError, load_yaml


@dataclass(frozen=True)
class RegimeModelsConfig:
    """Validated configuration for the regime-model suite."""

    random_seed: int
    train_end: str
    validation_end: str | None
    feature_columns: tuple[str, ...]
    raw: dict[str, Any]


def load_regime_models_config(path: str | Path) -> RegimeModelsConfig:
    """Load regime model configuration from YAML."""
    data = load_yaml(path)
    required = {"random_seed", "train_end", "feature_columns"}
    missing = required - set(data)
    if missing:
        raise ConfigError(
            f"Regime model config missing required fields: {sorted(missing)}"
        )
    feature_columns = data["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ConfigError("feature_columns must be a non-empty list")
    if any(not isinstance(column, str) or not column for column in feature_columns):
        raise ConfigError("feature_columns must contain non-empty strings")
    seed = data["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("random_seed must be an integer")
    train_end = data["train_end"]
    if not isinstance(train_end, str) or not train_end.strip():
        raise ConfigError("train_end must be a non-empty string")
    validation_end = data.get("validation_end")
    if validation_end is not None and (
        not isinstance(validation_end, str) or not validation_end.strip()
    ):
        raise ConfigError("validation_end must be a non-empty string when provided")
    return RegimeModelsConfig(
        random_seed=seed,
        train_end=train_end,
        validation_end=validation_end,
        feature_columns=tuple(feature_columns),
        raw=data,
    )
