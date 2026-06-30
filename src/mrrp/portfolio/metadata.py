"""Asset metadata loading and grouped portfolio exposure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mrrp.portfolio.weights import validate_weights
from mrrp.utils.config import ConfigError, load_yaml


AssetMetadata = dict[str, dict[str, str]]


def load_asset_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Load ticker metadata from YAML and validate string field values."""
    data = load_yaml(path)
    metadata: AssetMetadata = {}

    for ticker, attributes in data.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ConfigError(f"Invalid metadata ticker: {ticker!r}")
        if not isinstance(attributes, dict):
            raise ConfigError(f"Metadata for {ticker!r} must be a mapping")

        normalized_ticker = ticker.strip().upper()
        if normalized_ticker in metadata:
            raise ConfigError(f"Duplicate metadata ticker: {normalized_ticker}")
        metadata[normalized_ticker] = _validate_attributes(
            normalized_ticker,
            attributes,
        )

    return metadata


def compute_group_exposure(
    weights: pd.Series,
    metadata: dict,
    group_key: str,
) -> pd.Series:
    """Aggregate signed weights by a metadata field, using ``Unknown`` if absent.

    Signed weights are retained, so grouped exposure sums to portfolio net
    exposure. Missing tickers, missing fields, and blank field values are
    assigned to the ``Unknown`` group.
    """
    validate_weights(weights, allow_short=True)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    if not isinstance(group_key, str) or not group_key.strip():
        raise ValueError("group_key must be a non-empty string")
    normalized_group_key = group_key.strip()

    groups = [
        _metadata_group(metadata, str(ticker), normalized_group_key)
        for ticker in weights.index
    ]
    exposure = pd.Series(
        weights.to_numpy(dtype=float),
        index=pd.Index(groups, name=normalized_group_key),
        name="weight",
    )
    return exposure.groupby(level=0, sort=False).sum().sort_values(
        ascending=False,
        kind="stable",
    )


def _validate_attributes(ticker: str, attributes: dict[Any, Any]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in attributes.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"Metadata keys for {ticker} must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"Metadata value for {ticker}.{key} must be a non-empty string"
            )
        validated[key.strip()] = value.strip()
    return validated


def _metadata_group(metadata: dict, ticker: str, group_key: str) -> str:
    attributes = metadata.get(ticker.upper())
    if not isinstance(attributes, dict):
        return "Unknown"

    group = attributes.get(group_key)
    if not isinstance(group, str) or not group.strip():
        return "Unknown"
    return group.strip()
