"""Parquet cache helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to parquet, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a cached parquet DataFrame."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet cache does not exist: {input_path}")

    return pd.read_parquet(input_path)


def cache_exists(path: str | Path) -> bool:
    """Return whether a parquet cache path exists."""
    return Path(path).exists()
