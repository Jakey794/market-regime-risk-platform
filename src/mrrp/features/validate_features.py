"""Validate persisted raw and scaled regime feature artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrrp.data.cache import load_parquet
from mrrp.features.build_features import DEFAULT_RAW_OUTPUT, DEFAULT_SCALED_OUTPUT
from mrrp.features.preprocessing import validate_feature_frame
from mrrp.features.schema import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    validate_regime_feature_columns,
)


def validate_feature_artifacts(
    raw_path: str | Path = DEFAULT_RAW_OUTPUT,
    scaled_path: str | Path = DEFAULT_SCALED_OUTPUT,
) -> dict[str, Any]:
    """Validate persisted feature matrices and return report fields."""
    raw_features = load_parquet(raw_path)
    scaled_features = load_parquet(scaled_path)

    _validate_matrix(raw_features, label="Raw")
    _validate_matrix(scaled_features, label="Scaled")
    if not raw_features.index.equals(scaled_features.index):
        raise ValueError("Raw and scaled feature indexes do not match")
    if not raw_features.columns.equals(scaled_features.columns):
        raise ValueError("Raw and scaled feature columns do not match")

    return {
        "raw_shape": raw_features.shape,
        "scaled_shape": scaled_features.shape,
        "start": raw_features.index.min().isoformat(),
        "end": raw_features.index.max().isoformat(),
        "feature_count": len(EXPECTED_REGIME_FEATURE_COLUMNS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--scaled", default=str(DEFAULT_SCALED_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_feature_artifacts(args.raw, args.scaled)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"Feature validation failed: {exc}", file=sys.stderr)
        return 1

    print("Feature validation passed")
    print(f"Raw shape: {report['raw_shape']}")
    print(f"Scaled shape: {report['scaled_shape']}")
    print(f"Date range: {report['start']} to {report['end']}")
    print(f"Expected feature columns: {report['feature_count']}")
    return 0


def _validate_matrix(features: pd.DataFrame, *, label: str) -> None:
    validate_feature_frame(features)
    validate_regime_feature_columns(features)
    if features.empty:
        raise ValueError(f"{label} feature data is empty")

    required_columns = list(EXPECTED_REGIME_FEATURE_COLUMNS)
    missing_values = features.loc[:, required_columns].isna().sum()
    columns_with_missing = missing_values[missing_values.gt(0)].index.tolist()
    if columns_with_missing:
        raise ValueError(
            f"{label} feature data contains missing required values in columns: "
            f"{columns_with_missing}"
        )

    try:
        values = features.loc[:, required_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} feature data must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError(f"{label} feature data contains infinite values")


if __name__ == "__main__":
    raise SystemExit(main())
