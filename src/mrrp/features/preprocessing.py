"""Structural validation helpers for regime feature data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_feature_frame(features: pd.DataFrame) -> None:
    """Validate the structural integrity of a regime feature DataFrame."""
    _validate_dataframe(features)

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Feature data index must be a DatetimeIndex")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature data index must be monotonic increasing")
    if features.index.has_duplicates:
        raise ValueError("Feature data contains duplicate dates")


def clean_feature_matrix(
    features: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """Replace non-finite values and retain rows complete for required columns."""
    validate_feature_frame(features)
    _validate_required_columns(required_columns)

    missing_columns = [
        column for column in required_columns if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(f"Feature data missing required columns: {missing_columns}")

    cleaned = features.replace([np.inf, -np.inf], np.nan)
    return cleaned.dropna(subset=required_columns).copy()


def date_train_test_split(
    features: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered features into training and testing periods without shuffling."""
    validate_feature_frame(features)
    cutoff = _parse_train_end(train_end, features.index)

    train_features = features.loc[features.index <= cutoff].copy()
    test_features = features.loc[features.index > cutoff].copy()
    if train_features.empty:
        raise ValueError("Training feature data is empty for the configured train_end")
    if test_features.empty:
        raise ValueError("Test feature data is empty for the configured train_end")
    if train_features.index.max() >= test_features.index.min():
        raise ValueError("Training dates must be strictly before test dates")

    return train_features, test_features


def fit_train_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a standard scaler exclusively on complete training feature data."""
    _validate_scaling_matrix(train_features, label="Training feature data")
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def transform_features(
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Transform features while preserving their columns and DatetimeIndex."""
    if not isinstance(scaler, StandardScaler):
        raise ValueError("scaler must be a fitted StandardScaler")
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError("scaler must be fitted before transforming features")
    _validate_scaling_matrix(features, label="Feature data")

    transformed = scaler.transform(features)
    return pd.DataFrame(transformed, index=features.index, columns=features.columns)


def scale_train_test_features(
    features: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Clean, chronologically split, and train-scale a complete feature matrix."""
    _validate_dataframe(features)
    cleaned = clean_feature_matrix(features, required_columns=list(features.columns))
    train_features, test_features = date_train_test_split(cleaned, train_end)
    scaler = fit_train_scaler(train_features)
    return (
        transform_features(scaler, train_features),
        transform_features(scaler, test_features),
        scaler,
    )


def _validate_dataframe(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise ValueError("Feature data must be a pandas DataFrame")


def _validate_required_columns(required_columns: list[str]) -> None:
    if not isinstance(required_columns, list) or not required_columns:
        raise ValueError("required_columns must be a non-empty list of column names")
    if any(not isinstance(column, str) or not column for column in required_columns):
        raise ValueError("required_columns must contain non-empty strings")
    if len(required_columns) != len(set(required_columns)):
        raise ValueError("required_columns must not contain duplicates")


def _validate_scaling_matrix(features: pd.DataFrame, *, label: str) -> None:
    validate_feature_frame(features)
    if features.empty:
        raise ValueError(f"{label} must contain at least one row")
    if features.shape[1] == 0:
        raise ValueError(f"{label} must contain at least one column")
    if features.columns.has_duplicates:
        raise ValueError(f"{label} columns must be unique")

    try:
        values = features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite, non-missing values")


def _parse_train_end(train_end: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    if not isinstance(train_end, str) or not train_end.strip():
        raise ValueError("train_end must be a non-empty date string")
    try:
        cutoff = pd.Timestamp(train_end)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_end must be a valid date string") from exc

    if index.tz is not None:
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(index.tz)
        else:
            cutoff = cutoff.tz_convert(index.tz)
    elif cutoff.tzinfo is not None:
        raise ValueError("train_end timezone must match the feature index timezone")
    return cutoff
