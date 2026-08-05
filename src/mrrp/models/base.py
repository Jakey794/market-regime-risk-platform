"""Shared regime-model interfaces and validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from mrrp.models.result import RegimeModelResult


SUPPORTED_STATE_COUNTS = frozenset({2, 3, 4})


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological feature partitions used for leakage-safe model fitting."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def full(self) -> pd.DataFrame:
        """Return concatenated partitions in chronological order."""
        return pd.concat([self.train, self.validation, self.test], axis=0)


class RegimeModel(ABC):
    """Minimal interface implemented by concrete regime models."""

    model_name: str
    model_version: str = "1.0.0"

    @abstractmethod
    def fit(self, train_features: pd.DataFrame) -> RegimeModelResult:
        """Fit using training-period features only."""

    @abstractmethod
    def transform(self, features: pd.DataFrame) -> RegimeModelResult:
        """Infer regimes for an arbitrary chronological feature matrix."""


def validate_feature_matrix(
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    min_observations: int = 1,
) -> pd.DataFrame:
    """Validate and optionally column-select a feature matrix."""
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must contain at least one row")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("features index must be a DatetimeIndex")
    if features.index.has_duplicates:
        raise ValueError("features index contains duplicate dates")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features index must be monotonic increasing")
    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in features.columns]
        if missing:
            raise ValueError(f"features missing required columns: {missing}")
        selected = features.loc[:, list(feature_columns)].copy()
    else:
        selected = features.copy()
    if selected.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if selected.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite, non-missing values")
    if len(selected) < min_observations:
        raise ValueError(
            f"features require at least {min_observations} observations; "
            f"found {len(selected)}"
        )
    return selected


def chronological_split(
    features: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str | None = None,
) -> ChronologicalSplit:
    """Split features into train / validation / test without shuffling.

    If ``validation_end`` is omitted, validation is empty and every date after
    ``train_end`` is assigned to the test partition.
    """
    validated = validate_feature_matrix(features)
    train_cut = pd.Timestamp(train_end)
    train = validated.loc[validated.index <= train_cut].copy()
    remainder = validated.loc[validated.index > train_cut].copy()
    if train.empty:
        raise ValueError("training partition is empty for the configured train_end")
    if remainder.empty:
        raise ValueError("post-train partition is empty for the configured train_end")

    if validation_end is None:
        return ChronologicalSplit(
            train=train, validation=train.iloc[0:0], test=remainder
        )

    validation_cut = pd.Timestamp(validation_end)
    if validation_cut <= train_cut:
        raise ValueError("validation_end must be after train_end")
    validation = remainder.loc[remainder.index <= validation_cut].copy()
    test = remainder.loc[remainder.index > validation_cut].copy()
    if validation.empty:
        raise ValueError("validation partition is empty for the configured dates")
    if test.empty:
        raise ValueError("test partition is empty for the configured dates")
    return ChronologicalSplit(train=train, validation=validation, test=test)


def validate_n_states(n_states: int) -> int:
    """Validate a supported regime state count."""
    if isinstance(n_states, bool) or not isinstance(n_states, int):
        raise ValueError("n_states must be an integer")
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(
            f"n_states must be one of {sorted(SUPPORTED_STATE_COUNTS)}; got {n_states}"
        )
    return n_states


def reject_constant_features(features: pd.DataFrame) -> None:
    """Raise when any selected feature has zero training variance."""
    variances = features.to_numpy(dtype=float).var(axis=0)
    constant = [
        str(column)
        for column, variance in zip(features.columns, variances, strict=True)
        if variance <= 0.0
    ]
    if constant:
        raise ValueError(f"constant features are not supported: {constant}")
