"""Typed result contract for regime model fits and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeModelResult:
    """Common output contract for regime detection models.

    Attributes
    ----------
    model_name:
        Stable model family identifier (e.g. ``threshold``, ``kmeans``, ``gmm``).
    model_version:
        Semantic version string for the implementing module contract.
    fitted_parameters:
        Serialisable parameter snapshot used for diagnostics and persistence.
    feature_columns:
        Ordered feature names used during fit/transform.
    fit_start / fit_end:
        Inclusive chronological bounds of the training observations used to fit.
    dates:
        Observation timestamps aligned with ``states``.
    states:
        Integer state assignments aligned with ``dates``.
    economic_labels:
        Human-interpretable labels derived from training-period state statistics.
    state_probabilities:
        Optional soft assignments with shape ``(n_obs, n_states)``; rows should
        sum to one when present.
    state_summary:
        Per-state descriptive statistics used for labeling and reporting.
    diagnostics:
        Model-specific scores and diagnostics (AIC/BIC, silhouette, etc.).
    warnings:
        Non-fatal issues encountered during fit or inference.
    random_seed:
        Seed used for stochastic initialisation, when applicable.
    """

    model_name: str
    model_version: str
    fitted_parameters: dict[str, Any]
    feature_columns: tuple[str, ...]
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    dates: pd.DatetimeIndex
    states: np.ndarray
    economic_labels: tuple[str, ...]
    state_probabilities: np.ndarray | None
    state_summary: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    random_seed: int | None = None

    def labeled_states(self) -> pd.Series:
        """Return economic labels aligned to ``dates`` as a Series."""
        if len(self.states) != len(self.dates):
            raise ValueError("states and dates must have the same length")
        if self.states.size and (
            int(np.min(self.states)) < 0
            or int(np.max(self.states)) >= len(self.economic_labels)
        ):
            raise ValueError("states contain indices outside economic_labels")
        labels = [self.economic_labels[int(state)] for state in self.states]
        return pd.Series(labels, index=self.dates, name="regime_label")

    def state_series(self) -> pd.Series:
        """Return integer states aligned to ``dates``."""
        if len(self.states) != len(self.dates):
            raise ValueError("states and dates must have the same length")
        return pd.Series(self.states.astype(int), index=self.dates, name="regime_state")

    def probability_frame(self) -> pd.DataFrame | None:
        """Return soft probabilities as a DataFrame when available."""
        if self.state_probabilities is None:
            return None
        columns = [f"state_{idx}" for idx in range(self.state_probabilities.shape[1])]
        return pd.DataFrame(
            self.state_probabilities,
            index=self.dates,
            columns=columns,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (arrays converted to lists)."""
        payload = asdict(self)
        payload["fit_start"] = self.fit_start.isoformat()
        payload["fit_end"] = self.fit_end.isoformat()
        payload["dates"] = [ts.isoformat() for ts in self.dates]
        payload["states"] = self.states.astype(int).tolist()
        payload["feature_columns"] = list(self.feature_columns)
        payload["economic_labels"] = list(self.economic_labels)
        payload["warnings"] = list(self.warnings)
        if self.state_probabilities is None:
            payload["state_probabilities"] = None
        else:
            payload["state_probabilities"] = self.state_probabilities.tolist()
        payload["state_summary"] = self.state_summary.to_dict(orient="index")
        return payload


def validate_regime_model_result(result: RegimeModelResult) -> None:
    """Validate structural invariants of a regime model result."""
    if not isinstance(result, RegimeModelResult):
        raise TypeError("result must be a RegimeModelResult")
    if not result.model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not result.model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    if not result.feature_columns:
        raise ValueError("feature_columns must be non-empty")
    if not isinstance(result.dates, pd.DatetimeIndex):
        raise ValueError("dates must be a DatetimeIndex")
    if result.dates.has_duplicates:
        raise ValueError("dates must be unique")
    if not result.dates.is_monotonic_increasing:
        raise ValueError("dates must be monotonic increasing")
    if result.fit_start > result.fit_end:
        raise ValueError("fit_start must not be after fit_end")
    states = np.asarray(result.states)
    if states.ndim != 1:
        raise ValueError("states must be a 1-d array")
    if len(states) != len(result.dates):
        raise ValueError("states and dates must have the same length")
    if len(result.economic_labels) == 0:
        raise ValueError("economic_labels must be non-empty")
    if states.size:
        if np.any(states < 0) or np.any(states >= len(result.economic_labels)):
            raise ValueError("states contain indices outside economic_labels")
    if result.state_probabilities is not None:
        probs = np.asarray(result.state_probabilities, dtype=float)
        if probs.ndim != 2:
            raise ValueError("state_probabilities must be 2-d")
        if probs.shape[0] != len(result.dates):
            raise ValueError("state_probabilities rows must align with dates")
        if probs.shape[1] != len(result.economic_labels):
            raise ValueError("state_probabilities columns must match label count")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each probability row must sum to one")
    if not isinstance(result.state_summary, pd.DataFrame):
        raise ValueError("state_summary must be a DataFrame")
