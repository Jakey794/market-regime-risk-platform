"""Model comparison helpers with family-aware, non-forced metric comparability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mrrp.models.result import RegimeModelResult


@dataclass(frozen=True)
class ModelComparisonRow:
    """One comparable diagnostics row for a fitted regime model."""

    model_family: str
    model_name: str
    n_states: int | None
    train_likelihood: float | None
    aic: float | None
    bic: float | None
    silhouette: float | None
    average_regime_duration: float | None
    n_transitions: int | None
    regime_stability: float | None
    interpretability: str
    convergence_warnings: tuple[str, ...]
    limitations: str
    comparable_metric_groups: tuple[str, ...]


def compute_regime_stability(states: np.ndarray) -> tuple[float, int, float]:
    """Return stability, transition count, and average regime duration."""
    arr = np.asarray(states, dtype=int)
    if arr.size == 0:
        return 1.0, 0, float("nan")
    transitions = int(np.sum(arr[1:] != arr[:-1]))
    stability = float(1.0 - transitions / max(arr.size - 1, 1))
    if transitions == 0:
        avg_duration = float(arr.size)
    else:
        # Duration of contiguous runs.
        change_idx = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        bounds = np.r_[0, change_idx, arr.size]
        durations = np.diff(bounds)
        avg_duration = float(durations.mean())
    return stability, transitions, avg_duration


def comparison_row_from_result(result: RegimeModelResult) -> ModelComparisonRow:
    """Build a comparison row from a typed regime model result."""
    stability, transitions, avg_duration = compute_regime_stability(result.states)
    diagnostics = result.diagnostics
    n_states = diagnostics.get("n_states")
    if n_states is None:
        n_states = len(result.economic_labels)

    family = result.model_name
    if family == "threshold":
        groups = ("stability", "interpretability")
        interpretability = "high"
        limitations = (
            "Rule-based baseline; no likelihood, AIC/BIC, or soft probabilities."
        )
    elif family == "kmeans":
        groups = ("silhouette", "stability", "interpretability")
        interpretability = "medium"
        limitations = (
            "Distance-based clustering; silhouette is not comparable to likelihood "
            "or AIC/BIC from probabilistic models."
        )
    elif family == "gmm":
        groups = ("likelihood_ic", "stability", "probabilities")
        interpretability = "medium"
        limitations = (
            "AIC/BIC are comparable among GMM fits with the same feature matrix; "
            "not directly comparable to KMeans silhouette or HMM train score."
        )
    elif family == "hmm":
        groups = ("likelihood_ic", "stability", "transition_structure")
        interpretability = "medium"
        limitations = (
            "Train log-likelihood and transition structure are HMM-specific; "
            "do not rank against KMeans silhouette as if they share a loss."
        )
    elif family == "changepoint":
        groups = ("stability", "interpretability")
        interpretability = "high"
        limitations = (
            "Offline segmentation; breakpoints are not generative state labels "
            "and lack soft probabilities."
        )
    else:
        groups = ("stability",)
        interpretability = "unknown"
        limitations = "Unknown model family; treat metrics as descriptive only."

    return ModelComparisonRow(
        model_family=family,
        model_name=f"{result.model_name}:{n_states}",
        n_states=int(n_states) if n_states is not None else None,
        train_likelihood=_as_optional_float(
            diagnostics.get("train_log_likelihood", diagnostics.get("lower_bound"))
        ),
        aic=_as_optional_float(diagnostics.get("aic")),
        bic=_as_optional_float(diagnostics.get("bic")),
        silhouette=_as_optional_float(diagnostics.get("silhouette")),
        average_regime_duration=avg_duration,
        n_transitions=transitions,
        regime_stability=stability,
        interpretability=interpretability,
        convergence_warnings=tuple(result.warnings),
        limitations=limitations,
        comparable_metric_groups=groups,
    )


def build_comparison_table(results: Iterable[RegimeModelResult]) -> pd.DataFrame:
    """Assemble a comparison table that preserves metric non-comparability notes."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row = comparison_row_from_result(result)
        rows.append(
            {
                "model_family": row.model_family,
                "model_name": row.model_name,
                "n_states": row.n_states,
                "train_likelihood": row.train_likelihood,
                "aic": row.aic,
                "bic": row.bic,
                "silhouette": row.silhouette,
                "average_regime_duration": row.average_regime_duration,
                "n_transitions": row.n_transitions,
                "regime_stability": row.regime_stability,
                "interpretability": row.interpretability,
                "convergence_warnings": "; ".join(row.convergence_warnings),
                "limitations": row.limitations,
                "comparable_metric_groups": ",".join(row.comparable_metric_groups),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_family",
                "model_name",
                "n_states",
                "train_likelihood",
                "aic",
                "bic",
                "silhouette",
                "average_regime_duration",
                "n_transitions",
                "regime_stability",
                "interpretability",
                "convergence_warnings",
                "limitations",
                "comparable_metric_groups",
            ]
        )
    return pd.DataFrame(rows)


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
