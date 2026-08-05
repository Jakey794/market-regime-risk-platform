"""Deterministic economic labeling of regime states from training statistics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_VOL_COLUMN = "portfolio_vol_63d"
DEFAULT_CORR_COLUMN = "mean_corr_63d"
DEFAULT_DRAWDOWN_COLUMN = "portfolio_drawdown"
DEFAULT_MOMENTUM_COLUMN = "portfolio_momentum_63d"


def build_state_summary(
    features: pd.DataFrame,
    states: np.ndarray | Sequence[int],
    *,
    vol_column: str = DEFAULT_VOL_COLUMN,
    corr_column: str = DEFAULT_CORR_COLUMN,
    drawdown_column: str = DEFAULT_DRAWDOWN_COLUMN,
    momentum_column: str = DEFAULT_MOMENTUM_COLUMN,
) -> pd.DataFrame:
    """Summarise training-period feature means by integer state."""
    state_array = np.asarray(states, dtype=int)
    if len(state_array) != len(features):
        raise ValueError("states must align with features")
    if state_array.ndim != 1:
        raise ValueError("states must be 1-d")

    required = [vol_column, corr_column, drawdown_column, momentum_column]
    missing = [column for column in required if column not in features.columns]
    if missing:
        # Fall back to first available numeric columns for synthetic fixtures.
        numeric = list(features.columns)
        if len(numeric) < 2:
            raise ValueError(
                "features must include volatility/correlation columns or at least "
                "two numeric columns for state labeling"
            )
        vol_column = numeric[0]
        corr_column = numeric[1]
        drawdown_column = numeric[min(2, len(numeric) - 1)]
        momentum_column = numeric[min(3, len(numeric) - 1)]

    frame = features.copy()
    frame["__state__"] = state_array
    grouped = frame.groupby("__state__", sort=True)
    summary = grouped.agg(
        n_obs=("__state__", "size"),
        mean_vol=(vol_column, "mean"),
        mean_corr=(corr_column, "mean"),
        mean_drawdown=(drawdown_column, "mean"),
        mean_momentum=(momentum_column, "mean"),
    )
    summary.index.name = "state"
    return summary


def map_states_to_economic_labels(
    state_summary: pd.DataFrame,
) -> dict[int, str]:
    """Map raw state ids to stable economic labels using training statistics.

    Labeling is intentionally transparent and rank-based:

    - lowest mean volatility -> ``calm``
    - highest mean volatility and high correlation -> ``high_vol_high_corr_risk_off``
    - remaining high-vol / weak-momentum states -> ``elevated_risk``
    - improving momentum / shallower drawdown residual -> ``recovery``

    The mapping depends only on training-period state statistics so inference on
    later periods cannot reshuffle historical economic labels.
    """
    if state_summary.empty:
        raise ValueError("state_summary must not be empty")
    required = {"mean_vol", "mean_corr", "mean_drawdown", "mean_momentum"}
    missing = required - set(state_summary.columns)
    if missing:
        raise ValueError(f"state_summary missing columns: {sorted(missing)}")

    states = [int(state) for state in state_summary.index]
    n_states = len(states)
    ordered_by_vol = sorted(
        states, key=lambda s: float(state_summary.loc[s, "mean_vol"])
    )
    labels: dict[int, str] = {}

    if n_states == 2:
        labels[ordered_by_vol[0]] = "calm"
        labels[ordered_by_vol[1]] = "elevated_risk"
        return labels

    if n_states == 3:
        labels[ordered_by_vol[0]] = "calm"
        high_states = ordered_by_vol[1:]
        risk_off = max(
            high_states,
            key=lambda s: (
                float(state_summary.loc[s, "mean_corr"]),
                float(state_summary.loc[s, "mean_vol"]),
            ),
        )
        recovery = [state for state in high_states if state != risk_off][0]
        labels[risk_off] = "high_vol_high_corr_risk_off"
        labels[recovery] = "recovery"
        return labels

    # n_states == 4 (and any larger supported count uses the same priority set)
    labels[ordered_by_vol[0]] = "calm"
    remaining = ordered_by_vol[1:]
    risk_off = max(
        remaining,
        key=lambda s: (
            float(state_summary.loc[s, "mean_vol"]),
            float(state_summary.loc[s, "mean_corr"]),
        ),
    )
    labels[risk_off] = "high_vol_high_corr_risk_off"
    remaining = [state for state in remaining if state != risk_off]
    recovery = max(
        remaining,
        key=lambda s: (
            float(state_summary.loc[s, "mean_momentum"]),
            -float(state_summary.loc[s, "mean_drawdown"]),
        ),
    )
    labels[recovery] = "recovery"
    for state in remaining:
        if state != recovery:
            labels[state] = "elevated_risk"
    return labels


def ordered_economic_labels(label_map: Mapping[int, str]) -> tuple[str, ...]:
    """Return economic labels ordered by raw state integer."""
    if not label_map:
        raise ValueError("label_map must not be empty")
    max_state = max(int(state) for state in label_map)
    ordered: list[str] = []
    for state in range(max_state + 1):
        if state not in label_map:
            raise ValueError(f"label_map missing state {state}")
        ordered.append(str(label_map[state]))
    return tuple(ordered)


def apply_label_map(
    states: np.ndarray | Sequence[int],
    label_map: Mapping[int, str],
) -> tuple[str, ...]:
    """Translate integer states through a frozen training-period label map."""
    return tuple(label_map[int(state)] for state in states)
