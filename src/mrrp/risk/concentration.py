"""Portfolio concentration metrics based on gross exposure shares."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from mrrp.portfolio.weights import validate_weights


def compute_hhi(weights: pd.Series) -> float:
    """Return the Herfindahl-Hirschman Index of absolute exposure shares.

    Absolute weights are normalized by gross exposure so short positions
    contribute to concentration without leverage changing the scale.
    """
    concentration_weights = _concentration_weights(weights)
    return float(concentration_weights.pow(2).sum())


def compute_effective_num_holdings(weights: pd.Series) -> float:
    """Return the inverse HHI effective holding count using absolute weights."""
    return float(1.0 / compute_hhi(weights))


def compute_top_weight(weights: pd.Series, n: int = 1) -> float:
    """Return the ``n``th-largest normalized absolute portfolio weight."""
    concentration_weights = _concentration_weights(weights)
    position = _validate_n(n, len(concentration_weights), allow_oversized=False)
    return float(concentration_weights.sort_values(ascending=False).iloc[position - 1])


def compute_top_n_weight(weights: pd.Series, n: int = 3) -> float:
    """Return the sum of the largest ``n`` normalized absolute weights."""
    concentration_weights = _concentration_weights(weights)
    count = _validate_n(n, len(concentration_weights), allow_oversized=True)
    return float(concentration_weights.nlargest(count).sum())


def compute_weight_entropy(weights: pd.Series) -> float:
    """Return Shannon entropy of normalized absolute portfolio weights."""
    concentration_weights = _concentration_weights(weights)
    positive_weights = concentration_weights[concentration_weights > 0]
    entropy = float(-(positive_weights * np.log(positive_weights)).sum())
    return max(0.0, entropy)


def classify_concentration_risk(weights: pd.Series) -> str:
    """Classify risk using gross-normalized absolute portfolio weights."""
    effective_holdings = compute_effective_num_holdings(weights)
    top_one = compute_top_weight(weights)
    top_three = compute_top_n_weight(weights)

    if effective_holdings < 4 or top_one > 0.40:
        return "High"
    if 4 <= effective_holdings <= 8 or top_three > 0.70:
        return "Moderate"
    if effective_holdings > 8 and top_three < 0.60:
        return "Low"
    return "Moderate"


def _concentration_weights(weights: pd.Series) -> pd.Series:
    validate_weights(weights, allow_short=True)
    absolute_weights = weights.abs().astype(float)
    return absolute_weights / float(absolute_weights.sum())


def _validate_n(n: int, holdings: int, *, allow_oversized: bool) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not allow_oversized and n > holdings:
        raise ValueError(f"n cannot exceed the number of holdings ({holdings})")
    return min(int(n), holdings)
