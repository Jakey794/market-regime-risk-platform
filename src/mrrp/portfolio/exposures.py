"""Portfolio concentration and long/short exposure calculations."""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from mrrp.portfolio.weights import validate_weights


def compute_weight_exposure(weights: pd.Series) -> pd.DataFrame:
    """Build an exposure table sorted by descending absolute weight."""
    validate_weights(weights, allow_short=True)
    exposure = pd.DataFrame(
        {
            "weight": weights.astype(float),
            "absolute_weight": weights.abs().astype(float),
        }
    )
    return exposure.sort_values(
        "absolute_weight",
        ascending=False,
        kind="stable",
    )


def compute_top_n_exposure(weights: pd.Series, n: int) -> float:
    """Return the absolute exposure represented by the largest ``n`` weights."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n must be a positive integer")

    exposure = compute_weight_exposure(weights)
    return float(exposure["absolute_weight"].head(n).sum())


def compute_long_short_exposure(weights: pd.Series) -> dict[str, float]:
    """Return gross, net, long, and short portfolio exposure."""
    exposure = compute_weight_exposure(weights)["weight"]
    long_exposure = float(exposure.clip(lower=0).sum())
    short_exposure = float(-exposure.clip(upper=0).sum())
    return {
        "gross": long_exposure + short_exposure,
        "net": long_exposure - short_exposure,
        "long": long_exposure,
        "short": short_exposure,
    }
