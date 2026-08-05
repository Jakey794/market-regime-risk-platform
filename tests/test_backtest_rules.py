from __future__ import annotations

import pandas as pd
import pytest

from mrrp.backtest.rules import (
    RuleConfig,
    compute_signal_scalar,
    target_weights_from_scalar,
)


def _config(name: str = "combined_risk_aware") -> RuleConfig:
    return RuleConfig(
        name=name,
        base_weights=pd.Series({"A": 0.7, "B": 0.3}),
        defensive_weights=pd.Series({"A": 0.2, "B": 0.8}),
        min_risk_scalar=0.25,
    )


def test_combined_rule_uses_only_same_row_trailing_features() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    features = pd.DataFrame(
        {
            "portfolio_vol_63d": [0.1, 0.3, 0.1],
            "mean_corr_63d": [0.2, 0.8, 0.2],
            "portfolio_drawdown": [0.0, -0.2, 0.0],
        },
        index=index,
    )
    scalar = compute_signal_scalar(features, _config())
    assert scalar.iloc[0] == 1.0
    assert scalar.iloc[1] == 0.25


def test_target_weights_sum_to_one_and_invalid_weights_fail() -> None:
    target = target_weights_from_scalar(0.5, _config())
    assert target.sum() == pytest.approx(1.0)
    bad = RuleConfig(
        name="static_benchmark",
        base_weights=pd.Series({"A": 0.8}),
        defensive_weights=pd.Series({"A": 1.0}),
    )
    with pytest.raises(ValueError, match="sum to 1"):
        target_weights_from_scalar(1.0, bad)
