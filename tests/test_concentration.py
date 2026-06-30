from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.portfolio import compute_group_exposure, load_asset_metadata
from mrrp.risk import (
    classify_concentration_risk,
    compute_effective_num_holdings,
    compute_hhi,
    compute_top_n_weight,
    compute_top_weight,
    compute_weight_entropy,
)


def test_hhi_known_weights() -> None:
    weights = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})

    assert compute_hhi(weights) == pytest.approx(0.38)


def test_hhi_uses_gross_normalized_absolute_short_weights() -> None:
    weights = pd.Series({"LONG": 1.20, "SHORT": -0.20})

    assert compute_hhi(weights) == pytest.approx((6 / 7) ** 2 + (1 / 7) ** 2)


def test_effective_num_holdings_known_weights() -> None:
    weights = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})

    assert compute_effective_num_holdings(weights) == pytest.approx(1 / 0.38)


def test_top_3_weight() -> None:
    weights = pd.Series({"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10})

    assert compute_top_n_weight(weights, n=3) == pytest.approx(0.90)
    assert compute_top_weight(weights) == pytest.approx(0.40)


def test_weight_entropy_nonnegative() -> None:
    weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})

    assert compute_weight_entropy(weights) == pytest.approx(np.log(4))
    assert compute_weight_entropy(weights) >= 0


def test_concentration_classification_high() -> None:
    weights = pd.Series({"A": 0.50, "B": 0.20, "C": 0.15, "D": 0.15})

    assert classify_concentration_risk(weights) == "High"


def test_concentration_classification_moderate() -> None:
    weights = pd.Series({ticker: 0.20 for ticker in "ABCDE"})

    assert classify_concentration_risk(weights) == "Moderate"


def test_concentration_classification_low() -> None:
    weights = pd.Series({f"A{i}": 0.10 for i in range(10)})

    assert classify_concentration_risk(weights) == "Low"


def test_load_asset_metadata() -> None:
    metadata = load_asset_metadata("configs/asset_metadata.yaml")

    assert set(metadata) == {"SPY", "QQQ", "XIU.TO", "EFA", "EEM", "XLK"}
    assert metadata["SPY"]["asset_class"] == "Equity"
    assert metadata["XIU.TO"]["region"] == "Canada"
    assert metadata["XLK"]["sector_proxy"] == "Technology"


def test_group_exposure_sums_to_one() -> None:
    metadata = load_asset_metadata("configs/asset_metadata.yaml")
    weights = pd.Series(
        {"SPY": 0.35, "QQQ": 0.15, "XIU.TO": 0.20, "EFA": 0.30}
    )

    result = compute_group_exposure(weights, metadata, group_key="region")

    assert result.sum() == pytest.approx(1.0)
    assert result.index.tolist() == [
        "United States",
        "Developed ex North America",
        "Canada",
    ]


def test_group_exposure_unknown_metadata() -> None:
    metadata = load_asset_metadata("configs/asset_metadata.yaml")
    weights = pd.Series({"SPY": 0.60, "MISSING": 0.40})

    result = compute_group_exposure(weights, metadata, group_key="region")

    assert result["Unknown"] == pytest.approx(0.40)
    assert result.sum() == pytest.approx(1.0)
