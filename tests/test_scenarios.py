from __future__ import annotations

import pandas as pd

from mrrp.risk.scenarios import (
    DeterministicShockScenario,
    HistoricalWindowScenario,
    window_coverage,
)


def test_window_coverage_discloses_partial_history() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 101.0]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    start, end, covered, warning = window_coverage(
        prices,
        "2019-01-01",
        "2020-01-02",
    )
    assert (start, end) == (prices.index[0], prices.index[-1])
    assert not covered
    assert warning and "partially covers" in warning


def test_scenario_dataclasses_preserve_explicit_inputs() -> None:
    historical = HistoricalWindowScenario("event", "2020-01-01", "2020-02-01", "test")
    deterministic = DeterministicShockScenario(
        "shock",
        "deterministic_asset_shock",
        {"A": -0.1},
        "test",
        {},
    )
    assert historical.start == "2020-01-01"
    assert deterministic.shocks == {"A": -0.1}
