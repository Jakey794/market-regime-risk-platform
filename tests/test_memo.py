from __future__ import annotations

from mrrp.reporting.memo import memo_inputs_from_summary, render_quarterly_memo


def test_memo_marks_missing_values_unavailable() -> None:
    inputs = memo_inputs_from_summary(
        as_of="2026-08-05",
        portfolio_name="synthetic",
        benchmark="SPY",
        summary_cards={},
        regime_info={},
        stress_results={},
        backtest_metrics={},
        top_risk_contributors={},
        factor_proxy_exposure={},
    )
    memo = render_quarterly_memo(inputs)
    assert "unavailable" in memo
    assert ": nan" not in memo.lower()
    assert "not a prediction" not in memo.lower()
    assert "not investment advice" in memo.lower()
