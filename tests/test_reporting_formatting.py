from __future__ import annotations

from mrrp.reporting.formatting import (
    format_comparison_csv,
    format_named_metric,
    format_number,
)


def test_comparison_csv_rounds_model_diagnostics_stably() -> None:
    header = "model,train_likelihood,aic,silhouette,regime_stability\n"
    first = header + "hmm,-24059.92100721148,,0.3124412815,0.9809024134"
    second = header + "hmm,-24059.921007212124,,0.3124412816,0.9809024133"

    expected = header + "hmm,-24059.921007,,0.3124,0.9809"
    assert format_comparison_csv(first) == expected
    assert format_comparison_csv(second) == expected


def test_named_metrics_use_percent_ratio_and_integer_formats() -> None:
    assert format_named_metric("cagr", 0.0986104927) == "9.86%"
    assert format_named_metric("sharpe", 0.5145934609) == "0.5146"
    assert format_named_metric("underperformance_duration_days", 4002) == "4002"


def test_non_finite_numbers_are_unavailable() -> None:
    assert format_number(float("nan"), percent=True) == "unavailable"
