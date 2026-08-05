"""Stable numeric formatting for Markdown report exports."""

from __future__ import annotations

import csv
from io import StringIO
import math
from numbers import Real
from typing import Any


_PERCENT_METRICS = frozenset(
    {
        "annualized_volatility",
        "cagr",
        "fraction_outperforming_months",
        "max_drawdown",
        "rolling_12m_return_last",
        "tracking_error",
        "transaction_cost_drag",
        "turnover",
        "worst_month",
    }
)
_INTEGER_METRICS = frozenset({"underperformance_duration_days"})
_COMPARISON_PRECISION = {
    "train_likelihood": 6,
    "aic": 6,
    "bic": 6,
    "silhouette": 4,
    "average_regime_duration": 2,
    "regime_stability": 4,
}


def format_number(value: Real, *, precision: int = 2, percent: bool = False) -> str:
    """Format a finite number at a fixed precision, or mark it unavailable."""
    numeric = float(value)
    if not math.isfinite(numeric):
        return "unavailable"
    suffix = "%" if percent else "f"
    return format(numeric, f".{precision}{suffix}")


def format_named_metric(name: str, value: Any) -> str:
    """Format a report metric according to its financial interpretation."""
    if not isinstance(value, Real):
        return str(value)
    if name in _INTEGER_METRICS:
        return format_number(value, precision=0)
    if name in _PERCENT_METRICS:
        return format_number(value, percent=True)
    return format_number(value, precision=4)


def format_comparison_csv(text: str) -> str:
    """Round model-comparison floats while preserving valid CSV quoting."""
    source = StringIO(text)
    reader = csv.DictReader(source)
    if reader.fieldnames is None:
        return text

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        for column, precision in _COMPARISON_PRECISION.items():
            value = row.get(column, "")
            if value:
                row[column] = format_number(float(value), precision=precision)
        writer.writerow(row)
    return output.getvalue().rstrip("\n")
