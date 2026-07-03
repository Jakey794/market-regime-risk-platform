"""Deterministic display formatting for dashboard values."""

from __future__ import annotations

from datetime import date
from numbers import Real


def humanize_identifier(value: str) -> str:
    """Convert a machine-oriented identifier into a display label."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    return value.strip().replace("_", " ").title()


def format_date_range(start_date: date, end_date: date) -> str:
    """Format an inclusive dashboard date range."""
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise ValueError("start_date and end_date must be dates")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    return f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"


def format_percentage(value: Real, decimals: int = 2) -> str:
    """Format a decimal value as a percentage."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be numeric")
    return f"{float(value):.{decimals}%}"


def format_decimal(value: Real, decimals: int = 2) -> str:
    """Format a numeric value to a fixed number of decimal places."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be numeric")
    return f"{float(value):.{decimals}f}"
