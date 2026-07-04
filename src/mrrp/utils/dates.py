"""Date utilities for freshness and staleness checks."""

from __future__ import annotations

from datetime import date

import pandas as pd


def days_since(reference: pd.Timestamp | date, as_of: date | None = None) -> int:
    """Return the number of calendar days between ``reference`` and ``as_of``.

    ``as_of`` defaults to today. Passing it explicitly keeps callers testable
    without depending on wall-clock time.
    """
    if as_of is None:
        as_of = date.today()
    reference_date = pd.Timestamp(reference).date()
    return (as_of - reference_date).days
