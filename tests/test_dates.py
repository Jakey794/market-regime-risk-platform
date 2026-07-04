from __future__ import annotations

from datetime import date

import pandas as pd

from mrrp.utils.dates import days_since


def test_days_since_computes_calendar_days() -> None:
    assert days_since(date(2024, 1, 1), as_of=date(2024, 1, 8)) == 7


def test_days_since_is_zero_for_same_day() -> None:
    assert days_since(date(2024, 1, 1), as_of=date(2024, 1, 1)) == 0


def test_days_since_accepts_timestamp_reference() -> None:
    assert days_since(pd.Timestamp("2024-01-01"), as_of=date(2024, 1, 10)) == 9
