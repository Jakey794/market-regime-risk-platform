"""Typed, framework-light state management for dashboard controls."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from mrrp.dashboard.loaders import DashboardData


PORTFOLIO_KEY = "dashboard_selected_portfolio"
BENCHMARK_KEY = "dashboard_selected_benchmark"
DATE_RANGE_KEY = "dashboard_date_range"
DASHBOARD_DATA_KEY = "dashboard_data"


@dataclass(frozen=True)
class DashboardState:
    """Validated snapshot of the shared dashboard selections."""

    portfolio: str
    benchmark: str
    start_date: date
    end_date: date


def initialize_dashboard_state(
    session_state: MutableMapping[str, Any],
    *,
    portfolios: Sequence[str],
    benchmarks: Sequence[str],
    minimum_date: date,
    maximum_date: date,
    default_benchmark: str,
) -> None:
    """Initialize or repair shared selections against available options."""
    if not portfolios:
        raise ValueError("At least one portfolio must be available")
    if not benchmarks:
        raise ValueError("At least one benchmark must be available")
    if minimum_date > maximum_date:
        raise ValueError("minimum_date must not be after maximum_date")

    if session_state.get(PORTFOLIO_KEY) not in portfolios:
        session_state[PORTFOLIO_KEY] = portfolios[0]

    benchmark = session_state.get(BENCHMARK_KEY)
    if benchmark not in benchmarks:
        session_state[BENCHMARK_KEY] = (
            default_benchmark if default_benchmark in benchmarks else benchmarks[0]
        )

    selected_dates = session_state.get(DATE_RANGE_KEY)
    if not _valid_date_range(selected_dates, minimum_date, maximum_date):
        session_state[DATE_RANGE_KEY] = (minimum_date, maximum_date)


def get_dashboard_state(session_state: MutableMapping[str, Any]) -> DashboardState:
    """Return the current shared selections as a typed value."""
    portfolio = session_state.get(PORTFOLIO_KEY)
    benchmark = session_state.get(BENCHMARK_KEY)
    selected_dates = session_state.get(DATE_RANGE_KEY)

    if not isinstance(portfolio, str) or not portfolio:
        raise ValueError("Dashboard portfolio selection is not initialized")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("Dashboard benchmark selection is not initialized")
    if not _date_pair(selected_dates):
        raise ValueError("Dashboard date range is not initialized")

    start_date, end_date = selected_dates
    if start_date > end_date:
        raise ValueError("Dashboard start date must not be after end date")
    return DashboardState(portfolio, benchmark, start_date, end_date)


def set_dashboard_data(
    session_state: MutableMapping[str, Any],
    dashboard_data: DashboardData,
) -> None:
    """Store the validated selected dataset for use by dashboard pages."""
    if not isinstance(dashboard_data, DashboardData):
        raise ValueError("dashboard_data must be a DashboardData")
    session_state[DASHBOARD_DATA_KEY] = dashboard_data


def get_dashboard_data(session_state: MutableMapping[str, Any]) -> DashboardData:
    """Return the selected dataset stored by the dashboard entrypoint."""
    dashboard_data = session_state.get(DASHBOARD_DATA_KEY)
    if not isinstance(dashboard_data, DashboardData):
        raise ValueError("Dashboard data is not initialized")
    return dashboard_data


def _valid_date_range(
    value: object,
    minimum_date: date,
    maximum_date: date,
) -> bool:
    if not _date_pair(value):
        return False
    start_date, end_date = value
    return minimum_date <= start_date <= end_date <= maximum_date


def _date_pair(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, date) for item in value)
    )
