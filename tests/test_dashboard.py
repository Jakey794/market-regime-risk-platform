from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mrrp.dashboard.formatting import (
    format_date_range,
    format_or_na,
    format_percentage,
    humanize_identifier,
)
from mrrp.dashboard.state import (
    BENCHMARK_KEY,
    DATE_RANGE_KEY,
    PORTFOLIO_KEY,
    get_dashboard_state,
    initialize_dashboard_state,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_shell_renders_default_page() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "Portfolio Overview"
    assert len(app.metric) == 12
    assert len(app.sidebar.selectbox) == 2
    assert len(app.sidebar.date_input) == 1
    assert len(app.sidebar.button) == 1
    assert len(app.get("plotly_chart")) == 5
    assert not app.error


def test_risk_metrics_page_renders() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.switch_page("pages/2_Risk_Metrics.py").run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "Risk Metrics"
    assert len(app.metric) == 21
    assert len(app.get("plotly_chart")) == 5
    assert len(app.get("dataframe")) == 1


def test_correlation_beta_page_renders() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.switch_page("pages/3_Correlation_Beta.py").run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "Correlation & Beta"
    assert len(app.metric) == 12
    assert len(app.get("plotly_chart")) == 6


def test_data_quality_page_renders() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.switch_page("pages/4_Data_Quality.py").run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "Data Quality"
    assert len(app.metric) == 15
    assert len(app.get("dataframe")) == 1


def test_dashboard_state_initializes_and_repairs_selections() -> None:
    session_state: dict[str, object] = {
        PORTFOLIO_KEY: "missing",
        BENCHMARK_KEY: "missing",
        DATE_RANGE_KEY: (date(1990, 1, 1), date(1990, 2, 1)),
    }

    initialize_dashboard_state(
        session_state,
        portfolios=("sample_portfolio",),
        benchmarks=("SPY", "QQQ"),
        minimum_date=date(2020, 1, 1),
        maximum_date=date(2024, 12, 31),
        default_benchmark="SPY",
    )

    assert get_dashboard_state(session_state).portfolio == "sample_portfolio"
    assert session_state[BENCHMARK_KEY] == "SPY"
    assert session_state[DATE_RANGE_KEY] == (
        date(2020, 1, 1),
        date(2024, 12, 31),
    )


def test_dashboard_formatting_is_deterministic() -> None:
    assert humanize_identifier("sample_global_portfolio") == "Sample Global Portfolio"
    assert format_date_range(date(2024, 1, 1), date(2024, 12, 31)) == (
        "2024-01-01 to 2024-12-31"
    )

    with pytest.raises(ValueError, match="must not be after"):
        format_date_range(date(2024, 2, 1), date(2024, 1, 1))


def test_format_or_na_formats_valid_values_and_falls_back_on_nan() -> None:
    assert format_or_na(0.1234, format_percentage) == "12.34%"
    assert format_or_na(float("nan"), format_percentage) == "N/A"

    with pytest.raises(ValueError, match="must be numeric"):
        format_or_na(True, format_percentage)
