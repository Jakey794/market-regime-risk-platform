from __future__ import annotations

from mrrp.reporting.research_report import (
    architecture_mermaid,
    default_report_inputs_from_artifacts,
    render_research_report,
)


def test_report_does_not_invent_missing_artifacts() -> None:
    report = render_research_report(default_report_inputs_from_artifacts())
    assert "not available" in report
    assert "pending full pipeline artifacts" in report
    assert "Not a prediction system" in report


def test_architecture_is_versionable_mermaid() -> None:
    diagram = architecture_mermaid()
    assert diagram.startswith("```mermaid")
    assert "mrrp.models" in diagram
    assert "Streamlit app pages" in diagram
