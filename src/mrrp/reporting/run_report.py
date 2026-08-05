"""Generate conservative Markdown reports from available pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mrrp.reporting.memo import memo_inputs_from_summary, render_quarterly_memo
from mrrp.reporting.research_report import (
    architecture_mermaid,
    default_report_inputs_from_artifacts,
    render_research_report,
)


def _load_summary(path: str) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.exists():
        return {}
    value = json.loads(artifact.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-comparison",
        default="data/processed/regime_models/model_comparison.csv",
    )
    parser.add_argument("--stress", default="data/processed/stress_results.json")
    parser.add_argument("--backtest", default="data/processed/backtest/summary.json")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stress_payload = _load_summary(args.stress)
    stress = {
        str(item["name"]): float(item["portfolio_impact"])
        for item in stress_payload.get("results", [])
        if isinstance(item, dict) and item.get("portfolio_impact") is not None
    }
    backtest_payload = _load_summary(args.backtest)
    first_backtest = next(iter(backtest_payload.values()), {})
    backtest = (
        first_backtest.get("metrics", {}) if isinstance(first_backtest, dict) else {}
    )

    memo_inputs = memo_inputs_from_summary(
        as_of=str(stress_payload.get("data_as_of", "unavailable")),
        portfolio_name=str(stress_payload.get("portfolio", "sample portfolio")),
        benchmark="SPY",
        summary_cards={},
        regime_info={},
        stress_results=stress,
        backtest_metrics=backtest,
        top_risk_contributors={},
        factor_proxy_exposure={},
    )
    (out / "quarterly_memo_example.md").write_text(
        render_quarterly_memo(memo_inputs),
        encoding="utf-8",
    )
    report_inputs = default_report_inputs_from_artifacts(
        comparison_csv=args.model_comparison,
        stress_summary=stress,
        backtest_summary=backtest,
    )
    (out / "final_research_report.md").write_text(
        render_research_report(report_inputs),
        encoding="utf-8",
    )
    (out / "architecture.md").write_text(
        "# Architecture\n\n" + architecture_mermaid() + "\n",
        encoding="utf-8",
    )
    print(f"Wrote memo, research report, and architecture to {out}")


if __name__ == "__main__":
    main()
