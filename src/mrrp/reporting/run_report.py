"""Generate conservative Markdown reports from available pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mrrp.reporting.memo_context import (
    build_memo_summary_cards,
    classify_correlation_regime_from_features,
    classify_volatility_regime_from_features,
    regime_agreement_text,
)
from mrrp.data.cache import load_parquet
from mrrp.models import (
    GMMConfig,
    GMMRegimeModel,
    ThresholdConfig,
    ThresholdRegimeModel,
)
from mrrp.portfolio.config import load_portfolio_config
from mrrp.portfolio.metadata import compute_group_exposure, load_asset_metadata
from mrrp.portfolio.returns import compute_asset_returns, compute_portfolio_returns
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


def _optional_memo_context(
    prices_path: str,
    portfolio_path: str,
    feature_raw_path: str,
    feature_scaled_path: str,
    metadata_path: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, float], dict[str, float]]:
    """Build memo context from prices/features when artifacts exist."""
    summary_cards: dict[str, Any] = {}
    regime_info: dict[str, str] = {}
    contributors: dict[str, float] = {}
    factor_exposure: dict[str, float] = {}
    prices_file = Path(prices_path)
    if not prices_file.exists():
        sample = Path("data/sample/synthetic_prices.parquet")
        prices_file = sample if sample.exists() else prices_file
    if not prices_file.exists():
        return summary_cards, regime_info, contributors, factor_exposure

    prices = load_parquet(prices_file)
    portfolio = load_portfolio_config(portfolio_path)
    asset_returns = compute_asset_returns(
        prices.loc[:, portfolio.holdings.index], method="simple"
    )
    portfolio_returns = compute_portfolio_returns(asset_returns, portfolio.holdings)
    benchmark_returns = compute_asset_returns(
        prices.loc[:, [portfolio.benchmark]], method="simple"
    )[portfolio.benchmark]
    summary_cards = build_memo_summary_cards(
        portfolio_returns,
        benchmark_returns,
        portfolio.holdings,
        asset_returns,
    )
    contributors = dict(summary_cards.get("top_risk_contributors", {}))

    try:
        metadata = load_asset_metadata(metadata_path)
        exposure = compute_group_exposure(portfolio.holdings, metadata, "factor_proxy")
        factor_exposure = {str(key): float(value) for key, value in exposure.items()}
    except (OSError, ValueError):
        pass

    raw_path = Path(feature_raw_path)
    scaled_path = Path(feature_scaled_path)
    if raw_path.exists() and scaled_path.exists():
        raw_features = load_parquet(raw_path).dropna(how="all")
        scaled = load_parquet(scaled_path).dropna()
        regime_info["volatility_regime"] = classify_volatility_regime_from_features(
            raw_features
        )
        regime_info["correlation_regime"] = classify_correlation_regime_from_features(
            raw_features
        )
        try:
            train = scaled.iloc[: max(60, int(len(scaled) * 0.7))]
            threshold = ThresholdRegimeModel(ThresholdConfig(n_states=3))
            gmm = GMMRegimeModel(
                GMMConfig(
                    n_states=3,
                    feature_columns=tuple(scaled.columns),
                    random_seed=7,
                )
            )
            threshold.fit(train)
            gmm.fit(train)
            labels = {
                "threshold": str(threshold.transform(scaled).labeled_states().iloc[-1]),
                "gmm": str(gmm.transform(scaled).labeled_states().iloc[-1]),
            }
            regime_info["agreement"] = regime_agreement_text(labels)
        except (RuntimeError, ValueError):
            pass
    return summary_cards, regime_info, contributors, factor_exposure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-comparison",
        default="data/processed/regime_models/model_comparison.csv",
    )
    parser.add_argument("--stress", default="data/processed/stress_results.json")
    parser.add_argument("--backtest", default="data/processed/backtest/summary.json")
    parser.add_argument("--prices", default="data/processed/adjusted_close.parquet")
    parser.add_argument("--portfolio", default="configs/sample_portfolio.yaml")
    parser.add_argument(
        "--feature-raw", default="data/processed/regime_features_raw.parquet"
    )
    parser.add_argument(
        "--feature-scaled", default="data/processed/regime_features_scaled.parquet"
    )
    parser.add_argument("--metadata", default="configs/asset_metadata.yaml")
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
    summary_cards, regime_info, contributors, factor_exposure = _optional_memo_context(
        args.prices,
        args.portfolio,
        args.feature_raw,
        args.feature_scaled,
        args.metadata,
    )
    portfolio = load_portfolio_config(args.portfolio)
    as_of = str(stress_payload.get("data_as_of", "unavailable"))
    if as_of == "unavailable":
        prices_file = Path(args.prices)
        if not prices_file.exists():
            prices_file = Path("data/sample/synthetic_prices.parquet")
        if prices_file.exists():
            as_of = str(load_parquet(prices_file).index.max().date())

    memo_inputs = memo_inputs_from_summary(
        as_of=as_of,
        portfolio_name=str(stress_payload.get("portfolio", portfolio.name)),
        benchmark=portfolio.benchmark,
        summary_cards=summary_cards,
        regime_info=regime_info,
        stress_results=stress,
        backtest_metrics=backtest,
        top_risk_contributors=contributors,
        factor_proxy_exposure=factor_exposure,
    )
    (out / "quarterly_memo_example.md").write_text(
        render_quarterly_memo(memo_inputs) + "\n",
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
