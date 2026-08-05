"""Deterministic quarterly portfolio-risk memo generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mrrp.reporting.formatting import format_named_metric, format_number


@dataclass(frozen=True)
class MemoInputs:
    """Computed inputs required to render a quarterly memo."""

    as_of: str
    portfolio_name: str
    benchmark: str
    volatility_regime: str
    correlation_regime: str
    regime_agreement: str
    drawdown: float
    var_95: float
    cvar_95: float
    benchmark_beta: float
    concentration_label: str
    top_risk_contributors: Mapping[str, float]
    factor_proxy_exposure: Mapping[str, float]
    stress_results: Mapping[str, float]
    backtest_metrics: Mapping[str, float]
    methodology_notes: tuple[str, ...]
    limitations: tuple[str, ...]


def render_quarterly_memo(inputs: MemoInputs) -> str:
    """Render a Markdown quarterly risk memo from computed outputs only."""
    stress_lines = (
        "\n".join(
            f"- {name}: {format_number(value, percent=True)}"
            for name, value in inputs.stress_results.items()
        )
        or "- No stress results supplied."
    )
    backtest_lines = (
        "\n".join(
            f"- {name}: {format_named_metric(name, value)}"
            for name, value in inputs.backtest_metrics.items()
        )
        or "- No backtest metrics supplied."
    )
    contrib_lines = (
        "\n".join(
            f"- {name}: {format_number(value, percent=True)}"
            for name, value in inputs.top_risk_contributors.items()
        )
        or "- No contribution data supplied."
    )
    factor_lines = (
        "\n".join(
            f"- {name}: {format_number(value, percent=True)}"
            for name, value in inputs.factor_proxy_exposure.items()
        )
        or "- No factor-proxy exposures supplied."
    )
    methodology = "\n".join(f"- {item}" for item in inputs.methodology_notes)
    limitations = "\n".join(f"- {item}" for item in inputs.limitations)

    return f"""# Quarterly Portfolio Risk Memo

## Executive summary

As of **{inputs.as_of}**, the **{inputs.portfolio_name}** portfolio is reviewed
against benchmark **{inputs.benchmark}** for risk monitoring and research. This
memo summarises volatility/correlation regimes, drawdown and tail risk,
concentration, stress tests, and historical backtest evidence. It does **not**
provide buy/sell instructions or personalised financial advice.

## Data as-of date

{inputs.as_of}

## Current volatility regime

{inputs.volatility_regime}

## Current correlation regime

{inputs.correlation_regime}

## Regime-model agreement / disagreement

{inputs.regime_agreement}

## Drawdown and tail risk

- Current / recent drawdown context: {format_number(inputs.drawdown, percent=True)}
- Historical VaR (95%): {format_number(inputs.var_95, percent=True)}
- Historical CVaR (95%): {format_number(inputs.cvar_95, percent=True)}

## Benchmark beta

Portfolio beta vs {inputs.benchmark}: **{format_number(inputs.benchmark_beta)}**

## Concentration and risk contribution

Concentration label: **{inputs.concentration_label}**

Top risk contributors:

{contrib_lines}

## Factor / sector proxy exposure

These are ETF sector/factor **proxies**, not a statistical factor model.

{factor_lines}

## Stress-test results

{stress_lines}

## Backtest evidence

Historical simulation metrics for research comparison only. They are not a
guarantee of future performance or alpha.

{backtest_lines}

## Review considerations

- Revisit defensive sleeves if high-volatility and high-correlation regimes agree.
- Check whether concentration and risk contribution remain aligned with policy.
- Confirm stress coverage limitations when ETF history is shorter than a scenario.
- Treat backtest out/under-performance as descriptive evidence, not a trading signal.

## Methodology

{methodology}

## Limitations

{limitations}

---
Generated deterministically from platform outputs. Not investment advice.
"""


def memo_inputs_from_summary(
    *,
    as_of: str,
    portfolio_name: str,
    benchmark: str,
    summary_cards: Mapping[str, Any],
    regime_info: Mapping[str, str],
    stress_results: Mapping[str, float],
    backtest_metrics: Mapping[str, float],
    top_risk_contributors: Mapping[str, float],
    factor_proxy_exposure: Mapping[str, float],
) -> MemoInputs:
    """Convenience builder from dashboard/summary dictionaries."""
    return MemoInputs(
        as_of=as_of,
        portfolio_name=portfolio_name,
        benchmark=benchmark,
        volatility_regime=str(regime_info.get("volatility_regime", "unavailable")),
        correlation_regime=str(regime_info.get("correlation_regime", "unavailable")),
        regime_agreement=str(regime_info.get("agreement", "unavailable")),
        drawdown=float(summary_cards.get("drawdown", float("nan"))),
        var_95=float(summary_cards.get("var_95", float("nan"))),
        cvar_95=float(summary_cards.get("cvar_95", float("nan"))),
        benchmark_beta=float(summary_cards.get("beta", float("nan"))),
        concentration_label=str(summary_cards.get("concentration", "unavailable")),
        top_risk_contributors=top_risk_contributors,
        factor_proxy_exposure=factor_proxy_exposure,
        stress_results=stress_results,
        backtest_metrics=backtest_metrics,
        methodology_notes=(
            "Trailing risk metrics and regimes use information available at each date.",
            "Scalers and models are fit on training periods only.",
            "Stress tests disclose historical replay vs approximation methodologies.",
            "Backtests shift signals to avoid same-close execution.",
        ),
        limitations=(
            "ETF data may be incomplete, revised, or shorter than requested scenarios.",
            "Regime labels are descriptive risk states, not forecasts.",
            "Backtests can overfit research choices and ignore market impact beyond modelled costs.",
            "This memo is not personalised financial advice.",
        ),
    )


def write_memo_markdown(path: str, markdown: str) -> None:
    """Persist a memo as UTF-8 Markdown."""
    from pathlib import Path

    Path(path).write_text(
        markdown if markdown.endswith("\n") else markdown + "\n",
        encoding="utf-8",
    )
