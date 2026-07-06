"""Risk metrics dashboard page."""

from __future__ import annotations

import streamlit as st

from mrrp.dashboard.components import (
    render_context_summary,
    render_disclaimer,
    render_metric_cards,
    render_page_header,
)
from mrrp.dashboard.formatting import format_decimal, format_or_na, format_percentage
from mrrp.dashboard.state import get_dashboard_data, get_dashboard_state
from mrrp.reporting import build_histogram_figure, build_time_series_figure
from mrrp.risk.drawdown import (
    current_drawdown,
    drawdown_series,
    drawdown_duration,
    max_drawdown,
    worst_drawdown_periods,
)
from mrrp.risk.performance import calmar_ratio, sharpe_ratio, sortino_ratio
from mrrp.risk.tail import (
    historical_cvar,
    historical_var,
    worst_return,
    worst_rolling_return,
)
from mrrp.risk.volatility import (
    annualized_return,
    annualized_volatility,
    rolling_volatility,
)


ROLLING_WINDOWS = (21, 63, 252)
LONG_WINDOW_OBSERVATIONS = 252


render_page_header(
    "Risk Metrics",
    "Historical return, volatility, drawdown, and tail-risk measures from the "
    "existing deterministic engine. This does not predict future performance.",
)

try:
    state = get_dashboard_state(st.session_state)
    data = get_dashboard_data(st.session_state)
except ValueError as exc:
    st.error(f"Dashboard inputs are unavailable: {exc}")
    st.stop()

render_context_summary(state)

if data.prices.empty or data.portfolio_returns.dropna().empty:
    st.error("The selected portfolio contains no usable observations.")
    st.stop()

returns = data.portfolio_returns
valid_observations = len(returns.dropna())
if valid_observations < LONG_WINDOW_OBSERVATIONS:
    st.warning(
        f"The selected range has {valid_observations} valid daily observations, "
        f"fewer than the {LONG_WINDOW_OBSERVATIONS} required for a full 252-day "
        "rolling window. Longer-window rolling metrics may be undefined for part "
        "or all of the selected range."
    )

try:
    latest_rolling_volatility = {
        window: rolling_volatility(returns, window=window).dropna()
        for window in ROLLING_WINDOWS
    }
    worst_drawdowns = worst_drawdown_periods(returns, top_n=5)
    return_distribution = returns.dropna()
except ValueError as exc:
    st.error(f"Unable to calculate risk metrics: {exc}")
    st.stop()


def _latest(series) -> float:
    return float(series.iloc[-1]) if not series.empty else float("nan")


st.subheader("Return and volatility")
with st.expander("What do these metrics mean?"):
    st.write(
        "Annualized return and volatility summarize the full selected period. "
        "Rolling volatility shows how volatility has evolved over shorter "
        "windows. Sharpe, Sortino, and Calmar are risk-adjusted return ratios "
        "computed over the full period."
    )
render_metric_cards(
    [
        ("Annualized return", format_percentage(annualized_return(returns))),
        ("Annualized volatility", format_percentage(annualized_volatility(returns))),
        (
            "Latest 21D volatility",
            format_or_na(_latest(latest_rolling_volatility[21]), format_percentage),
        ),
        (
            "Latest 63D volatility",
            format_or_na(_latest(latest_rolling_volatility[63]), format_percentage),
        ),
        (
            "Latest 252D volatility",
            format_or_na(_latest(latest_rolling_volatility[252]), format_percentage),
        ),
        ("Sharpe ratio", format_or_na(sharpe_ratio(returns), format_decimal)),
        ("Sortino ratio", format_or_na(sortino_ratio(returns), format_decimal)),
        (
            "Calmar ratio",
            format_or_na(calmar_ratio(returns), format_decimal),
            "Annualized return divided by maximum drawdown; higher is "
            "better, more return per unit of worst historical loss.",
        ),
    ]
)

left_column, middle_column, right_column = st.columns(3)
for column, window in zip((left_column, middle_column, right_column), ROLLING_WINDOWS):
    with column:
        st.plotly_chart(
            build_time_series_figure(
                latest_rolling_volatility[window].rename(
                    f"Rolling {window}D volatility"
                ),
                title=f"Rolling {window}-day volatility",
                yaxis_title="Annualized volatility",
                tickformat=".0%",
            ),
            width="stretch",
        )

st.subheader("Drawdown")
with st.expander("What do these metrics mean?"):
    st.write(
        "Drawdown measures the decline from the portfolio's running peak value. "
        "Current drawdown reflects the most recent observation; the table below "
        "lists the worst historical episodes ranked by depth."
    )
render_metric_cards(
    [
        ("Current drawdown", format_percentage(current_drawdown(returns))),
        ("Max drawdown", format_percentage(max_drawdown(returns))),
        (
            "Drawdown duration (periods)",
            format_decimal(drawdown_duration(returns), decimals=0),
        ),
    ]
)
if worst_drawdowns.empty:
    st.write("No drawdown episodes occurred over the selected period.")
else:
    st.dataframe(worst_drawdowns, width="stretch")
st.plotly_chart(
    build_time_series_figure(
        drawdown_series(returns).rename("Drawdown"),
        title="Drawdown",
        yaxis_title="Drawdown",
        tickformat=".0%",
    ),
    width="stretch",
)

st.subheader("Tail risk")
with st.expander("What do these metrics mean?"):
    st.write(
        "Historical VaR and CVaR describe the loss threshold and average loss "
        "beyond that threshold at the given confidence level, based on observed "
        "returns. Worst-period figures show the single worst daily, weekly, and "
        "monthly compounded returns in the selected range."
    )
render_metric_cards(
    [
        ("VaR 95", format_percentage(historical_var(returns, confidence=0.95))),
        ("VaR 99", format_percentage(historical_var(returns, confidence=0.99))),
        ("CVaR 95", format_percentage(historical_cvar(returns, confidence=0.95))),
        (
            "CVaR 99",
            format_percentage(historical_cvar(returns, confidence=0.99)),
            "Average daily return on the worst 1% of historical days; more "
            "sensitive to tail risk than VaR.",
        ),
        ("Worst daily return", format_percentage(worst_return(returns))),
        (
            "Worst weekly return",
            format_or_na(worst_rolling_return(returns, window=5), format_percentage),
        ),
        (
            "Worst monthly return",
            format_or_na(worst_rolling_return(returns, window=21), format_percentage),
        ),
    ]
)
st.plotly_chart(
    build_histogram_figure(
        return_distribution,
        title="Daily return distribution",
        xaxis_title="Daily return",
        tickformat=".1%",
    ),
    width="stretch",
)

render_disclaimer()
