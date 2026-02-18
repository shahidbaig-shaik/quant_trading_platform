"""
performance.py — Performance Analytics for the Trading Engine.

Computes industry-standard risk-adjusted return metrics from
portfolio equity and trade history.

Metrics:
    - Annualised Return & Volatility
    - Sharpe Ratio & Sortino Ratio
    - Maximum Drawdown (depth + duration)
    - Calmar Ratio
    - Win Rate, Profit Factor, Avg Win / Avg Loss
    - Benchmark comparison (buy-and-hold)

Usage:
    report = create_performance_report(portfolio)
    print_performance_report(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Annualisation factor (trading days per year)
TRADING_DAYS_PER_YEAR: int = 252


# ──────────────────────────────────────────────
#  Core Metric Functions
# ──────────────────────────────────────────────

def annualised_return(equity_curve: np.ndarray) -> float:
    """
    Compute the compound annual growth rate (CAGR).

    Parameters
    ----------
    equity_curve : np.ndarray
        Daily equity values (length >= 2).

    Returns
    -------
    float
        Annualised return as a decimal (e.g. 0.12 = 12 %).
    """
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    total_return = equity_curve[-1] / equity_curve[0]
    n_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if n_years == 0:
        return 0.0
    return float(total_return ** (1.0 / n_years) - 1.0)


def annualised_volatility(equity_curve: np.ndarray) -> float:
    """
    Annualised standard deviation of daily log returns.

    Parameters
    ----------
    equity_curve : np.ndarray
        Daily equity values (length >= 2).

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    if len(equity_curve) < 2:
        return 0.0
    # Daily log returns
    log_returns = np.diff(np.log(equity_curve))
    return float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    equity_curve : np.ndarray
        Daily equity values.
    risk_free_rate : float
        Annual risk-free rate (default 2 %).

    Returns
    -------
    float
        Sharpe ratio.  Returns 0.0 if volatility is zero.
    """
    vol = annualised_volatility(equity_curve)
    if vol == 0:
        return 0.0
    ann_ret = annualised_return(equity_curve)
    return float((ann_ret - risk_free_rate) / vol)


def sortino_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Annualised Sortino ratio (penalises only downside deviation).

    Parameters
    ----------
    equity_curve : np.ndarray
        Daily equity values.
    risk_free_rate : float
        Annual risk-free rate (default 2 %).

    Returns
    -------
    float
        Sortino ratio.  Returns 0.0 if downside deviation is zero.
    """
    if len(equity_curve) < 2:
        return 0.0

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = float(np.std(downside, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std == 0:
        return 0.0

    ann_ret = annualised_return(equity_curve)
    return float((ann_ret - risk_free_rate) / downside_std)


@dataclass
class DrawdownResult:
    """Container for drawdown analysis results."""
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # in bars / days
    drawdown_series: np.ndarray = field(default_factory=lambda: np.array([]))


def max_drawdown(equity_curve: np.ndarray) -> DrawdownResult:
    """
    Compute maximum drawdown (peak-to-trough decline).

    Parameters
    ----------
    equity_curve : np.ndarray
        Daily equity values.

    Returns
    -------
    DrawdownResult
        Contains max drawdown %, max duration, and full drawdown series.
    """
    if len(equity_curve) < 2:
        return DrawdownResult()

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max

    max_dd_pct = float(np.min(drawdowns))  # most negative = deepest

    # Duration: longest consecutive period below previous peak
    is_in_drawdown = drawdowns < 0
    max_duration = 0
    current_duration = 0
    for flag in is_in_drawdown:
        if flag:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return DrawdownResult(
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration=max_duration,
        drawdown_series=drawdowns,
    )


def calmar_ratio(equity_curve: np.ndarray) -> float:
    """
    Calmar ratio = annualised return / |max drawdown|.

    Returns 0.0 if max drawdown is zero.
    """
    dd = max_drawdown(equity_curve)
    if dd.max_drawdown_pct == 0:
        return 0.0
    ann_ret = annualised_return(equity_curve)
    return float(ann_ret / abs(dd.max_drawdown_pct))


# ──────────────────────────────────────────────
#  Trade-Level Metrics
# ──────────────────────────────────────────────

def _extract_round_trip_pnls(portfolio) -> List[float]:
    """
    Extract approximate per-trade P&L from the portfolio's holding
    snapshots by detecting equity changes around fill events.

    This is a simplified approach — in a production system you'd
    track explicit round-trip trades.
    """
    if len(portfolio.all_holdings) < 2:
        return []

    pnls: List[float] = []
    for i in range(1, len(portfolio.all_holdings)):
        delta = (
            portfolio.all_holdings[i]["total"]
            - portfolio.all_holdings[i - 1]["total"]
        )
        if delta != 0:
            pnls.append(delta)
    return pnls


def win_rate(pnls: List[float]) -> float:
    """Fraction of trades with positive P&L."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: List[float]) -> float:
    """Gross profit / gross loss.  Returns inf if no losses."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_win_loss_ratio(pnls: List[float]) -> float:
    """Average winning trade / average losing trade (absolute)."""
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    if not wins or not losses:
        return 0.0
    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))


# ──────────────────────────────────────────────
#  Benchmark Comparison
# ──────────────────────────────────────────────

def buy_and_hold_return(
    data_handler,
    symbol: str,
    initial_capital: float,
) -> float:
    """
    Compute the buy-and-hold return for a single symbol.

    Assumes buying as many shares as possible at the first bar's
    close, and holding until the last bar's close.
    """
    bars = data_handler._latest_symbol_data.get(symbol, [])
    if len(bars) < 2:
        return 0.0
    first_close = float(bars[0]["close"])
    last_close = float(bars[-1]["close"])
    if first_close == 0:
        return 0.0
    shares = int(initial_capital / first_close)
    if shares == 0:
        return 0.0
    invested = shares * first_close
    final_value = shares * last_close + (initial_capital - invested)
    return (final_value - initial_capital) / initial_capital


# ──────────────────────────────────────────────
#  Report Generator
# ──────────────────────────────────────────────

def create_performance_report(
    portfolio,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Generate a comprehensive performance report from a completed
    backtest portfolio.

    Parameters
    ----------
    portfolio : NaivePortfolio
        The portfolio after the backtest has finished.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino (default 2 %).

    Returns
    -------
    dict
        Dictionary of all computed metrics.
    """
    equity = np.array([h["total"] for h in portfolio.all_holdings])
    pnls = _extract_round_trip_pnls(portfolio)
    dd = max_drawdown(equity)

    # Benchmark
    benchmarks: Dict[str, float] = {}
    for sym in portfolio.symbol_list:
        bh = buy_and_hold_return(
            portfolio.data_handler, sym, portfolio.initial_capital
        )
        benchmarks[sym] = bh

    report: Dict[str, Any] = {
        # Equity-curve metrics
        "initial_capital": portfolio.initial_capital,
        "final_equity": float(equity[-1]) if len(equity) > 0 else 0.0,
        "total_return_pct": (
            (float(equity[-1]) - portfolio.initial_capital)
            / portfolio.initial_capital
            * 100.0
            if len(equity) > 0
            else 0.0
        ),
        "annualised_return_pct": annualised_return(equity) * 100.0,
        "annualised_volatility_pct": annualised_volatility(equity) * 100.0,
        "sharpe_ratio": sharpe_ratio(equity, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity, risk_free_rate),
        "max_drawdown_pct": dd.max_drawdown_pct * 100.0,
        "max_drawdown_duration": dd.max_drawdown_duration,
        "calmar_ratio": calmar_ratio(equity),
        # Trade-level metrics
        "total_trades": len(pnls),
        "win_rate_pct": win_rate(pnls) * 100.0,
        "profit_factor": profit_factor(pnls),
        "avg_win_loss_ratio": avg_win_loss_ratio(pnls),
        "total_commission": portfolio.current_holdings.get("commission", 0.0),
        # Benchmark
        "benchmarks_buy_hold_pct": {
            s: v * 100.0 for s, v in benchmarks.items()
        },
    }
    return report


def print_performance_report(report: Dict[str, Any]) -> None:
    """Pretty-print a performance report to the logger."""
    log = logging.getLogger("PERFORMANCE")

    log.info("=" * 62)
    log.info("  PERFORMANCE REPORT")
    log.info("=" * 62)
    log.info("  Initial Capital      : $%s", f"{report['initial_capital']:>12,.2f}")
    log.info("  Final Equity         : $%s", f"{report['final_equity']:>12,.2f}")
    log.info("  Total Return         : %11.2f%%", report["total_return_pct"])
    log.info("  Annualised Return    : %11.2f%%", report["annualised_return_pct"])
    log.info("  Annualised Volatility: %11.2f%%", report["annualised_volatility_pct"])
    log.info("-" * 62)
    log.info("  Sharpe Ratio         : %11.4f", report["sharpe_ratio"])
    log.info("  Sortino Ratio        : %11.4f", report["sortino_ratio"])
    log.info("  Calmar Ratio         : %11.4f", report["calmar_ratio"])
    log.info("-" * 62)
    log.info("  Max Drawdown         : %11.2f%%", report["max_drawdown_pct"])
    log.info("  Max DD Duration      : %8d bars", report["max_drawdown_duration"])
    log.info("-" * 62)
    log.info("  Total Trades         : %8d", report["total_trades"])
    log.info("  Win Rate             : %11.2f%%", report["win_rate_pct"])
    log.info("  Profit Factor        : %11.4f", report["profit_factor"])
    log.info("  Avg Win/Loss Ratio   : %11.4f", report["avg_win_loss_ratio"])
    log.info("  Total Commission     : $%s", f"{report['total_commission']:>12,.2f}")
    log.info("-" * 62)
    log.info("  Benchmark (Buy & Hold):")
    for sym, ret in report["benchmarks_buy_hold_pct"].items():
        log.info("    %-6s             : %11.2f%%", sym, ret)
    log.info("=" * 62)

