"""
main.py — Backtest Entry Point for the Quant Trading Engine.

Wires together all components (DataHandler, Strategy, Portfolio,
ExecutionHandler) and drives the core event loop.

Usage:
    python main.py
"""

from __future__ import annotations

import logging
import queue
import sys
import time
from datetime import datetime
from typing import List

from src.data import HistoricCSVDataHandler
from src.events import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from src.execution import SimulatedExecutionHandler
from src.portfolio import NaivePortfolio
from src.strategy import MovingAverageCrossStrategy


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

CSV_DIR: str = "data/"
SYMBOL_LIST: List[str] = ["AAPL", "GOOG"]
INITIAL_CAPITAL: float = 100_000.0
SHORT_WINDOW: int = 50
LONG_WINDOW: int = 200
ORDER_QUANTITY: int = 100

LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-5s | %(name)-18s | %(message)s"
)
LOG_LEVEL: int = logging.INFO


# ──────────────────────────────────────────────
#  Event Loop
# ──────────────────────────────────────────────

def run_backtest() -> NaivePortfolio:
    """
    Execute the full backtest and return the final portfolio.

    Returns
    -------
    NaivePortfolio
        The portfolio object containing all historical state.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("BACKTEST")

    # ── Shared event queue ───────────────────────
    events_queue: queue.Queue[Event] = queue.Queue()

    # ── Component assembly ───────────────────────
    logger.info("Initialising components...")
    logger.info("  Symbols     : %s", SYMBOL_LIST)
    logger.info("  Capital     : $%s", f"{INITIAL_CAPITAL:,.2f}")
    logger.info("  MA Windows  : %d / %d", SHORT_WINDOW, LONG_WINDOW)

    data_handler = HistoricCSVDataHandler(
        csv_dir=CSV_DIR,
        symbol_list=SYMBOL_LIST,
        events_queue=events_queue,
    )

    strategy = MovingAverageCrossStrategy(
        data_handler=data_handler,
        symbol_list=SYMBOL_LIST,
        events_queue=events_queue,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW,
    )

    portfolio = NaivePortfolio(
        data_handler=data_handler,
        events_queue=events_queue,
        symbol_list=SYMBOL_LIST,
        initial_capital=INITIAL_CAPITAL,
        order_quantity=ORDER_QUANTITY,
    )

    execution = SimulatedExecutionHandler(
        data_handler=data_handler,
        events_queue=events_queue,
        commission_type="fixed",
        commission_fixed=1.00,
    )

    # ── Main event loop ──────────────────────────
    logger.info("=" * 60)
    logger.info("BACKTEST STARTED")
    logger.info("=" * 60)

    start_time: float = time.time()
    bar_count: int = 0
    signal_count: int = 0
    order_count: int = 0
    fill_count: int = 0

    while data_handler.continue_backtest:
        # Drip the next bar for all symbols
        data_handler.update_bars()
        bar_count += 1

        # Process every event that was generated
        while not events_queue.empty():
            event: Event = events_queue.get()

            if isinstance(event, MarketEvent):
                strategy.calculate_signals(event)

            elif isinstance(event, SignalEvent):
                signal_count += 1
                portfolio.update_signal(event)

            elif isinstance(event, OrderEvent):
                order_count += 1
                execution.execute_order(event)

            elif isinstance(event, FillEvent):
                fill_count += 1
                portfolio.update_fill(event)

    elapsed: float = time.time() - start_time

    # ── Post-analysis ────────────────────────────
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info("  Duration    : %.2f seconds", elapsed)
    logger.info("  Bars        : %d", bar_count)
    logger.info("  Signals     : %d", signal_count)
    logger.info("  Orders      : %d", order_count)
    logger.info("  Fills       : %d", fill_count)

    _print_results(portfolio)
    _plot_equity_curve(portfolio)

    return portfolio


# ──────────────────────────────────────────────
#  Post-Analysis Helpers
# ──────────────────────────────────────────────

def _print_results(portfolio: NaivePortfolio) -> None:
    """Print a summary of portfolio performance."""
    logger = logging.getLogger("RESULTS")

    final_equity: float = portfolio.current_holdings["total"]
    total_return: float = (
        (final_equity - portfolio.initial_capital)
        / portfolio.initial_capital
    ) * 100.0
    total_commission: float = portfolio.current_holdings["commission"]

    logger.info("-" * 60)
    logger.info("  Initial Capital  : $%s", f"{portfolio.initial_capital:>12,.2f}")
    logger.info("  Final Equity     : $%s", f"{final_equity:>12,.2f}")
    logger.info("  Total Return     :  %11.2f%%", total_return)
    logger.info("  Total Commission : $%s", f"{total_commission:>12,.2f}")
    logger.info("-" * 60)

    # Per-symbol positions
    logger.info("  Final Positions:")
    for symbol in portfolio.symbol_list:
        pos: int = portfolio.current_positions[symbol]
        val: float = portfolio.current_holdings.get(symbol, 0.0)
        logger.info("    %-6s : %6d shares  ($%s)", symbol, pos, f"{val:,.2f}")

    logger.info("  Cash Remaining   : $%s", f"{portfolio.current_holdings['cash']:>12,.2f}")


def _plot_equity_curve(portfolio: NaivePortfolio) -> None:
    """Plot the equity curve if matplotlib is available."""
    logger = logging.getLogger("PLOT")

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed — skipping equity plot.")
        return

    if len(portfolio.all_holdings) < 2:
        logger.info("Not enough data points to plot equity curve.")
        return

    equity: List[float] = [h["total"] for h in portfolio.all_holdings]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.2, color="#2196F3", label="Total Equity")
    ax.axhline(
        y=portfolio.initial_capital,
        color="#FF5722",
        linestyle="--",
        linewidth=0.8,
        label="Initial Capital",
    )
    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path: str = "data/equity_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved to: %s", output_path)


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest()
