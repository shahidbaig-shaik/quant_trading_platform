"""
main.py — Backtest Entry Point for the Quant Trading Engine.

Wires together all components (DataHandler, Strategy, RiskManager,
Portfolio, ExecutionHandler) and drives the core event loop.

Supports multiple strategies via command-line arguments.

Usage:
    python main.py                          # default: MA crossover
    python main.py --strategy bollinger     # Bollinger Band mean reversion
    python main.py --strategy rsi           # RSI momentum
    python main.py --symbols AAPL MSFT --capital 200000
"""

from __future__ import annotations

import argparse
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
from src.performance import create_performance_report, print_performance_report
from src.portfolio import NaivePortfolio
from src.risk import RiskManager
from src.strategy import (
    BollingerBandStrategy,
    MovingAverageCrossStrategy,
    RSIMomentumStrategy,
)


LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-5s | %(name)-18s | %(message)s"
)


# ──────────────────────────────────────────────
#  CLI Argument Parsing
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Event-driven quant backtesting engine"
    )
    parser.add_argument(
        "--strategy",
        choices=["ma_cross", "bollinger", "rsi"],
        default="ma_cross",
        help="Strategy to backtest (default: ma_cross)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOG"],
        help="Symbols to trade (default: AAPL GOOG)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital in USD (default: 100000)",
    )
    parser.add_argument(
        "--csv-dir",
        default="data/",
        help="Directory containing CSV data files (default: data/)",
    )
    parser.add_argument(
        "--order-qty",
        type=int,
        default=100,
        help="Fixed order quantity per signal (default: 100)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.001,
        help="Slippage as fraction (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--commission-type",
        choices=["fixed", "variable"],
        default="fixed",
        help="Commission type (default: fixed)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=1.00,
        help="Commission amount — USD if fixed, fraction if variable",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.20,
        help="Max position as %% of equity (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.15,
        help="Drawdown circuit breaker threshold (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Strategy Factory
# ──────────────────────────────────────────────

def create_strategy(
    name: str,
    data_handler,
    symbol_list: List[str],
    events_queue: queue.Queue,
):
    """Instantiate the selected strategy."""
    if name == "ma_cross":
        return MovingAverageCrossStrategy(
            data_handler=data_handler,
            symbol_list=symbol_list,
            events_queue=events_queue,
            short_window=50,
            long_window=200,
        )
    elif name == "bollinger":
        return BollingerBandStrategy(
            data_handler=data_handler,
            symbol_list=symbol_list,
            events_queue=events_queue,
            window=20,
            num_std=2.0,
        )
    elif name == "rsi":
        return RSIMomentumStrategy(
            data_handler=data_handler,
            symbol_list=symbol_list,
            events_queue=events_queue,
            period=14,
            overbought=70.0,
            oversold=30.0,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


# ──────────────────────────────────────────────
#  Event Loop
# ──────────────────────────────────────────────

def run_backtest(args: argparse.Namespace) -> NaivePortfolio:
    """
    Execute the full backtest and return the final portfolio.

    Returns
    -------
    NaivePortfolio
        The portfolio object containing all historical state.
    """
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=LOG_FORMAT,
    )
    logger = logging.getLogger("BACKTEST")

    # ── Shared event queue ───────────────────────
    events_queue: queue.Queue[Event] = queue.Queue()

    # ── Component assembly ───────────────────────
    logger.info("Initialising components...")
    logger.info("  Strategy    : %s", args.strategy)
    logger.info("  Symbols     : %s", args.symbols)
    logger.info("  Capital     : $%s", f"{args.capital:,.2f}")
    logger.info("  Slippage    : %.2f%%", args.slippage * 100)
    logger.info("  Commission  : %s %s",
                args.commission_type,
                f"${args.commission:.2f}" if args.commission_type == "fixed"
                else f"{args.commission*100:.3f}%")

    data_handler = HistoricCSVDataHandler(
        csv_dir=args.csv_dir,
        symbol_list=args.symbols,
        events_queue=events_queue,
    )

    strategy = create_strategy(
        name=args.strategy,
        data_handler=data_handler,
        symbol_list=args.symbols,
        events_queue=events_queue,
    )

    portfolio = NaivePortfolio(
        data_handler=data_handler,
        events_queue=events_queue,
        symbol_list=args.symbols,
        initial_capital=args.capital,
        order_quantity=args.order_qty,
    )

    risk_manager = RiskManager(
        portfolio=portfolio,
        data_handler=data_handler,
        max_position_pct=args.max_position_pct,
        max_drawdown_pct=args.max_drawdown_pct,
    )

    execution = SimulatedExecutionHandler(
        data_handler=data_handler,
        events_queue=events_queue,
        commission_type=args.commission_type,
        commission_fixed=(
            args.commission if args.commission_type == "fixed" else 1.00
        ),
        commission_pct=(
            args.commission if args.commission_type == "variable" else 0.0005
        ),
        slippage_pct=args.slippage,
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
    blocked_count: int = 0

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
                # ── Risk check before execution ──
                from src.events import Direction
                adjusted_qty = risk_manager.size_order(
                    symbol=event.symbol,
                    raw_quantity=event.quantity,
                    direction=event.direction,
                )
                if adjusted_qty <= 0:
                    blocked_count += 1
                    continue

                # Re-create order with adjusted quantity if changed
                if adjusted_qty != event.quantity:
                    event = OrderEvent(
                        symbol=event.symbol,
                        order_type=event.order_type,
                        quantity=adjusted_qty,
                        direction=event.direction,
                    )

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
    logger.info("  Risk Blocked: %d", blocked_count)

    # ── Performance Analytics ────────────────────
    report = create_performance_report(portfolio)
    print_performance_report(report)

    # ── Equity Curve Plot ────────────────────────
    _plot_equity_curve(portfolio, report)

    return portfolio


# ──────────────────────────────────────────────
#  Post-Analysis Helpers
# ──────────────────────────────────────────────

def _plot_equity_curve(portfolio: NaivePortfolio, report: dict) -> None:
    """Plot the equity curve with drawdown subplot."""
    logger = logging.getLogger("PLOT")

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info("matplotlib not installed — skipping equity plot.")
        return

    if len(portfolio.all_holdings) < 2:
        logger.info("Not enough data points to plot equity curve.")
        return

    equity = [h["total"] for h in portfolio.all_holdings]
    equity_arr = np.array(equity)

    # Compute drawdown series
    running_max = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Equity curve ─────────────────────────────
    ax1.plot(equity, linewidth=1.2, color="#2196F3", label="Strategy Equity")
    ax1.axhline(
        y=portfolio.initial_capital,
        color="#FF5722",
        linestyle="--",
        linewidth=0.8,
        label="Initial Capital",
    )
    ax1.set_title(
        f"Equity Curve  |  Sharpe: {report['sharpe_ratio']:.2f}  "
        f"Return: {report['total_return_pct']:.1f}%  "
        f"Max DD: {report['max_drawdown_pct']:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Drawdown subplot ─────────────────────────
    ax2.fill_between(
        range(len(drawdown_pct)),
        drawdown_pct,
        0,
        color="#E53935",
        alpha=0.4,
        label="Drawdown",
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Bar #")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path: str = "data/equity_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved to: %s", output_path)


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_backtest(args)
