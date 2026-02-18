"""End-to-end integration test for the backtesting engine."""

from __future__ import annotations

import queue

import pytest

from src.data import HistoricCSVDataHandler
from src.events import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from src.execution import SimulatedExecutionHandler
from src.performance import create_performance_report
from src.portfolio import NaivePortfolio
from src.risk import RiskManager
from src.strategy import MovingAverageCrossStrategy


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_full_backtest_runs_without_errors(self, sample_csv_dir, symbol_list):
        """
        Run the full event loop on synthetic CSV data and verify
        the backtest completes without exceptions and produces
        meaningful output.
        """
        events_queue = queue.Queue()

        data_handler = HistoricCSVDataHandler(
            csv_dir=sample_csv_dir,
            symbol_list=symbol_list,
            events_queue=events_queue,
        )

        strategy = MovingAverageCrossStrategy(
            data_handler=data_handler,
            symbol_list=symbol_list,
            events_queue=events_queue,
            short_window=5,
            long_window=20,
        )

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
            order_quantity=50,
        )

        risk_manager = RiskManager(
            portfolio=portfolio,
            data_handler=data_handler,
        )

        execution = SimulatedExecutionHandler(
            data_handler=data_handler,
            events_queue=events_queue,
        )

        bar_count = 0
        fill_count = 0

        while data_handler.continue_backtest:
            data_handler.update_bars()
            bar_count += 1

            while not events_queue.empty():
                event = events_queue.get()

                if isinstance(event, MarketEvent):
                    strategy.calculate_signals(event)
                elif isinstance(event, SignalEvent):
                    portfolio.update_signal(event)
                elif isinstance(event, OrderEvent):
                    adjusted = risk_manager.size_order(
                        event.symbol, event.quantity, event.direction
                    )
                    if adjusted > 0:
                        if adjusted != event.quantity:
                            event = OrderEvent(
                                symbol=event.symbol,
                                order_type=event.order_type,
                                quantity=adjusted,
                                direction=event.direction,
                            )
                        execution.execute_order(event)
                elif isinstance(event, FillEvent):
                    fill_count += 1
                    portfolio.update_fill(event)

        # Assertions
        assert bar_count > 0, "Should have processed at least one bar"
        assert portfolio.current_holdings["total"] > 0, "Equity should be positive"
        assert len(portfolio.all_holdings) >= 1, "Should have recorded holdings"

    def test_performance_report_is_complete(self, sample_csv_dir, symbol_list):
        """
        After a backtest, the performance report should contain all
        expected keys with numeric values.
        """
        events_queue = queue.Queue()

        data_handler = HistoricCSVDataHandler(
            csv_dir=sample_csv_dir,
            symbol_list=symbol_list,
            events_queue=events_queue,
        )

        strategy = MovingAverageCrossStrategy(
            data_handler=data_handler,
            symbol_list=symbol_list,
            events_queue=events_queue,
            short_window=5,
            long_window=20,
        )

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
        )

        execution = SimulatedExecutionHandler(
            data_handler=data_handler,
            events_queue=events_queue,
        )

        while data_handler.continue_backtest:
            data_handler.update_bars()
            while not events_queue.empty():
                event = events_queue.get()
                if isinstance(event, MarketEvent):
                    strategy.calculate_signals(event)
                elif isinstance(event, SignalEvent):
                    portfolio.update_signal(event)
                elif isinstance(event, OrderEvent):
                    execution.execute_order(event)
                elif isinstance(event, FillEvent):
                    portfolio.update_fill(event)

        report = create_performance_report(portfolio)

        expected_keys = [
            "initial_capital", "final_equity", "total_return_pct",
            "annualised_return_pct", "annualised_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
            "max_drawdown_duration", "calmar_ratio", "total_trades",
            "win_rate_pct", "profit_factor", "avg_win_loss_ratio",
            "total_commission", "benchmarks_buy_hold_pct",
        ]

        for key in expected_keys:
            assert key in report, f"Missing key: {key}"

        # Numeric sanity
        assert report["initial_capital"] == 100_000.0
        assert isinstance(report["sharpe_ratio"], float)
        assert isinstance(report["max_drawdown_pct"], float)
