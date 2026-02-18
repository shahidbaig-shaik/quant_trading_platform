"""Tests for portfolio management (portfolio.py)."""

from __future__ import annotations

import queue

import pytest

from src.events import (
    Direction,
    FillEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)
from src.portfolio import NaivePortfolio


class TestNaivePortfolio:
    """Tests for the NaivePortfolio."""

    def test_long_signal_produces_buy_order(self, data_handler, events_queue, symbol_list):
        """A LONG signal should result in a BUY OrderEvent on the queue."""
        # Advance one bar so data is available
        data_handler.update_bars()
        # Drain the MarketEvent
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
            order_quantity=100,
        )

        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=1.0)
        portfolio.update_signal(signal)

        # Should have placed an OrderEvent
        assert not events_queue.empty()
        order = events_queue.get()
        assert isinstance(order, OrderEvent)
        assert order.direction == Direction.BUY
        assert order.quantity == 100
        assert order.symbol == "AAPL"

    def test_short_signal_produces_sell_order(self, data_handler, events_queue, symbol_list):
        """A SHORT signal should result in a SELL OrderEvent."""
        data_handler.update_bars()
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
            order_quantity=50,
        )

        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.SHORT, strength=0.5)
        portfolio.update_signal(signal)

        order = events_queue.get()
        assert isinstance(order, OrderEvent)
        assert order.direction == Direction.SELL
        assert order.quantity == 50

    def test_buy_fill_updates_position(self, data_handler, events_queue, symbol_list):
        """After a BUY fill, position should increase and cash decrease."""
        data_handler.update_bars()
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
        )

        initial_cash = portfolio.current_holdings["cash"]

        fill = FillEvent(
            symbol="AAPL",
            direction=Direction.BUY,
            quantity=100,
            fill_price=150.0,
            commission=1.0,
        )
        portfolio.update_fill(fill)

        assert portfolio.current_positions["AAPL"] == 100
        assert portfolio.current_holdings["cash"] < initial_cash

    def test_sell_fill_updates_position(self, data_handler, events_queue, symbol_list):
        """After a SELL fill, position should decrease and cash increase."""
        data_handler.update_bars()
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
        )

        # First buy, then sell
        buy_fill = FillEvent(
            symbol="AAPL", direction=Direction.BUY,
            quantity=100, fill_price=150.0, commission=1.0,
        )
        portfolio.update_fill(buy_fill)
        cash_after_buy = portfolio.current_holdings["cash"]

        sell_fill = FillEvent(
            symbol="AAPL", direction=Direction.SELL,
            quantity=100, fill_price=155.0, commission=1.0,
        )
        portfolio.update_fill(sell_fill)

        assert portfolio.current_positions["AAPL"] == 0
        assert portfolio.current_holdings["cash"] > cash_after_buy

    def test_insufficient_cash_blocks_buy(self, data_handler, events_queue, symbol_list):
        """With very little capital, a BUY signal should be rejected."""
        data_handler.update_bars()
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=1.0,  # Only $1！
            order_quantity=100,
        )

        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=1.0)
        portfolio.update_signal(signal)

        # Queue should be empty — order was rejected
        assert events_queue.empty()

    def test_holdings_snapshot_recorded(self, data_handler, events_queue, symbol_list):
        """Each fill should add a snapshot to all_holdings."""
        data_handler.update_bars()
        events_queue.get()

        portfolio = NaivePortfolio(
            data_handler=data_handler,
            events_queue=events_queue,
            symbol_list=symbol_list,
            initial_capital=100_000.0,
        )

        initial_count = len(portfolio.all_holdings)

        fill = FillEvent(
            symbol="AAPL", direction=Direction.BUY,
            quantity=50, fill_price=150.0, commission=1.0,
        )
        portfolio.update_fill(fill)

        assert len(portfolio.all_holdings) == initial_count + 1
