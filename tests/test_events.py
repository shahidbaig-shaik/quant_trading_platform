"""Tests for the event class hierarchy (events.py)."""

from __future__ import annotations

import pytest
from datetime import datetime

from src.events import (
    Direction,
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)


class TestMarketEvent:
    """Tests for MarketEvent."""

    def test_creation(self):
        event = MarketEvent()
        assert event.event_type == "MarketEvent"
        assert isinstance(event.timestamp, datetime)

    def test_frozen(self):
        """MarketEvent dataclass should be immutable."""
        event = MarketEvent()
        with pytest.raises(AttributeError):
            event.timestamp = datetime.utcnow()


class TestSignalEvent:
    """Tests for SignalEvent."""

    def test_valid_creation(self):
        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=0.8)
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.LONG
        assert signal.strength == 0.8

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SignalEvent(symbol="", signal_type=SignalType.LONG)

    def test_strength_below_zero_raises(self):
        with pytest.raises(ValueError, match="strength"):
            SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=-0.1)

    def test_strength_above_one_raises(self):
        with pytest.raises(ValueError, match="strength"):
            SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=1.5)

    def test_strength_boundary_zero(self):
        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=0.0)
        assert signal.strength == 0.0

    def test_strength_boundary_one(self):
        signal = SignalEvent(symbol="AAPL", signal_type=SignalType.LONG, strength=1.0)
        assert signal.strength == 1.0


class TestOrderEvent:
    """Tests for OrderEvent."""

    def test_valid_creation(self):
        order = OrderEvent(
            symbol="GOOG",
            order_type=OrderType.MARKET,
            quantity=50,
            direction=Direction.BUY,
        )
        assert order.symbol == "GOOG"
        assert order.quantity == 50
        assert order.direction == Direction.BUY

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            OrderEvent(symbol="", quantity=10, direction=Direction.BUY)

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderEvent(symbol="AAPL", quantity=0, direction=Direction.BUY)

    def test_negative_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderEvent(symbol="AAPL", quantity=-5, direction=Direction.BUY)

    def test_repr_format(self):
        order = OrderEvent(symbol="AAPL", quantity=100, direction=Direction.SELL)
        repr_str = repr(order)
        assert "AAPL" in repr_str
        assert "SELL" in repr_str


class TestFillEvent:
    """Tests for FillEvent."""

    def test_valid_creation(self):
        fill = FillEvent(
            symbol="AAPL",
            direction=Direction.BUY,
            quantity=100,
            fill_price=150.50,
            commission=1.00,
        )
        assert fill.symbol == "AAPL"
        assert fill.fill_price == 150.50

    def test_cost_property(self):
        """cost = fill_price × quantity + commission."""
        fill = FillEvent(
            symbol="AAPL",
            direction=Direction.BUY,
            quantity=100,
            fill_price=150.00,
            commission=1.00,
        )
        assert fill.cost == 15001.00

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="negative"):
            FillEvent(
                symbol="AAPL", direction=Direction.BUY,
                quantity=10, fill_price=-1.0, commission=0.0,
            )

    def test_negative_commission_raises(self):
        with pytest.raises(ValueError, match="negative"):
            FillEvent(
                symbol="AAPL", direction=Direction.BUY,
                quantity=10, fill_price=100.0, commission=-1.0,
            )

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            FillEvent(symbol="", direction=Direction.BUY, quantity=10, fill_price=100.0)

    def test_frozen(self):
        fill = FillEvent(
            symbol="AAPL", direction=Direction.BUY,
            quantity=10, fill_price=100.0, commission=1.0,
        )
        with pytest.raises(AttributeError):
            fill.fill_price = 200.0
