"""Tests for strategy signal generation (strategy.py)."""

from __future__ import annotations

import queue
from collections import deque

import numpy as np
import pandas as pd
import pytest

from src.events import MarketEvent, SignalEvent, SignalType
from src.strategy import (
    BollingerBandStrategy,
    MovingAverageCrossStrategy,
    RSIMomentumStrategy,
)


# ──────────────────────────────────────────────
#  Helper: minimal mock data handler
# ──────────────────────────────────────────────

class MockDataHandler:
    """
    Minimal data handler that serves pre-loaded bars one at a time.
    Allows fine-grained control of what price the strategy sees.
    """

    def __init__(self, prices: dict[str, list[float]]):
        self._prices = prices
        self._index: dict[str, int] = {s: -1 for s in prices}

    def advance(self):
        """Move to the next bar for all symbols."""
        for s in self._prices:
            self._index[s] += 1

    def get_latest_bar(self, symbol: str) -> pd.Series:
        idx = self._index[symbol]
        if idx < 0 or idx >= len(self._prices[symbol]):
            raise KeyError(f"No data for {symbol}")
        price = self._prices[symbol][idx]
        return pd.Series({
            "datetime": pd.Timestamp("2023-01-01") + pd.Timedelta(days=idx),
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000,
        })

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        idx = self._index[symbol]
        if idx < 0:
            raise KeyError(f"No data for {symbol}")
        start = max(0, idx - n + 1)
        rows = []
        for i in range(start, idx + 1):
            p = self._prices[symbol][i]
            rows.append({
                "datetime": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                "open": p, "high": p * 1.01, "low": p * 0.99,
                "close": p, "volume": 1_000_000,
            })
        return pd.DataFrame(rows)


def _drain_signals(eq: queue.Queue) -> list[SignalEvent]:
    """Extract all SignalEvent instances from the queue."""
    signals = []
    while not eq.empty():
        event = eq.get()
        if isinstance(event, SignalEvent):
            signals.append(event)
    return signals


# ──────────────────────────────────────────────
#  MovingAverageCrossStrategy Tests
# ──────────────────────────────────────────────

class TestMovingAverageCross:
    """Tests for the MA cross strategy."""

    def test_no_signal_before_window_fills(self):
        """No signals should fire before we have `long_window` bars."""
        prices = {"TEST": [100.0 + i * 0.1 for i in range(50)]}
        handler = MockDataHandler(prices)
        eq = queue.Queue()

        strategy = MovingAverageCrossStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            short_window=5,
            long_window=20,
        )

        # Feed only 19 bars (one less than long_window)
        for _ in range(19):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        signals = _drain_signals(eq)
        assert len(signals) == 0

    def test_golden_cross_produces_long_signal(self):
        """
        Construct a price series where the short MA crosses above
        the long MA, and verify a LONG signal is emitted.
        """
        # Flat low prices → sudden jump → short MA rises above long MA
        prices = [100.0] * 20 + [100.0 + i * 2 for i in range(15)]
        handler = MockDataHandler({"TEST": prices})
        eq = queue.Queue()

        strategy = MovingAverageCrossStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            short_window=5,
            long_window=20,
        )

        for _ in range(len(prices)):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        signals = _drain_signals(eq)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_invalid_windows_raises(self):
        """short_window >= long_window should raise ValueError."""
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(
                data_handler=MockDataHandler({"X": [1.0]}),
                symbol_list=["X"],
                events_queue=queue.Queue(),
                short_window=20,
                long_window=10,
            )


# ──────────────────────────────────────────────
#  BollingerBandStrategy Tests
# ──────────────────────────────────────────────

class TestBollingerBand:
    """Tests for the Bollinger Band strategy."""

    def test_no_signal_before_window_fills(self):
        prices = {"TEST": [100.0] * 10}
        handler = MockDataHandler(prices)
        eq = queue.Queue()

        strategy = BollingerBandStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            window=20,
        )

        for _ in range(10):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        assert _drain_signals(eq) == []

    def test_oversold_produces_long_signal(self):
        """Price dropping sharply below the lower band → LONG signal."""
        # Need prices with some variation (std > 0), then a sharp drop
        np.random.seed(99)
        base_prices = list(100.0 + np.random.normal(0, 0.5, 25))
        prices = base_prices + [85.0]  # 85 is well below lower band
        handler = MockDataHandler({"TEST": prices})
        eq = queue.Queue()

        strategy = BollingerBandStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            window=20,
            num_std=2.0,
        )

        for _ in range(len(prices)):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        signals = _drain_signals(eq)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_overbought_produces_short_signal(self):
        """Price spiking above the upper band → SHORT signal."""
        np.random.seed(99)
        base_prices = list(100.0 + np.random.normal(0, 0.5, 25))
        prices = base_prices + [115.0]  # 115 is well above upper band
        handler = MockDataHandler({"TEST": prices})
        eq = queue.Queue()

        strategy = BollingerBandStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            window=20,
            num_std=2.0,
        )

        for _ in range(len(prices)):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        signals = _drain_signals(eq)
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) >= 1


# ──────────────────────────────────────────────
#  RSIMomentumStrategy Tests
# ──────────────────────────────────────────────

class TestRSIMomentum:
    """Tests for the RSI momentum strategy."""

    def test_no_signal_before_period_fills(self):
        prices = {"TEST": [100.0] * 10}
        handler = MockDataHandler(prices)
        eq = queue.Queue()

        strategy = RSIMomentumStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            period=14,
        )

        for _ in range(10):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        assert _drain_signals(eq) == []

    def test_oversold_recovery_produces_long(self):
        """
        Construct a series that drives RSI below 30 then recovers,
        which should trigger a LONG signal.
        """
        # Strong decline to push RSI low, then recovery
        prices = [100.0]
        for i in range(20):
            prices.append(prices[-1] * 0.98)  # decline
        for i in range(10):
            prices.append(prices[-1] * 1.03)  # recovery

        handler = MockDataHandler({"TEST": prices})
        eq = queue.Queue()

        strategy = RSIMomentumStrategy(
            data_handler=handler,
            symbol_list=["TEST"],
            events_queue=eq,
            period=14,
            oversold=30.0,
            overbought=70.0,
        )

        for _ in range(len(prices)):
            handler.advance()
            strategy.calculate_signals(MarketEvent())

        signals = _drain_signals(eq)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_invalid_thresholds_raises(self):
        """oversold >= overbought should raise ValueError."""
        with pytest.raises(ValueError):
            RSIMomentumStrategy(
                data_handler=MockDataHandler({"X": [1.0]}),
                symbol_list=["X"],
                events_queue=queue.Queue(),
                oversold=70.0,
                overbought=30.0,
            )
