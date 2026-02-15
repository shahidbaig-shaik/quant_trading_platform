"""
strategy.py — Strategy Engine for the Trading Engine.

Defines an abstract Strategy interface and a concrete
MovingAverageCrossStrategy that detects golden/death crosses
using O(1) deque-based sliding windows.

Event Flow:
    MarketEvent  →  Strategy.calculate_signals()  →  SignalEvent
"""

from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque, Dict, List

import numpy as np

from src.events import MarketEvent, SignalEvent, SignalType


# ──────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────

class Strategy(ABC):
    """
    Interface that all strategy objects must implement.

    A Strategy consumes `MarketEvent` instances, performs its
    calculations, and — when a trading opportunity is detected —
    pushes a `SignalEvent` onto the shared events queue.
    """

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        React to a new MarketEvent.

        Parameters
        ----------
        event : MarketEvent
            The market event indicating new data is available.

        Side Effects
        ------------
        May place one or more `SignalEvent` objects onto ``events_queue``.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────
#  Moving Average Crossover Strategy
# ──────────────────────────────────────────────

class MovingAverageCrossStrategy(Strategy):
    """
    Classic dual moving-average crossover strategy.

    Generates a **LONG** signal when the short-window MA crosses
    *above* the long-window MA (golden cross), and a **SHORT**
    signal when it crosses *below* (death cross).

    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler (used to fetch latest bars).
    symbol_list : list[str]
        Instruments to monitor.
    events_queue : queue.Queue
        Shared event queue for publishing signals.
    short_window : int
        Lookback period for the fast MA (default 50).
    long_window : int
        Lookback period for the slow MA (default 200).
    """

    def __init__(
        self,
        data_handler,                       # src.data.DataHandler
        symbol_list: List[str],
        events_queue: queue.Queue,          # type: ignore[type-arg]
        short_window: int = 50,
        long_window: int = 200,
    ) -> None:
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < "
                f"long_window ({long_window})."
            )

        self.data_handler = data_handler
        self.symbol_list: List[str] = symbol_list
        self.events_queue: queue.Queue = events_queue  # type: ignore[type-arg]
        self.short_window: int = short_window
        self.long_window: int = long_window

        # O(1) append / pop sliding window per symbol (maxlen = long_window)
        self._price_windows: Dict[str, Deque[float]] = {
            s: deque(maxlen=long_window) for s in symbol_list
        }

        # Track whether the short MA was above the long MA on the
        # *previous* bar so we can detect crossovers, not just levels.
        # ``None`` means "not yet determined" (need first valid reading).
        self._short_above_long: Dict[str, bool | None] = defaultdict(
            lambda: None
        )

    # ── core logic ───────────────────────────────

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Called on every MarketEvent.

        For each symbol:
        1.  Append the latest close to the deque.
        2.  If enough data has accumulated (``len == long_window``):
            a.  Compute the short and long simple moving averages.
            b.  Detect crossover vs. the previous state.
            c.  Emit a `SignalEvent` if a cross occurred.
        """
        if not isinstance(event, MarketEvent):
            return

        for symbol in self.symbol_list:
            # --- 1. Fetch & store latest close ---
            try:
                bar = self.data_handler.get_latest_bar(symbol)
            except KeyError:
                continue  # no data yet for this symbol

            close: float = float(bar["close"])
            self._price_windows[symbol].append(close)

            window: Deque[float] = self._price_windows[symbol]

            # --- 2. Guard: need a full long window ---
            if len(window) < self.long_window:
                continue

            # --- 3. Compute MAs ---
            prices = np.array(window)
            short_ma: float = float(np.mean(prices[-self.short_window:]))
            long_ma: float = float(np.mean(prices))  # full window = long MA

            short_is_above: bool = short_ma > long_ma
            prev_state: bool | None = self._short_above_long[symbol]

            # --- 4. Detect crossovers ---
            if prev_state is not None and short_is_above != prev_state:
                if short_is_above:
                    # Golden cross → BUY signal
                    signal = SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=1.0,
                    )
                else:
                    # Death cross → SELL signal
                    signal = SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        strength=1.0,
                    )
                self.events_queue.put(signal)

            # --- 5. Update state for next tick ---
            self._short_above_long[symbol] = short_is_above
