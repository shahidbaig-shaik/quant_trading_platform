"""
strategy.py — Strategy Engine for the Trading Engine.

Defines an abstract Strategy interface and three concrete strategies:
  1. MovingAverageCrossStrategy — golden/death cross (50/200 MA)
  2. BollingerBandStrategy      — mean reversion via Bollinger Bands
  3. RSIMomentumStrategy         — momentum via RSI crossovers

All strategies use O(1) deque-based sliding windows to avoid
recomputing over the full price history on every tick.

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


# ──────────────────────────────────────────────
#  Bollinger Band Mean Reversion Strategy
# ──────────────────────────────────────────────

class BollingerBandStrategy(Strategy):
    """
    Mean-reversion strategy using Bollinger Bands.

    Generates a **LONG** signal when the price crosses *below* the
    lower band (oversold — expect reversion up), and a **SHORT**
    signal when the price crosses *above* the upper band (overbought
    — expect reversion down).

    Bands are defined as:
        upper = SMA(close, window) + num_std × σ(close, window)
        lower = SMA(close, window) - num_std × σ(close, window)

    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler.
    symbol_list : list[str]
        Instruments to monitor.
    events_queue : queue.Queue
        Shared event queue.
    window : int
        Lookback period for the rolling mean and std (default 20).
    num_std : float
        Number of standard deviations for band width (default 2.0).
    """

    def __init__(
        self,
        data_handler,
        symbol_list: List[str],
        events_queue: queue.Queue,
        window: int = 20,
        num_std: float = 2.0,
    ) -> None:
        self.data_handler = data_handler
        self.symbol_list: List[str] = symbol_list
        self.events_queue: queue.Queue = events_queue
        self.window: int = window
        self.num_std: float = num_std

        self._price_windows: Dict[str, Deque[float]] = {
            s: deque(maxlen=window) for s in symbol_list
        }

        # Track previous position relative to bands
        # None = "not yet known", "inside" / "below" / "above"
        self._prev_zone: Dict[str, str | None] = defaultdict(lambda: None)

    def calculate_signals(self, event: MarketEvent) -> None:
        """Detect price crossing Bollinger Bands."""
        if not isinstance(event, MarketEvent):
            return

        for symbol in self.symbol_list:
            try:
                bar = self.data_handler.get_latest_bar(symbol)
            except KeyError:
                continue

            close: float = float(bar["close"])
            self._price_windows[symbol].append(close)

            window = self._price_windows[symbol]
            if len(window) < self.window:
                continue

            prices = np.array(window)
            sma = float(np.mean(prices))
            std = float(np.std(prices, ddof=1))

            if std == 0:
                continue

            upper_band = sma + self.num_std * std
            lower_band = sma - self.num_std * std

            # Determine current zone
            if close < lower_band:
                current_zone = "below"
            elif close > upper_band:
                current_zone = "above"
            else:
                current_zone = "inside"

            prev = self._prev_zone[symbol]

            # Detect crossovers
            if prev is not None:
                # Price just crossed below lower band → oversold → BUY
                if current_zone == "below" and prev != "below":
                    self.events_queue.put(SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=min(abs(close - lower_band) / std, 1.0),
                    ))
                # Price just crossed above upper band → overbought → SELL
                elif current_zone == "above" and prev != "above":
                    self.events_queue.put(SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        strength=min(abs(close - upper_band) / std, 1.0),
                    ))

            self._prev_zone[symbol] = current_zone


# ──────────────────────────────────────────────
#  RSI Momentum Strategy
# ──────────────────────────────────────────────

class RSIMomentumStrategy(Strategy):
    """
    Momentum strategy based on the Relative Strength Index (RSI).

    Generates a **LONG** signal when RSI crosses *above* the oversold
    threshold (momentum shifting bullish), and a **SHORT** signal when
    RSI crosses *below* the overbought threshold (momentum fading).

    RSI is computed using Wilder's smoothing method:
        avg_gain = prev_avg_gain × (period-1)/period + current_gain/period
        avg_loss = prev_avg_loss × (period-1)/period + current_loss/period
        RS  = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler.
    symbol_list : list[str]
        Instruments to monitor.
    events_queue : queue.Queue
        Shared event queue.
    period : int
        RSI lookback period (default 14).
    overbought : float
        Upper RSI threshold (default 70).
    oversold : float
        Lower RSI threshold (default 30).
    """

    def __init__(
        self,
        data_handler,
        symbol_list: List[str],
        events_queue: queue.Queue,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be < overbought ({overbought})."
            )

        self.data_handler = data_handler
        self.symbol_list: List[str] = symbol_list
        self.events_queue: queue.Queue = events_queue
        self.period: int = period
        self.overbought: float = overbought
        self.oversold: float = oversold

        # Store recent closes to seed the initial SMA-based RSI
        self._price_windows: Dict[str, Deque[float]] = {
            s: deque(maxlen=period + 1) for s in symbol_list
        }

        # Wilder's smoothed avg gain/loss (None until seeded)
        self._avg_gain: Dict[str, float | None] = defaultdict(lambda: None)
        self._avg_loss: Dict[str, float | None] = defaultdict(lambda: None)

        # Previous RSI for crossover detection
        self._prev_rsi: Dict[str, float | None] = defaultdict(lambda: None)

    def _compute_rsi(self, symbol: str) -> float | None:
        """Compute RSI using Wilder's smoothing method."""
        window = self._price_windows[symbol]
        if len(window) < 2:
            return None

        last_change = window[-1] - window[-2]
        gain = max(last_change, 0.0)
        loss = max(-last_change, 0.0)

        if self._avg_gain[symbol] is None:
            # Need full period to seed
            if len(window) < self.period + 1:
                return None
            # Seed with SMA of gains/losses
            changes = np.diff(list(window))
            self._avg_gain[symbol] = float(np.mean([max(c, 0) for c in changes]))
            self._avg_loss[symbol] = float(np.mean([max(-c, 0) for c in changes]))
        else:
            # Wilder's smoothing
            p = self.period
            self._avg_gain[symbol] = (
                self._avg_gain[symbol] * (p - 1) + gain
            ) / p
            self._avg_loss[symbol] = (
                self._avg_loss[symbol] * (p - 1) + loss
            ) / p

        avg_gain = self._avg_gain[symbol]
        avg_loss = self._avg_loss[symbol]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def calculate_signals(self, event: MarketEvent) -> None:
        """Detect RSI crossing overbought/oversold thresholds."""
        if not isinstance(event, MarketEvent):
            return

        for symbol in self.symbol_list:
            try:
                bar = self.data_handler.get_latest_bar(symbol)
            except KeyError:
                continue

            close: float = float(bar["close"])
            self._price_windows[symbol].append(close)

            rsi = self._compute_rsi(symbol)
            if rsi is None:
                continue

            prev_rsi = self._prev_rsi[symbol]

            if prev_rsi is not None:
                # RSI crosses ABOVE oversold → bullish momentum → BUY
                if prev_rsi <= self.oversold < rsi:
                    strength = min((rsi - self.oversold) / 20.0, 1.0)
                    self.events_queue.put(SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=strength,
                    ))
                # RSI crosses BELOW overbought → bearish momentum → SELL
                elif prev_rsi >= self.overbought > rsi:
                    strength = min((self.overbought - rsi) / 20.0, 1.0)
                    self.events_queue.put(SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        strength=strength,
                    ))

            self._prev_rsi[symbol] = rsi
