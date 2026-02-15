"""
portfolio.py — Portfolio & Position Management for the Trading Engine.

Provides an abstract Portfolio interface and a concrete NaivePortfolio
that converts signals to fixed-size orders and tracks positions,
holdings, and equity over time.

Event Flow:
    SignalEvent  →  Portfolio.update_signal()  →  OrderEvent
    FillEvent    →  Portfolio.update_fill()    →  (state update)
"""

from __future__ import annotations

import logging
import queue
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

from src.events import (
    Direction,
    FillEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────

class Portfolio(ABC):
    """
    Interface for portfolio management objects.

    A Portfolio consumes `SignalEvent` and `FillEvent` objects,
    manages positions and cash, and emits `OrderEvent` objects.
    """

    @abstractmethod
    def update_signal(self, event: SignalEvent) -> None:
        """
        React to a new SignalEvent by generating an OrderEvent.

        Parameters
        ----------
        event : SignalEvent
            Signal indicating a potential trade opportunity.
        """
        raise NotImplementedError

    @abstractmethod
    def update_fill(self, event: FillEvent) -> None:
        """
        React to a FillEvent by updating positions and holdings.

        Parameters
        ----------
        event : FillEvent
            Confirmation that an order has been executed.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────
#  Naive Portfolio Implementation
# ──────────────────────────────────────────────

class NaivePortfolio(Portfolio):
    """
    A simple portfolio that buys/sells a fixed quantity of shares
    for every signal, without sophisticated position sizing.

    Parameters
    ----------
    data_handler : DataHandler
        Reference for current market prices (used for equity snapshots).
    events_queue : queue.Queue
        Shared event queue for publishing OrderEvent instances.
    symbol_list : list[str]
        Instruments managed by this portfolio.
    initial_capital : float
        Starting cash balance in USD (default $100,000).
    order_quantity : int
        Fixed number of shares per order (default 100).
    """

    def __init__(
        self,
        data_handler,                       # src.data.DataHandler
        events_queue: queue.Queue,          # type: ignore[type-arg]
        symbol_list: List[str],
        initial_capital: float = 100_000.0,
        order_quantity: int = 100,
    ) -> None:
        self.data_handler = data_handler
        self.events_queue: queue.Queue = events_queue  # type: ignore[type-arg]
        self.symbol_list: List[str] = symbol_list
        self.initial_capital: float = initial_capital
        self.order_quantity: int = order_quantity

        # ── Position tracking ────────────────────
        # current_positions: symbol → signed quantity (+long / −short)
        self.current_positions: Dict[str, int] = {
            s: 0 for s in symbol_list
        }

        # Historical snapshots (one dict per update)
        self.all_positions: List[Dict[str, Any]] = [
            self._snapshot_positions(datetime.utcnow())
        ]

        # ── Holdings tracking ────────────────────
        # current_holdings stores per-symbol market value + cash + equity
        self.current_holdings: Dict[str, float] = self._init_holdings()

        # Historical snapshots
        self.all_holdings: List[Dict[str, float]] = [
            deepcopy(self.current_holdings)
        ]

    # ── initialisation helpers ───────────────────

    def _snapshot_positions(self, dt: datetime) -> Dict[str, Any]:
        """Return a timestamped copy of current positions."""
        snap: Dict[str, Any] = {"datetime": dt}
        snap.update(deepcopy(self.current_positions))
        return snap

    def _init_holdings(self) -> Dict[str, float]:
        """Build the initial holdings dictionary."""
        holdings: Dict[str, float] = {"datetime": datetime.utcnow()}  # type: ignore[assignment]
        for s in self.symbol_list:
            holdings[s] = 0.0          # market value of position
        holdings["cash"] = self.initial_capital
        holdings["commission"] = 0.0
        holdings["total"] = self.initial_capital
        return holdings

    # ── signal → order ───────────────────────────

    def update_signal(self, event: SignalEvent) -> None:
        """
        Convert a SignalEvent into an OrderEvent.

        Naive logic:
        - LONG  signal → BUY  ``order_quantity`` shares
        - SHORT signal → SELL ``order_quantity`` shares

        A basic cash check prevents buying when funds are insufficient.
        """
        if not isinstance(event, SignalEvent):
            return

        symbol: str = event.symbol
        direction: Direction

        if event.signal_type == SignalType.LONG:
            direction = Direction.BUY
        elif event.signal_type == SignalType.SHORT:
            direction = Direction.SELL
        else:
            logger.warning("Unknown signal type: %s", event.signal_type)
            return

        # ── Simple cash check for BUY orders ─────
        if direction == Direction.BUY:
            try:
                bar = self.data_handler.get_latest_bar(symbol)
                estimated_cost = float(bar["close"]) * self.order_quantity
                if self.current_holdings["cash"] < estimated_cost:
                    logger.warning(
                        "Insufficient cash (%.2f) to buy %d x %s @ %.2f",
                        self.current_holdings["cash"],
                        self.order_quantity,
                        symbol,
                        float(bar["close"]),
                    )
                    return
            except KeyError:
                pass  # no bar yet — let execution handler decide

        order = OrderEvent(
            symbol=symbol,
            order_type=OrderType.MARKET,
            quantity=self.order_quantity,
            direction=direction,
        )
        self.events_queue.put(order)
        logger.info(
            "ORDER %s %d %s (signal=%s, strength=%.2f)",
            direction.value,
            self.order_quantity,
            symbol,
            event.signal_type.value,
            event.strength,
        )

    # ── fill → state update ──────────────────────

    def update_fill(self, event: FillEvent) -> None:
        """
        Update positions and holdings after a fill.

        - BUY  → increase position, decrease cash
        - SELL → decrease position, increase cash
        """
        if not isinstance(event, FillEvent):
            return

        fill_dir: int = 1 if event.direction == Direction.BUY else -1
        fill_cost: float = event.fill_price * event.quantity

        # ── Update positions ─────────────────────
        self.current_positions[event.symbol] += fill_dir * event.quantity

        # ── Update holdings ──────────────────────
        self.current_holdings[event.symbol] += fill_dir * fill_cost
        self.current_holdings["commission"] += event.commission
        self.current_holdings["cash"] -= (fill_dir * fill_cost) + event.commission

        # Recalculate total equity
        self._update_total_equity()

        # ── Record snapshots ─────────────────────
        now = datetime.utcnow()
        self.all_positions.append(self._snapshot_positions(now))
        self.all_holdings.append(deepcopy(self.current_holdings))

        logger.info(
            "FILL  %s %d %s @ %.4f | pos=%d | cash=%.2f | equity=%.2f",
            event.direction.value,
            event.quantity,
            event.symbol,
            event.fill_price,
            self.current_positions[event.symbol],
            self.current_holdings["cash"],
            self.current_holdings["total"],
        )

    # ── equity helpers ───────────────────────────

    def _update_total_equity(self) -> None:
        """Recalculate total equity = cash + Σ(position market values)."""
        total: float = self.current_holdings["cash"]
        for s in self.symbol_list:
            # Use latest bar price to mark-to-market
            try:
                bar = self.data_handler.get_latest_bar(s)
                market_value = self.current_positions[s] * float(bar["close"])
                self.current_holdings[s] = market_value
            except KeyError:
                pass  # no data yet — keep existing value
            total += self.current_holdings[s]
        self.current_holdings["total"] = total
