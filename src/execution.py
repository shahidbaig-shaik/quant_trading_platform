"""
execution.py — Execution Handlers for the Trading Engine.

Provides an abstract ExecutionHandler interface and a concrete
SimulatedExecutionHandler that fills orders instantly at the
latest market price with configurable commission.

Event Flow:
    OrderEvent  →  ExecutionHandler.execute_order()  →  FillEvent
"""

from __future__ import annotations

import logging
import queue
from abc import ABC, abstractmethod
from datetime import datetime

from src.events import Direction, FillEvent, OrderEvent


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────

class ExecutionHandler(ABC):
    """
    Interface for all execution handlers (simulated or live).

    An ExecutionHandler receives `OrderEvent` objects, executes them
    against a market (real or simulated), and emits `FillEvent`
    objects back onto the queue.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """
        Process an incoming order.

        Parameters
        ----------
        event : OrderEvent
            The order to execute.

        Side Effects
        ------------
        Places a `FillEvent` on ``events_queue`` upon successful fill.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────
#  Simulated Execution Handler
# ──────────────────────────────────────────────

class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated broker that fills orders at the latest close price
    adjusted for configurable slippage and commission.

    Slippage moves the fill price adversely:
        BUY  → close × (1 + slippage_pct)
        SELL → close × (1 - slippage_pct)

    Parameters
    ----------
    data_handler : DataHandler
        Reference to the data handler for fetching the latest price.
    events_queue : queue.Queue
        Shared event queue for publishing FillEvent instances.
    commission_type : str
        ``"fixed"`` for a flat per-trade fee, or ``"variable"``
        for a percentage-of-value fee.  Default ``"fixed"``.
    commission_fixed : float
        Flat fee per trade in USD (default $1.00).
        Used only when ``commission_type == "fixed"``.
    commission_pct : float
        Fee as a fraction of trade value (default 0.0005 = 0.05 %).
        Used only when ``commission_type == "variable"``.
    slippage_pct : float
        Adverse price impact as a fraction (default 0.001 = 0.1 %).
    """

    def __init__(
        self,
        data_handler,                       # src.data.DataHandler
        events_queue: queue.Queue,          # type: ignore[type-arg]
        commission_type: str = "fixed",
        commission_fixed: float = 1.00,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.001,
    ) -> None:
        if commission_type not in ("fixed", "variable"):
            raise ValueError(
                f"commission_type must be 'fixed' or 'variable', "
                f"got '{commission_type}'."
            )

        self.data_handler = data_handler
        self.events_queue: queue.Queue = events_queue  # type: ignore[type-arg]
        self.commission_type: str = commission_type
        self.commission_fixed: float = commission_fixed
        self.commission_pct: float = commission_pct
        self.slippage_pct: float = slippage_pct

    # ── helpers ──────────────────────────────────

    def _calculate_commission(
        self, fill_price: float, quantity: int
    ) -> float:
        """Return the commission for a single fill."""
        if self.commission_type == "fixed":
            return self.commission_fixed
        # variable: percentage of notional value
        return fill_price * quantity * self.commission_pct

    # ── core logic ───────────────────────────────

    def execute_order(self, event: OrderEvent) -> None:
        """
        Simulate an instant fill at the latest close price.

        Steps
        -----
        1. Look up the latest bar for the order's symbol.
        2. Use the close price as the fill price.
        3. Compute commission.
        4. Create and enqueue a `FillEvent`.
        """
        if not isinstance(event, OrderEvent):
            return

        # Fetch latest close from the data handler
        try:
            bar = self.data_handler.get_latest_bar(event.symbol)
        except KeyError:
            logger.warning(
                "Cannot fill order for '%s': no market data available.",
                event.symbol,
            )
            return

        raw_price: float = float(bar["close"])

        # Apply slippage: adverse price movement
        if event.direction == Direction.BUY:
            fill_price = raw_price * (1.0 + self.slippage_pct)
        else:
            fill_price = raw_price * (1.0 - self.slippage_pct)

        commission: float = self._calculate_commission(
            fill_price, event.quantity
        )

        fill = FillEvent(
            symbol=event.symbol,
            direction=event.direction,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
        )

        self.events_queue.put(fill)
        logger.info(
            "FILL  %s %s %d @ %.4f (slip %.4f)  comm=%.2f  cost=%.2f",
            event.direction.name,
            event.symbol,
            event.quantity,
            fill_price,
            fill_price - raw_price,
            commission,
            fill.cost,
        )
