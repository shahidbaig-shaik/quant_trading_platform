"""
risk.py — Risk Management for the Trading Engine.

Provides a RiskManager that sits between signal generation and order
creation, enforcing position limits, volatility-based sizing, and
drawdown circuit breakers.

Usage:
    risk_mgr = RiskManager(
        portfolio=portfolio,
        data_handler=data_handler,
        max_position_pct=0.20,
        max_drawdown_pct=0.15,
    )
    adjusted_qty = risk_mgr.size_order(symbol, raw_qty, direction)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from src.events import Direction

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces risk constraints on proposed orders.

    Risk Controls
    -------------
    1. **Maximum position size** — caps any single position at
       ``max_position_pct`` of total equity.
    2. **Volatility-based sizing** — scales quantity inversely with
       recent realised volatility (higher vol → fewer shares).
    3. **Drawdown circuit breaker** — blocks all new BUY orders if
       the portfolio has drawn down more than ``max_drawdown_pct``
       from its high-water mark.

    Parameters
    ----------
    portfolio : NaivePortfolio
        Reference to the live portfolio for equity/position queries.
    data_handler : DataHandler
        Reference for fetching price history (used for vol calc).
    max_position_pct : float
        Maximum single-position value as fraction of equity (default 20 %).
    max_drawdown_pct : float
        Drawdown threshold that triggers the circuit breaker (default 15 %).
    vol_lookback : int
        Number of bars for the volatility calculation (default 20).
    vol_target : float
        Daily volatility target for sizing (default 2 %).
    """

    def __init__(
        self,
        portfolio,                           # src.portfolio.NaivePortfolio
        data_handler,                        # src.data.DataHandler
        max_position_pct: float = 0.20,
        max_drawdown_pct: float = 0.15,
        vol_lookback: int = 20,
        vol_target: float = 0.02,
    ) -> None:
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target

        # Track the high-water mark for drawdown calculations
        self._hwm: float = portfolio.initial_capital

    # ── Core API ─────────────────────────────────

    def size_order(
        self,
        symbol: str,
        raw_quantity: int,
        direction: Direction,
    ) -> int:
        """
        Apply all risk checks and return the adjusted quantity.

        Returns 0 if the order should be blocked entirely.

        Parameters
        ----------
        symbol : str
            Instrument being traded.
        raw_quantity : int
            The quantity the portfolio originally wants to trade.
        direction : Direction
            BUY or SELL.

        Returns
        -------
        int
            Adjusted (possibly reduced) quantity.  0 = order blocked.
        """
        equity = self.portfolio.current_holdings.get("total", 0.0)

        # ── 1. Drawdown circuit breaker ──────────
        if direction == Direction.BUY:
            if self._is_drawdown_breached(equity):
                logger.warning(
                    "RISK | Drawdown breaker triggered — blocking BUY %s",
                    symbol,
                )
                return 0

        # ── 2. Maximum position size ─────────────
        qty = self._cap_position_size(symbol, raw_quantity, direction, equity)
        if qty == 0:
            return 0

        # ── 3. Volatility-based sizing ───────────
        if direction == Direction.BUY:
            qty = self._vol_adjusted_size(symbol, qty)

        return max(qty, 0)

    # ── Private helpers ──────────────────────────

    def _is_drawdown_breached(self, equity: float) -> bool:
        """Check if current drawdown exceeds the allowed threshold."""
        self._hwm = max(self._hwm, equity)
        if self._hwm == 0:
            return False
        drawdown = (self._hwm - equity) / self._hwm
        return drawdown >= self.max_drawdown_pct

    def _cap_position_size(
        self,
        symbol: str,
        quantity: int,
        direction: Direction,
        equity: float,
    ) -> int:
        """Ensure position doesn't exceed max_position_pct of equity."""
        if equity <= 0:
            return 0

        try:
            bar = self.data_handler.get_latest_bar(symbol)
            price = float(bar["close"])
        except KeyError:
            return quantity  # no price data — let it through

        if price <= 0:
            return quantity

        max_value = equity * self.max_position_pct
        max_shares = int(max_value / price)

        # For BUY: limit total position (existing + new)
        if direction == Direction.BUY:
            current_pos = self.portfolio.current_positions.get(symbol, 0)
            remaining = max(max_shares - current_pos, 0)
            if quantity > remaining:
                logger.info(
                    "RISK | Position cap: %s capped %d → %d shares",
                    symbol,
                    quantity,
                    remaining,
                )
                return remaining

        return quantity

    def _vol_adjusted_size(self, symbol: str, quantity: int) -> int:
        """
        Scale position size inversely with recent realised volatility.

        If recent vol > vol_target, reduce the position proportionally.
        If recent vol < vol_target, allow full quantity (don't amplify).
        """
        try:
            bars = self.data_handler.get_latest_bars(
                symbol, self.vol_lookback
            )
        except (KeyError, ValueError):
            return quantity

        if len(bars) < 5:
            return quantity  # not enough data for reliable vol estimate

        closes = bars["close"].astype(float).values
        if len(closes) < 2:
            return quantity

        log_returns = np.diff(np.log(closes))
        realised_vol = float(np.std(log_returns, ddof=1))

        if realised_vol <= 0 or realised_vol <= self.vol_target:
            return quantity  # vol is fine — no reduction needed

        # Scale down: target_vol / realised_vol
        scale = self.vol_target / realised_vol
        adjusted = int(quantity * min(scale, 1.0))
        if adjusted != quantity:
            logger.info(
                "RISK | Vol sizing: %s vol=%.4f → scaled %d → %d shares",
                symbol,
                realised_vol,
                quantity,
                adjusted,
            )
        return max(adjusted, 1)  # always at least 1 share if not blocked
