"""
data.py — Data Handlers for the Trading Engine.

Provides an abstract DataHandler interface and a concrete
HistoricCSVDataHandler that drip-feeds bars one at a time via
a Python generator, eliminating look-ahead bias in backtests.

Usage:
    handler = HistoricCSVDataHandler(
        csv_dir="data/",
        symbol_list=["AAPL", "GOOG"],
        events_queue=queue.Queue(),
    )
    handler.update_bars()          # pushes the next bar + MarketEvent
    bar = handler.get_latest_bar("AAPL")
"""

from __future__ import annotations

import os
import queue
from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd

from src.events import MarketEvent


# ──────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────

class DataHandler(ABC):
    """
    Interface that all data handlers (live, historic) must implement.

    Subclasses guarantee that downstream consumers can only ever see
    data that has already "arrived", preventing look-ahead bias.
    """

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """
        Return the most recently observed bar for *symbol*.

        Returns
        -------
        pd.Series
            A single OHLCV row with index:
            [datetime, open, high, low, close, volume].

        Raises
        ------
        KeyError
            If *symbol* has not yet produced any bars.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Return the last *n* bars for *symbol* as a DataFrame.

        Parameters
        ----------
        symbol : str
            Ticker / instrument identifier.
        n : int
            Number of most-recent bars to return (default 1).

        Returns
        -------
        pd.DataFrame
            Up to *n* rows of OHLCV data (fewer if not enough bars
            have arrived yet).
        """
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> None:
        """
        Push the next bar for every tracked symbol onto the internal
        latest-data structure, then place a `MarketEvent` on the queue.

        Raises
        ------
        StopIteration
            (handled internally) — when data is exhausted the handler
            sets a ``continue_backtest`` flag to ``False``.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────
#  Historic CSV Implementation
# ──────────────────────────────────────────────

class HistoricCSVDataHandler(DataHandler):
    """
    Read OHLCV data from CSV files and replay it bar-by-bar.

    Each CSV is expected to have columns:
        datetime, open, high, low, close, volume
    with ``datetime`` parseable by ``pd.to_datetime``.

    Parameters
    ----------
    csv_dir : str
        Directory containing one ``<SYMBOL>.csv`` per instrument.
    symbol_list : list[str]
        Symbols to track (file names without extension).
    events_queue : queue.Queue
        Shared queue for publishing MarketEvent instances.
    """

    def __init__(
        self,
        csv_dir: str,
        symbol_list: List[str],
        events_queue: queue.Queue,  # type: ignore[type-arg]
    ) -> None:
        self.csv_dir: str = csv_dir
        self.symbol_list: List[str] = symbol_list
        self.events_queue: queue.Queue = events_queue  # type: ignore[type-arg]

        # Full DataFrames (never exposed directly — no look-ahead)
        self._symbol_data: Dict[str, pd.DataFrame] = {}

        # Generator per symbol — yields one bar at a time
        self._symbol_generators: Dict[str, Generator] = {}

        # Bars that have "arrived" so far (append-only)
        self._latest_symbol_data: Dict[str, List[pd.Series]] = {}

        # Flag to signal the event loop that data has been exhausted
        self.continue_backtest: bool = True

        self._load_csv_files()

    # ── private helpers ──────────────────────────

    def _load_csv_files(self) -> None:
        """Load each CSV into a DataFrame and initialise its generator."""
        for symbol in self.symbol_list:
            path = os.path.join(self.csv_dir, f"{symbol}.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"CSV file not found for symbol '{symbol}': {path}"
                )

            df = pd.read_csv(
                path,
                header=0,
                index_col=None,
                parse_dates=["datetime"],
                names=["datetime", "open", "high", "low", "close", "volume"],
            )
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            self._symbol_data[symbol] = df
            self._latest_symbol_data[symbol] = []
            self._symbol_generators[symbol] = self._bar_generator(symbol)

    def _bar_generator(
        self, symbol: str
    ) -> Generator[pd.Series, None, None]:
        """Yield one bar (row) at a time from the stored DataFrame."""
        for _, row in self._symbol_data[symbol].iterrows():
            yield row

    # ── public API ───────────────────────────────

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Return the single most-recent bar for *symbol*."""
        bars = self._latest_symbol_data.get(symbol)
        if not bars:
            raise KeyError(
                f"No bars available yet for symbol '{symbol}'."
            )
        return bars[-1]

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last *n* bars for *symbol* as a DataFrame."""
        bars = self._latest_symbol_data.get(symbol)
        if not bars:
            raise KeyError(
                f"No bars available yet for symbol '{symbol}'."
            )
        return pd.DataFrame(bars[-n:])

    def update_bars(self) -> None:
        """
        Advance every symbol's generator by one step.

        For each symbol, the next bar is appended to the "arrived"
        list.  After all symbols are updated a single `MarketEvent`
        is placed on the queue.

        If *any* symbol's data is exhausted, `continue_backtest` is
        set to ``False`` so the event loop knows to stop.
        """
        for symbol in self.symbol_list:
            try:
                bar: pd.Series = next(self._symbol_generators[symbol])
                self._latest_symbol_data[symbol].append(bar)
            except StopIteration:
                self.continue_backtest = False
                return  # don't fire a MarketEvent for a partial update

        # All symbols advanced successfully — notify the engine
        self.events_queue.put(MarketEvent())
