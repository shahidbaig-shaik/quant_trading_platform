"""Shared pytest fixtures for the trading engine tests."""

from __future__ import annotations

import os
import queue
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data import HistoricCSVDataHandler
from src.events import (
    Direction,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)
from src.portfolio import NaivePortfolio


@pytest.fixture
def events_queue():
    """Return a fresh event queue."""
    return queue.Queue()


@pytest.fixture
def sample_csv_dir(tmp_path):
    """
    Create a temp directory with small synthetic CSV files.

    Generates 100 bars of simple data for AAPL and GOOG.
    """
    np.random.seed(42)

    for symbol, start_price in [("AAPL", 150.0), ("GOOG", 2800.0)]:
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        returns = np.random.normal(0, 0.02, 100)
        closes = start_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            "datetime": dates,
            "open": closes * (1 + np.random.normal(0, 0.002, 100)),
            "high": closes * (1 + abs(np.random.normal(0, 0.005, 100))),
            "low": closes * (1 - abs(np.random.normal(0, 0.005, 100))),
            "close": closes,
            "volume": np.random.randint(100000, 5000000, 100),
        })
        df.to_csv(tmp_path / f"{symbol}.csv", index=False)

    return str(tmp_path)


@pytest.fixture
def symbol_list():
    """Default symbol list."""
    return ["AAPL", "GOOG"]


@pytest.fixture
def data_handler(sample_csv_dir, symbol_list, events_queue):
    """Create a pre-loaded data handler."""
    return HistoricCSVDataHandler(
        csv_dir=sample_csv_dir,
        symbol_list=symbol_list,
        events_queue=events_queue,
    )


@pytest.fixture
def loaded_data_handler(data_handler):
    """A data handler that has already consumed all bars."""
    while data_handler.continue_backtest:
        data_handler.update_bars()
    return data_handler


@pytest.fixture
def portfolio(data_handler, events_queue, symbol_list):
    """Create a fresh portfolio."""
    return NaivePortfolio(
        data_handler=data_handler,
        events_queue=events_queue,
        symbol_list=symbol_list,
        initial_capital=100_000.0,
        order_quantity=100,
    )
