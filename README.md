# Quant Trading Platform

An **event-driven algorithmic trading engine** built in Python. Designed for backtesting strategies on historical market data with a modular, extensible architecture.

## Architecture

```
MarketEvent → Strategy → SignalEvent → Portfolio → OrderEvent → Execution → FillEvent → Portfolio
```

All components communicate exclusively through an event queue, making the system fully decoupled and easy to extend.

## Project Structure

```
quant_trading_platform/
├── src/
│   ├── events.py       # Event class hierarchy (dataclasses)
│   ├── data.py         # DataHandler — CSV drip-feed, no look-ahead bias
│   ├── strategy.py     # MovingAverageCrossStrategy (50/200 MA)
│   ├── portfolio.py    # NaivePortfolio — position & equity tracking
│   └── execution.py    # SimulatedExecutionHandler — instant fill broker
├── data/               # CSV files (generated)
├── generate_data.py    # Synthetic OHLCV data generator (random walk)
└── main.py             # Backtest entry point & event loop
```

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib

# 2. Generate synthetic market data
python generate_data.py

# 3. Run the backtest
python main.py
```

## Key Features

- **No look-ahead bias** — generator-based data drip-feed
- **O(1) strategy windows** — `collections.deque` with `maxlen`
- **Configurable commission** — fixed ($1/trade) or variable (% of notional)
- **Equity tracking** — full position/holdings audit trail + equity curve plot
- **Type-safe** — strict type hints, frozen dataclasses, Enums throughout

## Sample Output

```
Initial Capital  : $  100,000.00
Final Equity     : $  224,789.99
Total Return     :       124.79%
Total Commission : $        6.00
```

## Tech Stack

Python 3.9+ · Pandas · NumPy · Matplotlib · dataclasses · ABC
