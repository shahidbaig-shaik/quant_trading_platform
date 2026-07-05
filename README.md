# Quant Trading Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![pandas](https://img.shields.io/badge/pandas-2.0-150458?logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-1.25-013243?logo=numpy) ![License](https://img.shields.io/badge/license-MIT-green)

> Event-driven algorithmic trading engine with backtesting, strategy execution, and P&L reporting.

## Overview

Discretionary trading is vulnerable to emotion and inconsistency. This platform implements a fully systematic, event-driven trading engine that executes a moving-average crossover strategy, simulates order fills, tracks positions, and produces detailed P&L reports — all without manual intervention. The modular architecture supports plugging in alternative strategies with minimal code changes.

## Tech Stack

| Component | Technology |
|---|---|
| Core Engine | Python (event-driven loop) |
| Data Layer | pandas, NumPy |
| Data Ingestion | `download_data.py` (market data fetch), `generate_data.py` (synthetic) |
| Strategy | Moving Average Crossover (configurable windows) |
| Reporting | matplotlib (equity curves, drawdown charts) |

## How It Works

- **Market data** is fetched or synthetically generated via dedicated scripts
- **Events** (bar data, signals, orders, fills) flow through a central event queue
- **Strategy module** computes fast/slow MA crossovers and emits BUY/SELL signals
- **Execution handler** simulates order fills with configurable slippage and commission
- **Portfolio tracker** maintains positions, cash, and equity curve in real time
- **Reporting** generates trade log, total return, Sharpe ratio, and max drawdown

## Quick Start

```bash
git clone https://github.com/shahidbaig-shaik/quant_trading_platform
cd quant_trading_platform
pip install -r requirements.txt
python download_data.py        # fetch market data
python main.py                 # run backtest
```
