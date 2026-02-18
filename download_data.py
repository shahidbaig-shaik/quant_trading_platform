"""
download_data.py — Fetch Real OHLCV Data from Yahoo Finance.

Downloads historical price data for specified symbols and saves
it in the CSV format expected by the trading engine.

Usage:
    python download_data.py
    python download_data.py --symbols AAPL MSFT GOOG --start 2018-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import yfinance as yf


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    output_dir: str = "data",
) -> None:
    """
    Download OHLCV data for a single symbol and save as CSV.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker (e.g. "AAPL").
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    output_dir : str
        Directory to save CSV files (default "data/").
    """
    print(f"Downloading {symbol} ({start} → {end})...", end=" ")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        print(f"WARNING: No data returned for {symbol}")
        return

    # Rename columns to match engine's expected format
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(output_path, index=False)
    print(f"✓ {len(df)} bars → {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from Yahoo Finance"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOG"],
        help="Ticker symbols to download (default: AAPL GOOG)",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for CSV files (default: data/)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Yahoo Finance Data Downloader")
    print("=" * 50)

    for symbol in args.symbols:
        try:
            download_symbol(symbol, args.start, args.end, args.output_dir)
        except Exception as e:
            print(f"ERROR downloading {symbol}: {e}")
            continue

    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
