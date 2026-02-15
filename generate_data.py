import pandas as pd
import numpy as np
import os

def generate_stock_data(symbol, days=1000, start_price=150):
    """Generates synthetic OHLCV data using a random walk."""
    np.random.seed(42 if symbol == 'AAPL' else 123)
    
    dates = pd.date_range(start='2020-01-01', periods=days, freq='B')
    returns = np.random.normal(0, 0.02, days) # 2% daily volatility
    price_paths = start_price * (1 + returns).cumprod()
    
    data = pd.DataFrame(index=dates)
    data['datetime'] = dates
    data['close'] = price_paths
    # Synthesize other columns based on Close
    data['open'] = data['close'] * (1 + np.random.normal(0, 0.005, days))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, days)))
    data['volume'] = np.random.randint(100000, 5000000, days)
    
    # Reorder columns to match DataHandler expectation
    data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    output_path = f"data/{symbol}.csv"
    os.makedirs("data", exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {days} rows.")

if __name__ == "__main__":
    generate_stock_data("AAPL", start_price=150)
    generate_stock_data("GOOG", start_price=2800)
