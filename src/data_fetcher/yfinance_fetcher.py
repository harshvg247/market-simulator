import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical price data for a given ticker.
    
    It tries to find the best price column in the order:
    1. 'Adj Close' (Adjusted for splits/dividends)
    2. 'Adjusted Close' (Alternative name)
    3. 'Close' (Not adjusted, but will work)
    
    Args:
        ticker (str): The stock ticker (e.g., 'SPY').
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        
    Returns:
        pd.Series: A Series of prices, indexed by date.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}.")
        
        # --- This is the new, robust logic ---
        if 'Adj Close' in data.columns:
            price_series = data['Adj Close']
        elif 'Adjusted Close' in data.columns:
            price_series = data['Adjusted Close']
        elif 'Close' in data.columns:
            price_series = data['Close']
        else:
            raise ValueError(f"Could not find 'Adj Close', 'Adjusted Close', or 'Close' in data columns.")
        
        return price_series
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None