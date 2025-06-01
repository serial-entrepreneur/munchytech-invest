import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
from kiteconnect import KiteConnect
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
import webbrowser
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent
env_path = project_root/'mt_inv'/'config'/'.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# Load environment variables
KITE_API_KEY = os.getenv('KITE_API_KEY')
KITE_API_SECRET = os.getenv('KITE_API_SECRET')
KITE_REDIRECT_URI = os.getenv('KITE_REDIRECT_URI')

# Initialize Kite Connect
kite = KiteConnect(api_key=KITE_API_KEY)

# Global variable to store the request token
request_token = None

# Valid intervals for historical data
VALID_INTERVALS = ['minute', '5minute', '15minute', '30minute', '60minute', 'day']

class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global request_token
        # Parse the URL and query parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Extract request token
        if 'request_token' in query_params:
            request_token = query_params['request_token'][0]
            
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Authentication successful! You can close this window.")
        
    def log_message(self, format, *args):
        # Suppress logging
        return

def start_redirect_server():
    """Start a local server to handle the redirect"""
    server = HTTPServer(('localhost', 3000), RedirectHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return server

def authenticate() -> bool:
    """
    Authenticate with Kite Connect API using redirect URI.
    Returns True if authentication is successful, False otherwise.
    """
    try:
        global request_token
        
        # Start local server to handle redirect
        server = start_redirect_server()
        
        # Generate login URL
        login_url = kite.login_url()
        logger.info(f"Opening browser for authentication...")
        
        # Open browser for authentication
        webbrowser.open(login_url)
        
        # Wait for request token (timeout after 60 seconds)
        timeout = 60
        start_time = time.time()
        while request_token is None and time.time() - start_time < timeout:
            time.sleep(1)
        
        # Stop the server
        server.shutdown()
        server.server_close()
        
        if request_token is None:
            logger.error("Authentication timed out. No request token received.")
            return False
            
        # Generate session
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        kite.set_access_token(data["access_token"])
        
        logger.info("Authentication successful.")
        return True
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False

def get_instrument_token(symbol: str) -> Optional[int]:
    """
    Get the instrument token for a given symbol.
    
    Args:
        symbol: The trading symbol (e.g., 'RELIANCE', 'INFY')
        
    Returns:
        The instrument token if found, None otherwise
    """
    try:
        # Get all instruments
        instruments = kite.instruments()
        # Find the instrument with matching symbol
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
        logger.warning(f"No instrument found for symbol: {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error fetching instrument token for {symbol}: {e}")
        return None

def validate_historical_data(data: List[Dict]) -> bool:
    """
    Validate the historical data for completeness and correctness.
    
    Args:
        data: List of historical data points
        
    Returns:
        True if data is valid, False otherwise
    """
    if not data:
        logger.warning("Empty data received")
        return False
        
    required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for point in data:
        # Check if all required fields are present
        if not all(field in point for field in required_fields):
            logger.warning(f"Missing required fields in data point: {point}")
            return False
            
        # Validate OHLC relationship
        if not (point['low'] <= point['open'] <= point['high'] and 
                point['low'] <= point['close'] <= point['high']):
            logger.warning(f"Invalid OHLC relationship in data point: {point}")
            return False
            
        # Validate volume
        if point['volume'] < 0:
            logger.warning(f"Negative volume in data point: {point}")
            return False
            
    return True

def fetch_historical_data(
    symbol: str,
    from_date: Union[str, datetime],
    to_date: Union[str, datetime],
    interval: str = 'day'
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a given symbol and time range.
    
    Args:
        symbol: The trading symbol (e.g., 'RELIANCE', 'INFY')
        from_date: Start date (string in 'YYYY-MM-DD' format or datetime object)
        to_date: End date (string in 'YYYY-MM-DD' format or datetime object)
        interval: Data interval ('minute', '5minute', '15minute', '30minute', '60minute', 'day')
        
    Returns:
        DataFrame containing historical data if successful, None otherwise
    """
    try:
        # Validate interval
        if interval not in VALID_INTERVALS:
            logger.error(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")
            return None
            
        # Convert string dates to datetime if needed
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d')
        if isinstance(to_date, str):
            to_date = datetime.strptime(to_date, '%Y-%m-%d')
            
        # Get instrument token
        instrument_token = get_instrument_token(symbol)
        if not instrument_token:
            return None
            
        # Fetch historical data
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        # Validate data
        if not validate_historical_data(data):
            logger.error("Data validation failed")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def fetch_multiple_symbols(
    symbols: List[str],
    from_date: Union[str, datetime],
    to_date: Union[str, datetime],
    interval: str = 'day'
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        from_date: Start date
        to_date: End date
        interval: Data interval
        
    Returns:
        Dictionary mapping symbols to their historical data DataFrames
    """
    results = {}
    for symbol in symbols:
        df = fetch_historical_data(symbol, from_date, to_date, interval)
        if df is not None:
            results[symbol] = df
    return results

# Example usage
if __name__ == "__main__":
    if authenticate():
        # Example: Fetch historical data for a single symbol
        symbol = "RELIANCE"
        from_date = "2024-01-01"
        to_date = "2024-03-14"
        
        df = fetch_historical_data(symbol, from_date, to_date, interval='day')
        if df is not None:
            print(f"\nHistorical data for {symbol}:")
            print(df.head())
            
        # Example: Fetch data for multiple symbols
        symbols = ["RELIANCE", "INFY", "TCS"]
        results = fetch_multiple_symbols(symbols, from_date, to_date)
        
        for symbol, data in results.items():
            print(f"\nHistorical data for {symbol}:")
            print(data.head()) 