from typing import Dict, List, Optional, Callable
from datetime import datetime, time
from loguru import logger
from kiteconnect import KiteTicker
from .kite_connect_integration import kite, get_instrument_token, KITE_API_KEY


def is_market_open() -> bool:
    """
    Check if the Indian stock market is currently open.
    Market hours: 9:15 AM to 3:30 PM IST on weekdays.

    Returns:
        bool: True if market is open, False otherwise
    """
    now = datetime.now()

    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if now.weekday() > 4:
        logger.debug("Market is closed (weekend)")
        return False

    # Market hours: 9:15 AM to 3:30 PM IST
    market_start = time(9, 15)
    market_end = time(15, 30)
    current_time = now.time()

    is_open = market_start <= current_time <= market_end
    if not is_open:
        logger.debug(f"Market is closed (current time: {current_time})")
    return is_open


class OrderBookManager:
    """
    Manages real-time order book data (Level II) using Kite Connect WebSocket.
    """

    def __init__(self):
        """Initialize the OrderBookManager with KiteTicker."""
        self.kws = KiteTicker(KITE_API_KEY, kite.access_token)
        # instrument_token -> order book
        self.order_books: Dict[int, Dict] = {}
        self.callbacks: List[Callable] = []
        self.is_connected = False
        # tokens to subscribe after connect
        self._pending_tokens = set()

        # Bind callbacks
        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_reconnect = self._on_reconnect

    def _on_ticks(self, ws, ticks):
        try:
            # Skip processing if market is closed
            if not is_market_open():
                return

            for tick in ticks:
                if (
                    "depth" in tick
                    and "bids" in tick["depth"]
                    and "asks" in tick["depth"]
                ):
                    instrument_token = tick["instrument_token"]
                    self.order_books[instrument_token] = {
                        "timestamp": datetime.now(),
                        "bids": tick["depth"]["bids"],
                        "asks": tick["depth"]["asks"],
                    }
                    for callback in self.callbacks:
                        callback(instrument_token, self.order_books[instrument_token])
                else:
                    logger.debug(f"Received tick without depth data: {tick}")
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")

    def _on_connect(self, ws, response):
        self.is_connected = True
        logger.info("WebSocket connected successfully")
        if self._pending_tokens:
            tokens = list(self._pending_tokens)
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            logger.info(
                f"Subscribed to order book updates for "
                f"{len(tokens)} instruments (on connect)"
            )

    def _on_close(self, ws, code, reason):
        self.is_connected = False
        logger.warning(f"WebSocket closed: {code} - {reason}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_reconnect(self, ws, attempts):
        logger.info(f"WebSocket reconnecting... Attempt {attempts}")

    def subscribe(self, symbols: List[str]) -> bool:
        """
        Subscribe to order book updates for given symbols.

        Args:
            symbols: List of trading symbols to subscribe to

        Returns:
            bool: True if subscription successful, False otherwise
        """
        try:
            # Check if market is open
            if not is_market_open():
                logger.warning(
                    "Market is currently closed. "
                    "Order book data may not be available."
                )

            instrument_tokens = []
            for symbol in symbols:
                token = get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
                else:
                    logger.warning(f"Could not get instrument token for {symbol}")

            if not instrument_tokens:
                logger.error("No valid instrument tokens found")
                return False

            self._pending_tokens.update(instrument_tokens)
            if not self.is_connected:
                self.kws.connect(threaded=True)
                logger.info("Connecting WebSocket, will subscribe on connect.")
            else:
                self.kws.subscribe(instrument_tokens)
                self.kws.set_mode(self.kws.MODE_FULL, instrument_tokens)
                logger.info(
                    f"Subscribed to order book updates for "
                    f"{len(instrument_tokens)} instruments (immediate)"
                )
            return True

        except Exception as e:
            logger.error(f"Error subscribing to order book: {e}")
            return False

    def unsubscribe(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from order book updates for given symbols.

        Args:
            symbols: List of trading symbols to unsubscribe from

        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        try:
            instrument_tokens = []
            for symbol in symbols:
                token = get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
                    self._pending_tokens.discard(token)

            if instrument_tokens:
                self.kws.unsubscribe(instrument_tokens)
                logger.info(
                    f"Unsubscribed from order book updates for "
                    f"{len(instrument_tokens)} instruments"
                )
                return True
            return False

        except Exception as e:
            logger.error(f"Error unsubscribing from order book: {e}")
            return False

    def register_callback(self, callback: Callable):
        """
        Register a callback function to be called when order book updates are received.

        Args:
            callback: Function that takes (instrument_token, order_book) as arguments
        """
        self.callbacks.append(callback)

    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get the current order book for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict containing order book data if available, None otherwise
        """
        token = get_instrument_token(symbol)
        if token and token in self.order_books:
            return self.order_books[token]
        return None

    def close(self):
        """Close the WebSocket connection."""
        if self.is_connected:
            self.kws.close()
            self.is_connected = False
            logger.info("WebSocket connection closed")
