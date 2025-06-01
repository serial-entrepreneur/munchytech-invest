import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from mt_inv.data_ingestion.order_book import OrderBookManager, is_market_open


class TestOrderBookManager(unittest.TestCase):
    """Test cases for OrderBookManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = OrderBookManager()
        self.mock_kite_ticker = MagicMock()
        self.manager.kws = self.mock_kite_ticker

    def test_initialization(self):
        """Test OrderBookManager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(len(self.manager.order_books), 0)
        self.assertEqual(len(self.manager.callbacks), 0)
        self.assertFalse(self.manager.is_connected)

    def test_handle_ticks(self):
        """Test handling of tick data."""
        # Mock tick data
        mock_tick = {
            "instrument_token": 123,
            "depth": {
                "bids": [(100.0, 10), (99.0, 20)],
                "asks": [(101.0, 15), (102.0, 25)],
            },
        }

        # Mock callback
        mock_callback = MagicMock()
        self.manager.register_callback(mock_callback)

        # Call _on_ticks
        self.manager._on_ticks(None, [mock_tick])

        # Verify order book was updated
        self.assertIn(123, self.manager.order_books)
        self.assertEqual(
            self.manager.order_books[123]["bids"], mock_tick["depth"]["bids"]
        )
        self.assertEqual(
            self.manager.order_books[123]["asks"], mock_tick["depth"]["asks"]
        )

        # Verify callback was called
        mock_callback.assert_called_once()

    def test_websocket_events(self):
        """Test WebSocket event handlers."""
        # Test connect
        self.manager._on_connect(None, None)
        self.assertTrue(self.manager.is_connected)

        # Test close
        self.manager._on_close(None, 1000, "Normal closure")
        self.assertFalse(self.manager.is_connected)

        # Test error
        self.manager._on_error(None, "Test error")

        # Test reconnect
        self.manager._on_reconnect(None, 1)

    def test_subscribe_unsubscribe(self):
        """Test subscription and unsubscription."""
        # Mock get_instrument_token
        with patch(
            "mt_inv.data_ingestion.order_book.get_instrument_token"
        ) as mock_get_token:
            mock_get_token.return_value = 123

            # Test subscribe
            result = self.manager.subscribe(["RELIANCE"])
            self.assertTrue(result)
            self.mock_kite_ticker.subscribe.assert_called_once()

            # Test unsubscribe
            result = self.manager.unsubscribe(["RELIANCE"])
            self.assertTrue(result)
            self.mock_kite_ticker.unsubscribe.assert_called_once()

    def test_get_order_book(self):
        """Test getting order book data."""
        # Mock get_instrument_token
        with patch(
            "mt_inv.data_ingestion.order_book.get_instrument_token"
        ) as mock_get_token:
            mock_get_token.return_value = 123

            # Add some test data
            test_data = {
                "timestamp": datetime.now(),
                "bids": [(100.0, 10)],
                "asks": [(101.0, 15)],
            }
            self.manager.order_books[123] = test_data

            # Test get_order_book
            result = self.manager.get_order_book("RELIANCE")
            self.assertEqual(result, test_data)

            # Test non-existent symbol
            result = self.manager.get_order_book("INVALID")
            self.assertIsNone(result)


class TestMarketHours(unittest.TestCase):
    """Test cases for market hours validation."""

    def test_market_hours(self):
        """Test market hours validation."""
        # Test during market hours
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0)
            self.assertTrue(is_market_open())

        # Test before market hours
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 9, 0)
            self.assertFalse(is_market_open())

        # Test after market hours
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 16, 0)
            self.assertFalse(is_market_open())

        # Test weekend
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 6, 10, 0)
            self.assertFalse(is_market_open())


if __name__ == "__main__":
    unittest.main()
