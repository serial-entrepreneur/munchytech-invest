from loguru import logger
from mt_inv.data_ingestion.order_book import OrderBookManager


def on_order_book_update(instrument_token: int, order_book: dict):
    """
    Callback function to handle order book updates.

    Args:
        instrument_token: The instrument token
        order_book: Dictionary containing order book data
    """
    logger.info(f"Order book update for token {instrument_token}:")
    logger.info(f"Bids: {order_book['bids']}")
    logger.info(f"Asks: {order_book['asks']}")


def main():
    """Main function to test live order book data."""
    # Initialize order book manager
    order_book = OrderBookManager()

    # Register callback
    order_book.register_callback(on_order_book_update)

    # Subscribe to some symbols
    symbols = ["RELIANCE", "INFY", "TCS"]
    if order_book.subscribe(symbols):
        logger.info(f"Subscribed to {symbols}")

        try:
            # Keep the script running
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            # Cleanup
            order_book.unsubscribe(symbols)
            order_book.close()
    else:
        logger.error("Failed to subscribe to symbols")


if __name__ == "__main__":
    main()
