import pytest
from unittest.mock import patch, MagicMock
from mt_inv.data_ingestion.kite_connect_integration import (
    authenticate,
    fetch_historical_data,
)


# Mock KiteConnect instance
@pytest.fixture
def mock_kite():
    with patch("mt_inv.data_ingestion.kite_connect_integration.kite") as mock:
        yield mock


# Test authentication function
def test_authenticate_success(mock_kite):
    mock_kite.login_url.return_value = "http://example.com/login"
    mock_kite.set_access_token.return_value = None
    with patch("builtins.input", return_value="dummy_access_token"):
        authenticate()
    mock_kite.login_url.assert_called_once()
    mock_kite.set_access_token.assert_called_once_with("dummy_access_token")


# Test authentication failure
def test_authenticate_failure(mock_kite):
    mock_kite.login_url.side_effect = Exception("Authentication failed")
    with patch("builtins.input", return_value="dummy_access_token"):
        authenticate()
    mock_kite.login_url.assert_called_once()
    mock_kite.set_access_token.assert_not_called()


# Test fetch historical data success
def test_fetch_historical_data_success(mock_kite):
    mock_data = [{"date": "2023-01-01", "open": 100, "close": 105}]
    mock_kite.historical_data.return_value = mock_data
    result = fetch_historical_data(
        "instrument_token", "2023-01-01", "2023-01-31", "day"
    )
    assert result == mock_data
    mock_kite.historical_data.assert_called_once_with(
        "instrument_token", "2023-01-01", "2023-01-31", "day"
    )


# Test fetch historical data failure
def test_fetch_historical_data_failure(mock_kite):
    mock_kite.historical_data.side_effect = Exception("Error fetching data")
    result = fetch_historical_data(
        "instrument_token", "2023-01-01", "2023-01-31", "day"
    )
    assert result is None
    mock_kite.historical_data.assert_called_once_with(
        "instrument_token", "2023-01-01", "2023-01-31", "day"
    )
