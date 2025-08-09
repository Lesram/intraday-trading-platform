"""Unit tests for the Alpaca client adapter."""
from unittest.mock import patch

import httpx
import pytest
import respx

from adapters.alpaca_client import AlpacaClient
from adapters.errors import AlpacaError, AlpacaAuthenticationError, AlpacaRateLimitError


class TestAlpacaClient:
    """Test suite for AlpacaClient."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return AlpacaClient()

    @respx.mock
    async def test_get_account_success(self, client):
        """Test successful account retrieval."""
        # Mock the API response
        respx.get("https://paper-api.alpaca.markets/v2/account").mock(
            return_value=httpx.Response(200, json={
                "account_number": "123456789",
                "status": "ACTIVE",
                "buying_power": "100000.0",
                "cash": "50000.0",
                "portfolio_value": "100000.0",
                "equity": "100000.0",
                "daytrade_count": 0,
                "pattern_day_trader": False
            })
        )

        async with client:
            account = await client.get_account()
            
        assert account.account_number == "123456789"
        assert account.buying_power == 100000.0
        assert account.equity == 100000.0

    @respx.mock
    async def test_get_account_api_error(self, client):
        """Test API error handling in account retrieval."""
        respx.get("https://paper-api.alpaca.markets/v2/account").mock(
            return_value=httpx.Response(500, json={"message": "Server error"})
        )

        async with client:
            with pytest.raises(Exception) as exc_info:  # Catch any exception from retry wrapper
                await client.get_account()
                
        # Check that the underlying error message is present
        assert "Server error" in str(exc_info.value)    @respx.mock
    async def test_get_positions_empty(self, client):
        """Test retrieving empty positions list."""
        respx.get("https://paper-api.alpaca.markets/v2/positions").mock(
            return_value=httpx.Response(200, json=[])
        )

        async with client:
            positions = await client.get_positions()

        assert positions == []

    @respx.mock
    async def test_get_positions_with_data(self, client):
        """Test retrieving positions with data."""
        positions_data = [
            {
                "symbol": "AAPL",
                "qty": "10",
                "side": "long",
                "market_value": "1500.0",
                "cost_basis": "1450.0",
                "unrealized_pl": "50.0"
            }
        ]

        respx.get("https://paper-api.alpaca.markets/v2/positions").mock(
            return_value=httpx.Response(200, json=positions_data)
        )

        async with client:
            positions = await client.get_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == "10"

    @respx.mock
    async def test_submit_order_success(self, client):
        """Test successful order submission."""
        order_request = {
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }

        order_response = {
            "id": "test_order_id",
            "status": "new",
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "order_type": "market",
            "submitted_at": "2024-01-01T10:00:00Z"
        }

        respx.post("https://paper-api.alpaca.markets/v2/orders").mock(
            return_value=httpx.Response(201, json=order_response)
        )

        async with client:
            order = await client.submit_order(order_request)

        assert order["id"] == "test_order_id"
        assert order["symbol"] == "AAPL"
        assert order["status"] == "new"

    @respx.mock
    async def test_submit_order_validation_error(self, client):
        """Test order submission with validation error."""
        order_request = {
            "symbol": "INVALID",
            "qty": 0,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }

        respx.post("https://paper-api.alpaca.markets/v2/orders").mock(
            return_value=httpx.Response(400, json={
                "code": 40010001,
                "message": "Invalid quantity"
            })
        )

        async with client:
            with pytest.raises(AlpacaError) as exc_info:
                await client.submit_order(order_request)

        assert "Invalid quantity" in str(exc_info.value)

    @respx.mock
    async def test_get_order_success(self, client):
        """Test successful order retrieval."""
        order_id = "test_order_id"
        order_data = {
            "id": order_id,
            "status": "filled",
            "symbol": "AAPL",
            "qty": "10",
            "filled_qty": "10",
            "side": "buy",
            "order_type": "market",
            "filled_avg_price": "150.0",
            "submitted_at": "2024-01-01T10:00:00Z",
            "filled_at": "2024-01-01T10:00:05Z"
        }

        respx.get(f"https://paper-api.alpaca.markets/v2/orders/{order_id}").mock(
            return_value=httpx.Response(200, json=order_data)
        )

        async with client:
            order = await client.get_order(order_id)

        assert order["id"] == order_id
        assert order["status"] == "filled"
        assert order["filled_qty"] == "10"

    @respx.mock
    async def test_cancel_order_success(self, client):
        """Test successful order cancellation."""
        order_id = "test_order_id"

        respx.delete(f"https://paper-api.alpaca.markets/v2/orders/{order_id}").mock(
            return_value=httpx.Response(204)
        )

        async with client:
            result = await client.cancel_order(order_id)

        assert result is True

    @respx.mock
    async def test_rate_limit_handling(self, client):
        """Test rate limit error handling."""
        respx.get("https://paper-api.alpaca.markets/v2/account").mock(
            return_value=httpx.Response(429, json={"message": "Rate limit exceeded"})
        )

        async with client:
            with pytest.raises(AlpacaRateLimitError):
                await client.get_account()

    @respx.mock
    async def test_authentication_error(self, client):
        """Test authentication error handling."""
        respx.get("https://paper-api.alpaca.markets/v2/account").mock(
            return_value=httpx.Response(401, json={"message": "Unauthorized"})
        )

        async with client:
            with pytest.raises(AlpacaAuthenticationError):
                await client.get_account()

    async def test_context_manager(self):
        """Test async context manager functionality."""
        client = AlpacaClient()

        async with client:
            assert client.client is not None

        # Client should be closed after exiting context
        assert client.client.is_closed

    @patch('adapters.alpaca_client.asyncio.sleep')
    @respx.mock
    async def test_retry_mechanism(self, mock_sleep, client):
        """Test retry mechanism with exponential backoff."""
        # First two calls return 503, third succeeds
        route = respx.get("https://paper-api.alpaca.markets/v2/account")
        route.side_effect = [
            httpx.Response(503, json={"message": "Service unavailable"}),
            httpx.Response(503, json={"message": "Service unavailable"}),
            httpx.Response(200, json={"id": "test_account"})
        ]

        async with client:
            account = await client.get_account()

        assert account["id"] == "test_account"
        assert mock_sleep.call_count == 2  # Two retries before success

    @respx.mock
    async def test_idempotency_key_usage(self, client):
        """Test idempotency key is included in requests."""
        order_request = {
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }

        route = respx.post("https://paper-api.alpaca.markets/v2/orders")
        route.mock(return_value=httpx.Response(201, json={"id": "test_order"}))

        async with client:
            await client.submit_order(order_request)

        # Check that idempotency key header was included
        request = route.calls[0].request
        assert "Idempotency-Key" in request.headers
