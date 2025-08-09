"""Test broker failure scenarios."""
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_alpaca_api_timeout(client, alpaca_urls):
    """Test handling of Alpaca API timeout."""
    with respx.mock:
        # Mock timeout on order submission
        respx.post(alpaca_urls["orders"]).mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        
        order_data = {
            "symbol": "AAPL",
            "qty": 10.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle timeout gracefully
        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert "timeout" in data["detail"].lower() or "unavailable" in data["detail"].lower()


@pytest.mark.asyncio
async def test_alpaca_api_500_error(client, alpaca_urls):
    """Test handling of Alpaca API 500 error."""
    with respx.mock:
        # Mock 500 error from Alpaca
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(500, json={
                "message": "Internal server error"
            })
        )
        
        order_data = {
            "symbol": "TSLA",
            "qty": 5.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle server error
        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert "broker" in data["detail"].lower() or "server error" in data["detail"].lower()


@pytest.mark.asyncio
async def test_alpaca_api_401_unauthorized(client, alpaca_urls):
    """Test handling of Alpaca API authentication failure."""
    with respx.mock:
        # Mock 401 unauthorized
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(401, json={
                "message": "Unauthorized"
            })
        )
        
        order_data = {
            "symbol": "MSFT",
            "qty": 8.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle auth failure
        assert response.status_code == 401
        data = response.json()
        assert "unauthorized" in data["detail"].lower() or "authentication" in data["detail"].lower()


@pytest.mark.asyncio
async def test_alpaca_api_422_validation_error(client, alpaca_urls):
    """Test handling of Alpaca API validation errors."""
    with respx.mock:
        # Mock 422 validation error
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(422, json={
                "message": "Invalid order parameters",
                "code": 40210000,
                "details": "Quantity must be positive"
            })
        )
        
        order_data = {
            "symbol": "NVDA",
            "qty": -5.0,  # Invalid negative quantity
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle validation error
        assert response.status_code == 422
        data = response.json()
        assert "validation" in data["detail"].lower() or "invalid" in data["detail"].lower()


@pytest.mark.asyncio
async def test_alpaca_connection_error(client, alpaca_urls):
    """Test handling of connection errors to Alpaca."""
    with respx.mock:
        # Mock connection error
        respx.post(alpaca_urls["orders"]).mock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        
        order_data = {
            "symbol": "AAPL",
            "qty": 10.0,
            "side": "sell"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle connection failure
        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert "connection" in data["detail"].lower() or "unavailable" in data["detail"].lower()


@pytest.mark.asyncio
async def test_order_rejection_insufficient_funds(client, alpaca_urls):
    """Test handling of order rejection due to insufficient funds."""
    with respx.mock:
        # Mock insufficient funds rejection
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(403, json={
                "message": "Insufficient buying power",
                "code": 40310000
            })
        )
        
        order_data = {
            "symbol": "AMZN",  # Expensive stock
            "qty": 1000.0,     # Large quantity
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle insufficient funds
        assert response.status_code == 400  # Bad Request
        data = response.json()
        assert "insufficient" in data["detail"].lower() or "funds" in data["detail"].lower()


@pytest.mark.asyncio
async def test_order_rejection_invalid_symbol(client, alpaca_urls):
    """Test handling of invalid symbol rejection."""
    with respx.mock:
        # Mock invalid symbol rejection
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(422, json={
                "message": "Invalid symbol",
                "code": 40210001
            })
        )
        
        order_data = {
            "symbol": "INVALIDTICKER",
            "qty": 10.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle invalid symbol
        assert response.status_code == 422
        data = response.json()
        assert "symbol" in data["detail"].lower() or "invalid" in data["detail"].lower()


@pytest.mark.asyncio
async def test_account_fetch_failure(client, alpaca_urls):
    """Test handling of account information fetch failure."""
    with respx.mock:
        # Mock account fetch failure
        respx.get(alpaca_urls["account"]).mock(
            return_value=httpx.Response(503, json={
                "message": "Service temporarily unavailable"
            })
        )
        
        response = await client.get("/trading/account")
        
        # Should handle account fetch failure
        assert response.status_code == 503
        data = response.json()
        assert "unavailable" in data["detail"].lower() or "service" in data["detail"].lower()


@pytest.mark.asyncio
async def test_positions_fetch_failure(client, alpaca_urls):
    """Test handling of positions fetch failure."""
    with respx.mock:
        # Mock positions fetch failure
        respx.get(alpaca_urls["positions"]).mock(
            side_effect=httpx.ReadTimeout("Read timeout")
        )
        
        response = await client.get("/trading/positions")
        
        # Should handle positions fetch failure
        assert response.status_code == 503
        data = response.json()
        assert "timeout" in data["detail"].lower() or "unavailable" in data["detail"].lower()


@pytest.mark.asyncio
async def test_order_status_fetch_failure(client, alpaca_urls):
    """Test handling of order status fetch failure."""
    order_id = "test_order_12345"
    
    with respx.mock:
        # Mock order status fetch failure
        respx.get(f"{alpaca_urls['single_order']}{order_id}").mock(
            return_value=httpx.Response(404, json={
                "message": "Order not found"
            })
        )
        
        response = await client.get(f"/trading/orders/{order_id}")
        
        # Should handle order not found
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower() or "order" in data["detail"].lower()


@pytest.mark.asyncio
async def test_multiple_retry_attempts_failure(client, alpaca_urls):
    """Test behavior when multiple retry attempts fail."""
    with respx.mock:
        # Mock consistent failures across retries
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(503, json={
                "message": "Service unavailable"
            })
        )
        
        order_data = {
            "symbol": "AAPL",
            "qty": 10.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should fail after retries exhausted
        assert response.status_code == 503
        data = response.json()
        assert "unavailable" in data["detail"].lower() or "retry" in data["detail"].lower()


@pytest.mark.asyncio
async def test_malformed_response_handling(client, alpaca_urls):
    """Test handling of malformed responses from Alpaca."""
    with respx.mock:
        # Mock malformed JSON response
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(200, content="Invalid JSON response {")
        )
        
        order_data = {
            "symbol": "AAPL",
            "qty": 10.0,
            "side": "buy"
        }
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle malformed response
        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert "invalid response" in data["detail"].lower() or "parse" in data["detail"].lower()
