"""Test happy path trading scenarios."""
import pytest
import httpx


@pytest.mark.asyncio
async def test_submit_market_buy_order_success(client, alpaca_mock):
    """Test successful market buy order submission."""
    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["side"] == "buy"
    assert data["status"] in ["accepted", "submitted", "new"]
    assert "order_ref" in data
    assert "request_id" in data


@pytest.mark.asyncio
async def test_get_order_status_success(client, alpaca_mock, alpaca_urls):
    """Test successful order status retrieval using actual order submission."""
    # Step 1: First submit an order to get a real order ID
    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy"
    }
    
    submit_response = await client.post("/api/trading/orders/market", json=order_data)
    assert submit_response.status_code == 200
    
    # Extract the internal order ID from the response
    submit_data = submit_response.json()
    internal_order_id = submit_data["order_ref"]
    
    # Step 2: Query that order by its internal ID  
    response = await client.get(f"/api/trading/orders/{internal_order_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert "order" in data
    order = data["order"]
    assert order["symbol"] == "AAPL"
    assert order["side"] == "buy"


@pytest.mark.asyncio
async def test_get_all_orders_success(client, alpaca_mock, alpaca_urls):
    """Test successful retrieval of all orders."""
    # Mock orders response
    alpaca_mock.get(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "test_order_12345",
                "symbol": "AAPL", 
                "qty": "10",
                "side": "buy",
                "order_type": "market",
                "status": "filled",
                "submitted_at": "2024-08-08T10:00:00Z",
                "filled_at": "2024-08-08T10:00:05Z",
                "filled_avg_price": "150.25"
            }
        ])
    )
    
    response = await client.get("/api/trading/orders")
    assert response.status_code == 200
    
    data = response.json()
    assert "orders" in data
    assert "count" in data
    assert isinstance(data["orders"], list)
    
    if len(data["orders"]) > 0:
        order = data["orders"][0]
        assert "symbol" in order
        assert "side" in order


@pytest.mark.asyncio
async def test_cancel_order_success(client, alpaca_mock, alpaca_urls):
    """Test successful order cancellation using real order submission."""
    # Step 1: Submit an order first
    order_data = {
        "symbol": "TSLA",
        "qty": 5.0,
        "side": "buy"
    }
    
    submit_response = await client.post("/api/trading/orders/market", json=order_data)
    assert submit_response.status_code == 200
    internal_order_id = submit_response.json()["order_ref"]
    
    # Step 2: Mock successful Alpaca cancellation
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}test_order_12345").mock(
        return_value=httpx.Response(204)
    )

    # Step 3: Cancel the order with proper request body
    cancel_data = {"reason": "Test cancellation"}
    response = await client.post(f"/api/trading/orders/{internal_order_id}/cancel", json=cancel_data)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["canceled", "cancellation_requested", "success"]
    assert data["cancelled"] == True
    assert "order_ref" in data
@pytest.mark.asyncio
async def test_risk_summary_endpoint(client):
    """Test risk summary endpoint."""
    response = await client.get("/api/trading/risk/summary")
    assert response.status_code == 200

    data = response.json()
    assert "risk_summary" in data

    risk_summary = data["risk_summary"]
    assert "portfolio_heat" in risk_summary
    assert "portfolio_value" in risk_summary
    assert "buying_power" in risk_summary
    assert "cash" in risk_summary
    assert "daily_losses" in risk_summary
    assert "risk_level" in risk_summary
    assert isinstance(risk_summary["portfolio_heat"], (int, float))
@pytest.mark.asyncio
async def test_trading_mode_configuration(client):
    """Test trading mode configuration."""
    mode_data = {
        "auto_trading": True,
        "risk_management": True,
        "max_portfolio_heat": 0.02
    }
    
    response = await client.post("/api/trading/mode", json=mode_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "auto_trading" in data
    assert "risk_management" in data
    assert data["auto_trading"] == True
    assert data["risk_management"] == True


@pytest.mark.asyncio
async def test_market_order_with_oco_stop(client, alpaca_mock):
    """Test market order with OCO stop loss."""
    order_data = {
        "symbol": "TSLA",
        "qty": 5.0,
        "side": "buy",
        "oco_stop": 190.0
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["symbol"] == "TSLA"
    assert data["side"] == "buy"
    assert data["status"] in ["accepted", "submitted", "new"]


@pytest.mark.asyncio
async def test_full_trading_workflow(client, alpaca_mock, alpaca_urls):
    """Test complete trading workflow: submit -> check status -> cancel if needed."""
    # Step 1: Submit market order
    order_data = {
        "symbol": "NVDA",
        "qty": 2.0,
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    assert response.status_code == 200
    internal_order_id = response.json()["order_ref"]
    
    # Step 2: Check order status using the internal order ID
    response = await client.get(f"/api/trading/orders/{internal_order_id}")
    assert response.status_code == 200
    
    order_data = response.json()["order"]
    assert order_data["symbol"] == "NVDA"
    assert order_data["side"] == "buy"
    
    # Step 3: Cancel order if it's still pending
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}test_order_12345").mock(
        return_value=httpx.Response(204)
    )
    
    cancel_data = {"reason": "Test workflow cancellation"}
    response = await client.post(f"/api/trading/orders/{internal_order_id}/cancel", json=cancel_data)
    assert response.status_code == 200
    cancel_data = response.json()
    assert cancel_data["status"] in ["canceled", "cancellation_requested", "success"]
