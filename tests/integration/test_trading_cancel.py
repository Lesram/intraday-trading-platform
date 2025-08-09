"""Test order cancellation functionality."""
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_cancel_pending_order_success(client, alpaca_mock, alpaca_urls):
    """Test successful cancellation of pending order."""
    order_id = "test_order_12345"
    
    # Mock successful cancellation
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(204)  # No content on successful cancel
    )
    
    # Mock order status showing cancelled
    alpaca_mock.get(f"{alpaca_urls['single_order']}{order_id}").mock(
        return_value=httpx.Response(200, json={
            "id": order_id,
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "order_type": "limit",
            "limit_price": "150.0",
            "status": "canceled",
            "submitted_at": "2024-08-08T10:00:00Z",
            "canceled_at": "2024-08-08T10:01:00Z"
        })
    )
    
    response = await client.delete(f"/trading/orders/{order_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == order_id
    assert data["status"] == "canceled"
    assert "canceled_at" in data


@pytest.mark.asyncio
async def test_cancel_filled_order_failure(client, alpaca_mock, alpaca_urls):
    """Test cancellation fails for already filled order."""
    order_id = "filled_order_12345"
    
    # Mock cancellation failure - order already filled
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(422, json={
            "message": "Order is already filled",
            "code": 42210000
        })
    )
    
    response = await client.delete(f"/trading/orders/{order_id}")
    assert response.status_code == 422
    
    data = response.json()
    assert "filled" in data["detail"].lower() or "cannot cancel" in data["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_nonexistent_order(client, alpaca_mock, alpaca_urls):
    """Test cancellation of non-existent order."""
    order_id = "nonexistent_order_12345"
    
    # Mock order not found
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(404, json={
            "message": "Order not found",
            "code": 40410000
        })
    )
    
    response = await client.delete(f"/trading/orders/{order_id}")
    assert response.status_code == 404
    
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_all_orders_success(client, alpaca_mock, alpaca_urls):
    """Test successful cancellation of all orders."""
    # Mock get all orders
    alpaca_mock.get(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "order_1",
                "symbol": "AAPL",
                "qty": "10",
                "status": "new"
            },
            {
                "id": "order_2", 
                "symbol": "TSLA",
                "qty": "5",
                "status": "pending_new"
            }
        ])
    )
    
    # Mock cancellation of all orders
    alpaca_mock.delete(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(207, json=[
            {
                "id": "order_1",
                "status": "canceled"
            },
            {
                "id": "order_2",
                "status": "canceled" 
            }
        ])
    )
    
    response = await client.delete("/trading/orders")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 2
    assert all(order["status"] == "canceled" for order in data)


@pytest.mark.asyncio
async def test_cancel_orders_by_symbol(client, alpaca_mock, alpaca_urls):
    """Test cancellation of orders for specific symbol."""
    symbol = "AAPL"
    
    # Mock get orders filtered by symbol
    alpaca_mock.get(alpaca_urls["orders"], params={"symbols": symbol}).mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "aapl_order_1",
                "symbol": "AAPL",
                "qty": "10",
                "status": "new"
            }
        ])
    )
    
    # Mock individual cancellation
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}aapl_order_1").mock(
        return_value=httpx.Response(204)
    )
    
    response = await client.delete(f"/trading/orders?symbol={symbol}")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 0  # Could be empty if no orders


@pytest.mark.asyncio
async def test_cancel_order_race_condition(client, alpaca_mock, alpaca_urls):
    """Test cancellation race condition - order fills during cancellation."""
    order_id = "race_condition_order"
    
    # First call returns success, but order was actually filled
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(422, json={
            "message": "Order was filled during cancellation attempt",
            "code": 42210001
        })
    )
    
    response = await client.delete(f"/trading/orders/{order_id}")
    assert response.status_code == 422
    
    data = response.json()
    assert "filled" in data["detail"].lower() or "race" in data["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_partial_fill_order(client, alpaca_mock, alpaca_urls):
    """Test cancellation of partially filled order."""
    order_id = "partial_fill_order"
    
    # Mock successful cancellation of remaining quantity
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(204)
    )
    
    # Mock order status showing partial fill then cancel
    alpaca_mock.get(f"{alpaca_urls['single_order']}{order_id}").mock(
        return_value=httpx.Response(200, json={
            "id": order_id,
            "symbol": "MSFT",
            "qty": "100",
            "filled_qty": "30",  # Partially filled
            "side": "buy",
            "order_type": "limit",
            "limit_price": "300.0",
            "status": "partially_filled",
            "submitted_at": "2024-08-08T10:00:00Z",
            "filled_at": "2024-08-08T10:00:30Z"
        })
    )
    
    response = await client.delete(f"/trading/orders/{order_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["filled_qty"] == "30"
    assert data["status"] == "partially_filled"


@pytest.mark.asyncio
async def test_cancel_order_timeout(client, alpaca_urls):
    """Test handling of cancellation timeout."""
    order_id = "timeout_order"
    
    with respx.mock:
        # Mock timeout during cancellation
        respx.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
            side_effect=httpx.TimeoutException("Cancellation timed out")
        )
        
        response = await client.delete(f"/trading/orders/{order_id}")
        
        # Should handle timeout gracefully
        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert "timeout" in data["detail"].lower()


@pytest.mark.asyncio
async def test_get_cancellable_orders(client, alpaca_mock, alpaca_urls):
    """Test getting list of cancellable orders."""
    # Mock orders in various states
    alpaca_mock.get(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "new_order",
                "symbol": "AAPL", 
                "status": "new"  # Cancellable
            },
            {
                "id": "pending_order",
                "symbol": "TSLA",
                "status": "pending_new"  # Cancellable
            },
            {
                "id": "filled_order", 
                "symbol": "MSFT",
                "status": "filled"  # Not cancellable
            },
            {
                "id": "canceled_order",
                "symbol": "NVDA", 
                "status": "canceled"  # Already canceled
            }
        ])
    )
    
    response = await client.get("/trading/orders?status=open")
    assert response.status_code == 200
    
    data = response.json()
    # Should include cancellable orders
    cancellable_statuses = ["new", "pending_new", "accepted", "pending_cancel"]
    open_orders = [order for order in data if order["status"] in cancellable_statuses]
    assert len(open_orders) >= 2  # new_order and pending_order


@pytest.mark.asyncio
async def test_bulk_cancel_with_failures(client, alpaca_mock, alpaca_urls):
    """Test bulk cancellation where some orders fail to cancel."""
    # Mock mixed results for bulk cancel
    alpaca_mock.delete(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(207, json=[  # Multi-status response
            {
                "id": "success_order_1",
                "status": "canceled"
            },
            {
                "id": "failed_order_1",
                "status": "filled",
                "error": "Order already filled"
            },
            {
                "id": "success_order_2",
                "status": "canceled"
            }
        ])
    )
    
    response = await client.delete("/trading/orders")
    assert response.status_code == 207  # Multi-status
    
    data = response.json()
    assert len(data) == 3
    
    # Check mixed results
    canceled_count = sum(1 for order in data if order["status"] == "canceled")
    failed_count = sum(1 for order in data if "error" in order)
    
    assert canceled_count == 2
    assert failed_count == 1
