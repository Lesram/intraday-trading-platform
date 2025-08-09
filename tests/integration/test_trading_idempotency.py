"""Test idempotency and duplicate request handling."""
import asyncio
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_duplicate_order_prevention(client, alpaca_mock, alpaca_urls):
    """Test that duplicate orders are prevented."""
    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy",
        "client_order_id": "unique_client_order_123"  # Same client ID
    }
    
    # First order submission - success
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": "order_12345",
            "client_order_id": "unique_client_order_123",
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "status": "new",
            "submitted_at": "2024-08-08T10:00:00Z"
        })
    )
    
    # Submit first order
    response1 = await client.post("/trading/orders/market", json=order_data)
    assert response1.status_code == 201
    order1_id = response1.json()["id"]
    
    # Mock duplicate order rejection for same client_order_id
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(422, json={
            "message": "Order with this client_order_id already exists",
            "code": 40210002,
            "existing_order_id": order1_id
        })
    )
    
    # Submit duplicate order with same client_order_id
    response2 = await client.post("/trading/orders/market", json=order_data)
    
    # Should be rejected as duplicate
    assert response2.status_code == 422
    data = response2.json()
    assert "duplicate" in data["detail"].lower() or "already exists" in data["detail"].lower()


@pytest.mark.asyncio
async def test_idempotent_order_status_check(client, alpaca_mock, alpaca_urls):
    """Test that multiple status checks return consistent results."""
    order_id = "test_order_12345"
    
    expected_order = {
        "id": order_id,
        "symbol": "TSLA",
        "qty": "5",
        "side": "buy",
        "status": "filled",
        "filled_qty": "5",
        "filled_avg_price": "800.50",
        "submitted_at": "2024-08-08T10:00:00Z",
        "filled_at": "2024-08-08T10:00:05Z"
    }
    
    # Mock consistent order status response
    alpaca_mock.get(f"{alpaca_urls['single_order']}{order_id}").mock(
        return_value=httpx.Response(200, json=expected_order)
    )
    
    # Make multiple requests
    responses = []
    for _ in range(3):
        response = await client.get(f"/trading/orders/{order_id}")
        responses.append(response)
        await asyncio.sleep(0.1)  # Small delay
    
    # All responses should be identical
    assert all(r.status_code == 200 for r in responses)
    
    orders = [r.json() for r in responses]
    assert all(order["id"] == order_id for order in orders)
    assert all(order["status"] == "filled" for order in orders)
    assert all(order["filled_qty"] == "5" for order in orders)


@pytest.mark.asyncio
async def test_idempotent_account_info_fetch(client, alpaca_mock, alpaca_urls):
    """Test that account info fetches are idempotent."""
    expected_account = {
        "account_number": "123456789",
        "status": "ACTIVE",
        "buying_power": "100000.00",
        "cash": "50000.00",
        "portfolio_value": "100000.00",
        "equity": "100000.00"
    }
    
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json=expected_account)
    )
    
    # Make multiple concurrent requests
    tasks = [client.get("/trading/account") for _ in range(5)]
    responses = await asyncio.gather(*tasks)
    
    # All should succeed with same data
    assert all(r.status_code == 200 for r in responses)
    
    accounts = [r.json() for r in responses]
    assert all(acc["account_number"] == "123456789" for acc in accounts)
    assert all(acc["buying_power"] == "100000.00" for acc in accounts)


@pytest.mark.asyncio
async def test_order_cancellation_idempotency(client, alpaca_mock, alpaca_urls):
    """Test that cancelling same order multiple times is handled properly."""
    order_id = "cancel_test_order"
    
    # First cancellation - success
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(204)
    )
    
    response1 = await client.delete(f"/trading/orders/{order_id}")
    assert response1.status_code == 200
    
    # Second cancellation - already canceled
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        return_value=httpx.Response(422, json={
            "message": "Order is already canceled",
            "code": 42210003
        })
    )
    
    response2 = await client.delete(f"/trading/orders/{order_id}")
    
    # Should handle gracefully (could be 422 or 200 depending on implementation)
    assert response2.status_code in [200, 422]
    if response2.status_code == 422:
        data = response2.json()
        assert "already canceled" in data["detail"].lower()


@pytest.mark.asyncio
async def test_rapid_successive_orders_same_symbol(client, alpaca_mock, alpaca_urls):
    """Test handling of rapid successive orders for same symbol."""
    symbol = "AAPL"
    
    # Mock successful responses for multiple orders
    order_responses = [
        {
            "id": f"order_{i}",
            "symbol": symbol,
            "qty": "1",
            "side": "buy",
            "status": "new",
            "submitted_at": f"2024-08-08T10:00:{i:02d}Z"
        }
        for i in range(5)
    ]
    
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        side_effect=[httpx.Response(201, json=order) for order in order_responses]
    )
    
    # Submit multiple orders rapidly
    tasks = [
        client.post("/trading/orders/market", json={
            "symbol": symbol,
            "qty": 1.0,
            "side": "buy",
            "client_order_id": f"rapid_order_{i}"
        })
        for i in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # All should succeed with unique order IDs
    assert all(r.status_code == 201 for r in responses)
    
    order_ids = [r.json()["id"] for r in responses]
    assert len(set(order_ids)) == 5  # All unique


@pytest.mark.asyncio
async def test_request_timeout_retry_idempotency(client, alpaca_urls):
    """Test idempotency when requests timeout and are retried."""
    order_data = {
        "symbol": "MSFT",
        "qty": 8.0,
        "side": "buy",
        "client_order_id": "timeout_retry_order"
    }
    
    with respx.mock:
        # First attempt times out
        respx.post(alpaca_urls["orders"]).mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        
        response = await client.post("/trading/orders/market", json=order_data)
        
        # Should handle timeout
        assert response.status_code == 503
        
        # Mock successful retry with same client_order_id
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(201, json={
                "id": "timeout_retry_order_id",
                "client_order_id": "timeout_retry_order",
                "symbol": "MSFT",
                "qty": "8",
                "side": "buy",
                "status": "new"
            })
        )
        
        # Retry should succeed
        response2 = await client.post("/trading/orders/market", json=order_data)
        assert response2.status_code == 201


@pytest.mark.asyncio
async def test_concurrent_order_modifications(client, alpaca_mock, alpaca_urls):
    """Test handling of concurrent order modifications."""
    order_id = "concurrent_mod_order"
    
    # Mock order replacement endpoint
    alpaca_mock.patch(f"{alpaca_urls['single_order']}{order_id}").mock(
        return_value=httpx.Response(200, json={
            "id": order_id,
            "symbol": "NVDA",
            "qty": "3",
            "limit_price": "850.0",  # Modified price
            "status": "new",
            "updated_at": "2024-08-08T10:01:00Z"
        })
    )
    
    # Simulate concurrent modification requests
    modification_data = {
        "limit_price": 850.0,
        "qty": 3.0
    }
    
    tasks = [
        client.patch(f"/trading/orders/{order_id}", json=modification_data)
        for _ in range(3)
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # At least one should succeed, others may fail due to race condition
    success_count = sum(
        1 for r in responses 
        if not isinstance(r, Exception) and r.status_code == 200
    )
    assert success_count >= 1


@pytest.mark.asyncio
async def test_state_consistency_after_network_partition(client, alpaca_mock, alpaca_urls):
    """Test state consistency after simulated network issues."""
    order_data = {
        "symbol": "AMZN",
        "qty": 2.0,
        "side": "buy",
        "client_order_id": "partition_test_order"
    }
    
    # Simulate network partition during order submission
    with respx.mock:
        respx.post(alpaca_urls["orders"]).mock(
            side_effect=httpx.ConnectError("Network partition")
        )
        
        response = await client.post("/trading/orders/market", json=order_data)
        assert response.status_code == 503
    
    # Simulate network recovery - check if order exists
    alpaca_mock.get(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "partition_order_id",
                "client_order_id": "partition_test_order",
                "symbol": "AMZN",
                "qty": "2",
                "side": "buy",
                "status": "new"  # Order did get through
            }
        ])
    )
    
    # Check orders after recovery
    response = await client.get("/trading/orders")
    assert response.status_code == 200
    
    orders = response.json()
    partition_order = next(
        (o for o in orders if o.get("client_order_id") == "partition_test_order"),
        None
    )
    
    # Order may or may not exist depending on when partition occurred
    if partition_order:
        assert partition_order["symbol"] == "AMZN"
