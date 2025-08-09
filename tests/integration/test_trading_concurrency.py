"""Test concurrent trading scenarios."""
import asyncio
import pytest
import httpx


@pytest.mark.asyncio
async def test_concurrent_order_submissions(client, alpaca_mock, alpaca_urls):
    """Test multiple concurrent order submissions."""
    # Mock successful responses for concurrent orders
    order_responses = [
        {
            "id": f"concurrent_order_{i}",
            "symbol": f"SYMBOL{i}",
            "qty": str(i + 1),
            "side": "buy",
            "status": "new",
            "submitted_at": f"2024-08-08T10:00:{i:02d}Z"
        }
        for i in range(10)
    ]
    
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        side_effect=[httpx.Response(201, json=order) for order in order_responses]
    )
    
    # Create concurrent order tasks
    order_tasks = [
        client.post("/trading/orders/market", json={
            "symbol": f"SYMBOL{i}",
            "qty": float(i + 1),
            "side": "buy",
            "client_order_id": f"concurrent_{i}"
        })
        for i in range(10)
    ]
    
    # Execute all orders concurrently
    responses = await asyncio.gather(*order_tasks)
    
    # All should succeed
    assert all(r.status_code == 201 for r in responses)
    
    # Verify unique order IDs
    order_ids = [r.json()["id"] for r in responses]
    assert len(set(order_ids)) == 10  # All unique


@pytest.mark.asyncio
async def test_concurrent_account_queries(client, alpaca_mock, alpaca_urls):
    """Test multiple concurrent account information queries."""
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
    
    # Execute multiple concurrent requests
    account_tasks = [client.get("/trading/account") for _ in range(20)]
    responses = await asyncio.gather(*account_tasks)
    
    # All should succeed with same data
    assert all(r.status_code == 200 for r in responses)
    
    accounts = [r.json() for r in responses]
    assert all(acc["account_number"] == "123456789" for acc in accounts)


@pytest.mark.asyncio
async def test_concurrent_position_queries(client, alpaca_mock, alpaca_urls):
    """Test multiple concurrent position queries."""
    expected_positions = [
        {
            "symbol": "AAPL",
            "qty": "100",
            "side": "long",
            "market_value": "15000.00",
            "unrealized_pl": "500.00"
        },
        {
            "symbol": "TSLA", 
            "qty": "50",
            "side": "long",
            "market_value": "40000.00",
            "unrealized_pl": "-1000.00"
        }
    ]
    
    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=expected_positions)
    )
    
    # Execute concurrent position queries
    position_tasks = [client.get("/trading/positions") for _ in range(15)]
    responses = await asyncio.gather(*position_tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    
    positions_lists = [r.json() for r in responses]
    assert all(len(positions) == 2 for positions in positions_lists)


@pytest.mark.asyncio
async def test_concurrent_order_status_checks(client, alpaca_mock, alpaca_urls):
    """Test concurrent order status checks for same order."""
    order_id = "status_check_order"
    
    expected_order = {
        "id": order_id,
        "symbol": "MSFT",
        "qty": "10",
        "side": "buy",
        "status": "filled",
        "filled_qty": "10",
        "filled_avg_price": "340.50"
    }
    
    alpaca_mock.get(f"{alpaca_urls['single_order']}{order_id}").mock(
        return_value=httpx.Response(200, json=expected_order)
    )
    
    # Check same order status concurrently
    status_tasks = [
        client.get(f"/trading/orders/{order_id}")
        for _ in range(25)
    ]
    
    responses = await asyncio.gather(*status_tasks)
    
    # All should return same result
    assert all(r.status_code == 200 for r in responses)
    
    orders = [r.json() for r in responses]
    assert all(order["id"] == order_id for order in orders)
    assert all(order["status"] == "filled" for order in orders)


@pytest.mark.asyncio
async def test_concurrent_order_cancellations(client, alpaca_mock, alpaca_urls):
    """Test concurrent cancellation attempts on different orders."""
    order_ids = [f"cancel_order_{i}" for i in range(5)]
    
    # Mock successful cancellations
    for order_id in order_ids:
        alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
            return_value=httpx.Response(204)
        )
        
        alpaca_mock.get(f"{alpaca_urls['single_order']}{order_id}").mock(
            return_value=httpx.Response(200, json={
                "id": order_id,
                "status": "canceled",
                "canceled_at": "2024-08-08T10:01:00Z"
            })
        )
    
    # Execute concurrent cancellations
    cancel_tasks = [
        client.delete(f"/trading/orders/{order_id}")
        for order_id in order_ids
    ]
    
    responses = await asyncio.gather(*cancel_tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    
    canceled_orders = [r.json() for r in responses]
    assert all(order["status"] == "canceled" for order in canceled_orders)


@pytest.mark.asyncio
async def test_concurrent_same_order_cancellation(client, alpaca_mock, alpaca_urls):
    """Test concurrent cancellation attempts on same order."""
    order_id = "double_cancel_order"
    
    # First cancellation succeeds
    cancel_responses = [
        httpx.Response(204),  # First succeeds
        httpx.Response(422, json={  # Rest fail - already canceled
            "message": "Order is already canceled",
            "code": 42210003
        })
    ]
    
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}{order_id}").mock(
        side_effect=cancel_responses * 10  # Enough for all concurrent requests
    )
    
    # Try to cancel same order concurrently
    cancel_tasks = [
        client.delete(f"/trading/orders/{order_id}")
        for _ in range(5)
    ]
    
    responses = await asyncio.gather(*cancel_tasks, return_exceptions=True)
    
    # Some should succeed (204->200), others fail (422) due to race condition
    status_codes = [
        r.status_code for r in responses 
        if not isinstance(r, Exception)
    ]
    
    assert 200 in status_codes or 204 in status_codes  # At least one success
    # May have 422 responses for already canceled


@pytest.mark.asyncio
async def test_load_test_order_submissions(client, alpaca_mock, alpaca_urls):
    """Test system under high concurrent load."""
    num_orders = 50
    
    # Mock responses for load test
    order_responses = [
        httpx.Response(201, json={
            "id": f"load_test_order_{i}",
            "symbol": "AAPL",
            "qty": "1",
            "side": "buy",
            "status": "new"
        })
        for i in range(num_orders)
    ]
    
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        side_effect=order_responses
    )
    
    # Create high-load concurrent requests
    start_time = asyncio.get_event_loop().time()
    
    order_tasks = [
        client.post("/trading/orders/market", json={
            "symbol": "AAPL",
            "qty": 1.0,
            "side": "buy",
            "client_order_id": f"load_test_{i}"
        })
        for i in range(num_orders)
    ]
    
    responses = await asyncio.gather(*order_tasks)
    end_time = asyncio.get_event_loop().time()
    
    # All should complete successfully
    assert all(r.status_code == 201 for r in responses)
    
    # Performance check - should complete within reasonable time
    duration = end_time - start_time
    assert duration < 30.0  # Should complete within 30 seconds
    
    # Check throughput
    throughput = num_orders / duration
    print(f"Throughput: {throughput:.2f} orders/second")


@pytest.mark.asyncio
async def test_mixed_concurrent_operations(client, alpaca_mock, alpaca_urls):
    """Test mixed concurrent operations: orders, queries, cancellations."""
    # Setup mocks for various operations
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": "mixed_ops_order",
            "symbol": "NVDA",
            "qty": "5",
            "status": "new"
        })
    )
    
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "buying_power": "100000.00"
        })
    )
    
    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )
    
    alpaca_mock.delete(f"{alpaca_urls['cancel_order']}existing_order").mock(
        return_value=httpx.Response(204)
    )
    
    # Create mixed operation tasks
    mixed_tasks = []
    
    # Add order submissions
    for i in range(10):
        mixed_tasks.append(
            client.post("/trading/orders/market", json={
                "symbol": "NVDA",
                "qty": 5.0,
                "side": "buy",
                "client_order_id": f"mixed_{i}"
            })
        )
    
    # Add account queries
    for _ in range(10):
        mixed_tasks.append(client.get("/trading/account"))
    
    # Add position queries
    for _ in range(10):
        mixed_tasks.append(client.get("/trading/positions"))
    
    # Add cancellations
    for _ in range(5):
        mixed_tasks.append(client.delete("/trading/orders/existing_order"))
    
    # Execute all operations concurrently
    responses = await asyncio.gather(*mixed_tasks, return_exceptions=True)
    
    # Most should succeed
    successful_responses = [
        r for r in responses
        if not isinstance(r, Exception) and 200 <= r.status_code < 300
    ]
    
    # Expect high success rate
    success_rate = len(successful_responses) / len(responses)
    assert success_rate > 0.8  # At least 80% success rate


@pytest.mark.asyncio
async def test_concurrent_risk_checks(client, alpaca_mock):
    """Test concurrent risk checks don't interfere with each other."""
    # Create multiple risk check requests
    risk_check_tasks = [
        client.post("/trading/risk-check", json={
            "symbol": f"SYMBOL{i}",
            "qty": float(i + 1),
            "side": "buy",
            "order_type": "market"
        })
        for i in range(15)
    ]
    
    responses = await asyncio.gather(*risk_check_tasks)
    
    # All risk checks should complete
    assert all(r.status_code == 200 for r in responses)
    
    risk_results = [r.json() for r in responses]
    assert all("approved" in result for result in risk_results)
    assert all("risk_score" in result for result in risk_results)
