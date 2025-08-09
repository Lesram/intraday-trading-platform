"""
🔥 STEP 8: CONCURRENT REQUESTS TESTS (Layer 1 Requirement)
Test system behavior under concurrent load
"""
import pytest
import httpx
import respx
import asyncio


@pytest.mark.asyncio
async def test_concurrent_orders_step8(client, alpaca_mock, alpaca_urls):
    """
    Test: Multiple concurrent orders (Step 8 requirement)
    Submit multiple orders concurrently and verify all are processed correctly.
    Ensure no race conditions or data corruption occurs.
    """
    # Mock account and positions
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "10000.00",
            "cash": "5000.00",
            "portfolio_value": "10000.00",
            "equity": "10000.00",
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "10000.00",
            "daytrading_buying_power": "10000.00",
            "daytrade_count": 0,
            "pattern_day_trader": False,
            "created_at": "2024-01-01T00:00:00Z",
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "sma": "0",
            "accrued_fees": "0"
        })
    )

    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )

    # Mock successful order submissions - return different order IDs
    order_counter = 0
    def create_order_response(*args, **kwargs):
        nonlocal order_counter
        order_counter += 1
        return httpx.Response(201, json={
            "id": f"step8_concurrent_order_{order_counter}",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1",
            "type": "market", 
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })

    orders_route = alpaca_mock.post(alpaca_urls["orders"])
    orders_route.mock(side_effect=create_order_response)

    # Create 5 concurrent order requests
    concurrent_orders = []
    for i in range(5):
        order_data = {
            "symbol": "AAPL",
            "qty": 1.0,
            "side": "buy"
        }
        concurrent_orders.append(
            client.post("/api/trading/orders/market", json=order_data)
        )

    # Execute all orders concurrently
    responses = await asyncio.gather(*concurrent_orders, return_exceptions=True)

    # Verify all requests succeeded
    successful_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            pytest.fail(f"Order {i} failed with exception: {response}")
        
        assert response.status_code == 200, f"Order {i} returned status {response.status_code}"
        
        data = response.json()
        assert data["status"] == "submitted", f"Order {i} has status {data['status']}"
        assert "order_ref" in data, f"Order {i} missing order_ref"
        
        successful_responses.append(data)

    # Verify all orders have unique order references
    order_refs = [resp["order_ref"] for resp in successful_responses]
    assert len(set(order_refs)) == 5, f"Expected 5 unique order references, got {len(set(order_refs))}"

    # Verify all Alpaca calls were made
    assert orders_route.called
    assert len(orders_route.calls) == 5, f"Expected 5 Alpaca calls, got {len(orders_route.calls)}"


@pytest.mark.asyncio 
async def test_concurrent_risk_limits_step8(client, alpaca_mock, alpaca_urls):
    """
    Test: Concurrent orders hitting risk limits (Step 8 requirement)
    Submit multiple large orders concurrently that would exceed risk limits.
    Verify risk management works correctly under concurrent load.
    """
    # Mock account and positions
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE", 
            "currency": "USD",
            "buying_power": "10000.00",
            "cash": "5000.00",
            "portfolio_value": "10000.00",
            "equity": "10000.00", 
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "10000.00",
            "daytrading_buying_power": "10000.00",
            "daytrade_count": 0,
            "pattern_day_trader": False,
            "created_at": "2024-01-01T00:00:00Z",
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "sma": "0",
            "accrued_fees": "0"
        })
    )

    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[])
    )

    # Mock Alpaca order submission (shouldn't be reached due to risk limits)
    orders_route = alpaca_mock.post(alpaca_urls["orders"])
    orders_route.mock(
        return_value=httpx.Response(201, json={
            "id": "should_not_reach_alpaca",
            "symbol": "AAPL", 
            "side": "buy",
            "qty": "100",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    # Create 3 concurrent large orders that should exceed position size limits
    # Each order for 100 shares of AAPL (~$15,000 each) should exceed 10% position limit
    concurrent_orders = []
    for i in range(3):
        order_data = {
            "symbol": "AAPL", 
            "qty": 100.0,  # Large quantity to trigger risk limits
            "side": "buy"
        }
        concurrent_orders.append(
            client.post("/api/trading/orders/market", json=order_data)
        )

    # Execute all orders concurrently
    responses = await asyncio.gather(*concurrent_orders, return_exceptions=True)

    # Verify responses (should be rejected or modified by risk management)
    rejected_count = 0
    modified_count = 0
    
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            pytest.fail(f"Order {i} failed with exception: {response}")
        
        assert response.status_code == 200, f"Order {i} returned status {response.status_code}"
        
        data = response.json()
        # Orders should be either rejected or modified due to risk limits
        if data["status"] == "rejected":
            rejected_count += 1
            assert "reasons" in data, f"Rejected order {i} missing reasons"
        elif data["status"] == "submitted":
            # If submitted, it should be modified (approved_qty < original qty)
            assert data["approved_qty"] < 100.0, f"Order {i} should be modified, got approved_qty={data['approved_qty']}"
            modified_count += 1

    # At least some orders should be affected by risk management
    assert (rejected_count + modified_count) >= 2, f"Expected at least 2 orders affected by risk management, got {rejected_count} rejected + {modified_count} modified"

    # Should not make unnecessary Alpaca calls for rejected orders
    alpaca_calls = len(orders_route.calls) if orders_route.called else 0
    assert alpaca_calls <= modified_count, f"Too many Alpaca calls ({alpaca_calls}) for modified orders ({modified_count})"
