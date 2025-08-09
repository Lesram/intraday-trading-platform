"""
🔥 STEP 7: IDEMPOTENCY TESTS (Layer 1 Requirement)
Ensure duplicate requests don't create multiple orders
"""
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_duplicate_order_idempotency_step7(client, alpaca_mock, alpaca_urls):
    """
    Test: Idempotency - Ensure duplicate requests don't create multiple orders (Step 7)
    Use respx to mock same order_id being returned.
    Call POST twice with same client_order_id, assert only one Alpaca call.
    """
    order_id = "step7_idempotency_order_12345"
    
    # Mock account and positions for orders
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

    # Mock POST /v2/orders → same order_id returned both times
    orders_route = alpaca_mock.post(alpaca_urls["orders"])
    orders_route.mock(
        return_value=httpx.Response(201, json={
            "id": order_id,
            "symbol": "AAPL",
            "side": "buy",
            "qty": "5",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    # Same order data for both requests
    order_data = {
        "symbol": "AAPL",
        "qty": 5.0,
        "side": "buy",
        "client_order_id": "IDEMPOTENCY_TEST_123"  # Same client_order_id
    }

    # First POST request
    response1 = await client.post("/api/trading/orders/market", json=order_data)
    assert response1.status_code == 200
    
    data1 = response1.json()
    assert data1["status"] == "submitted"
    first_order_ref = data1["order_ref"]

    # Second POST request with same client_order_id
    response2 = await client.post("/api/trading/orders/market", json=order_data)
    assert response2.status_code == 200
    
    data2 = response2.json()
    assert data2["status"] == "submitted"
    second_order_ref = data2["order_ref"]

    # Should return same order reference (idempotency)
    assert first_order_ref == second_order_ref, f"Expected same order reference, got {first_order_ref} and {second_order_ref}"

    # Assert only one Alpaca call was made (idempotency working)
    assert orders_route.called
    assert len(orders_route.calls) == 1, f"Expected exactly 1 Alpaca call, got {len(orders_route.calls)}"
