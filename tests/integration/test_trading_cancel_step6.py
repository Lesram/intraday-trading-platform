"""
🔥 STEP 6: CANCEL FLOW TESTS (Layer 1 Requirement)
Testing order cancellation workflow as specified
"""
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_order_cancel_flow_step6(client, alpaca_mock, alpaca_urls):
    """
    Test: Cancel flow (Step 6 requirement)
    Mock POST /v2/orders → accepted; then DELETE /v2/orders/{id} → 204.
    Call POST /api/trading/orders/market to get an order_id.
    Call POST /api/trading/orders/{order_id}/cancel.
    Assert 200 and a success message; assert DELETE was called once.
    """
    order_id = "step6_cancel_test_order_12345"
    
    # Mock account and positions for initial order
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

    # Mock POST /v2/orders → accepted
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(201, json={
            "id": order_id,
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "type": "market",
            "status": "new",
            "submitted_at": "2025-08-08T10:00:00Z"
        })
    )

    # Mock DELETE /v2/orders/{id} → 204
    delete_route = alpaca_mock.delete(f"{alpaca_urls['single_order']}{order_id}")
    delete_route.mock(return_value=httpx.Response(204))

    # Step 1: Call POST /api/trading/orders/market to get an order_id
    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy"
    }

    create_response = await client.post("/api/trading/orders/market", json=order_data)
    assert create_response.status_code == 200
    
    create_data = create_response.json()
    assert create_data["status"] == "submitted"
    assert "order_ref" in create_data
    
    created_order_id = create_data["order_ref"]

    # Step 2: Call POST /api/trading/orders/{order_id}/cancel
    cancel_request_body = {"reason": "Test cancellation"}
    cancel_response = await client.post(f"/api/trading/orders/{created_order_id}/cancel", json=cancel_request_body)
    
    # Assert 200 and a success message
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()
    
    # Should have success message
    assert "message" in cancel_data or "status" in cancel_data
    if "message" in cancel_data:
        assert "cancel" in cancel_data["message"].lower() or "success" in cancel_data["message"].lower()
    if "status" in cancel_data:
        assert cancel_data["status"] in ["cancelled", "canceled", "success"]

    # Assert DELETE was called once
    assert delete_route.called
    assert len(delete_route.calls) == 1
