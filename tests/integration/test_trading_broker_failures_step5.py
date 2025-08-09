"""
🔥 STEP 5: BROKER FAILURE TESTS
Testing broker failure scenarios as specified in Layer 1 Integration Test Harness
"""
import pytest
import httpx
import respx


@pytest.mark.asyncio
async def test_timeout_then_success_retry(client, alpaca_mock, alpaca_urls):
    """
    Test: Timeout then success
    First POST /v2/orders raises httpx.ReadTimeout, second returns 201.
    Assert your adapter retries and the API returns success.
    """
    # Track how many times the order endpoint is called
    call_count = 0
    
    def side_effect_timeout_then_success(request):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First call: timeout
            raise httpx.ReadTimeout("Request timed out")
        else:
            # Second call: success
            return httpx.Response(201, json={
                "id": "retry_order_12345",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "10",
                "type": "market",
                "status": "new",
                "submitted_at": "2025-08-08T10:00:00Z",
                "filled_avg_price": None,
                "filled_qty": "0"
            })

    # Mock the order endpoint to timeout first, then succeed
    alpaca_mock.post(alpaca_urls["orders"]).mock(side_effect=side_effect_timeout_then_success)

    # Mock account and positions for risk checks
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

    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy"
    }

    response = await client.post("/api/trading/orders/market", json=order_data)

    # Should succeed after retry
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "submitted"
    assert "order_ref" in data
    
    # Verify retry happened (endpoint was called twice)
    assert call_count == 2


@pytest.mark.asyncio  
async def test_permanent_5xx_returns_502_503(client, alpaca_mock, alpaca_urls):
    """
    Test: Permanent 5xx
    Return 500 three times; assert API maps to 502/503 and includes a friendly message.
    """
    # Mock account and positions first
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

    # Mock order endpoint to consistently return 500
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(500, json={
            "message": "Internal server error",
            "code": 50000000
        })
    )

    order_data = {
        "symbol": "TSLA",
        "qty": 5.0,
        "side": "buy"
    }

    response = await client.post("/api/trading/orders/market", json=order_data)

    # Should map to 502 or 503 with friendly message
    assert response.status_code in [502, 503]
    data = response.json()
    assert "detail" in data
    
    # Should have friendly message about broker being unavailable
    detail_lower = data["detail"].lower()
    assert any(keyword in detail_lower for keyword in [
        "broker", "unavailable", "temporary", "server", "try again"
    ])


@pytest.mark.asyncio
async def test_broker_reject_4xx_with_rejection_status(client, alpaca_mock, alpaca_urls):
    """
    Test: Broker reject 4xx  
    Return 422 with an Alpaca-like error payload; assert API returns 400/422 
    with status:"rejected" and reason.
    """
    # Mock account and positions first
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

    # Mock order endpoint to return 422 with Alpaca-like error payload
    alpaca_mock.post(alpaca_urls["orders"]).mock(
        return_value=httpx.Response(422, json={
            "message": "Invalid order parameters",
            "code": 40210000,
                        "details": "Symbol BADSYMBOL is not tradable"
        })
    )

    order_data = {
        "symbol": "BADSYMBOL",  # 9 chars - passes validation but broker rejects
        "qty": 10.0,
        "side": "buy"
    }

    response = await client.post("/api/trading/orders/market", json=order_data)

    # Should return 400/422 with rejection status and reason
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")
    
    assert response.status_code in [400, 422]
    data = response.json()
    
    # FastAPI wraps HTTPException detail in a "detail" key
    if "detail" in data:
        detail = data["detail"]
    else:
        detail = data
    
    # Should have rejection status and reason
    assert detail["status"] == "rejected"
    assert "reasons" in detail or "reason" in detail
    
    # Should include the broker's error message
    if "reasons" in detail:
        reason_text = " ".join(detail["reasons"]).lower()
    else:
        reason_text = str(detail["reason"]).lower()
        
    assert any(keyword in reason_text for keyword in [
        "invalid", "tradable", "symbol", "parameters"
    ])
