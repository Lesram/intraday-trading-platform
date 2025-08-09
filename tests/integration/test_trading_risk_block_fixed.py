"""Test risk management blocking scenarios."""
import pytest
import httpx


@pytest.mark.asyncio
async def test_order_blocked_by_position_size_limit(client, alpaca_mock, alpaca_urls):
    """Test order blocked when position size exceeds limit."""
    # Mock account with limited buying power to trigger position size rejection
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "1000.00",  # Very low buying power
            "cash": "1000.00",
            "portfolio_value": "1000.00",
            "equity": "1000.00",
            "last_equity": "1000.00",
            "multiplying_power": "1",
            "regt_buying_power": "1000.00",
            "daytrading_buying_power": "1000.00",
            "daytrade_count": 0,  # Required field!
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
    
    # Large order that should exceed position size limits
    large_order_data = {
        "symbol": "AAPL",  # $150 * 100 = $15,000 order on $1,000 portfolio  
        "qty": 100.0,
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=large_order_data)
    
    # Should return 200 but with rejected status
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["risk_decision"] == "REJECTED"
    assert len(data["reasons"]) > 0
    
    # Verify the rejection reason mentions risk/position size
    rejection_reasons = " ".join(data["reasons"]).lower()
    assert any(keyword in rejection_reasons for keyword in ["risk", "position", "size", "limit", "buying power"])


@pytest.mark.asyncio 
async def test_order_blocked_by_portfolio_heat_limit(client, alpaca_mock, alpaca_urls):
    """Test order blocked when portfolio heat limit exceeded."""
    # Mock account with existing positions that would exceed heat limit
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
            "daytrade_count": 0,  # Required field!
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

    # Mock existing high-risk positions (80% of portfolio in one volatile stock)
    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "symbol": "TSLA",
                "qty": "40",
                "side": "long",
                "market_value": "8000.00",  # 80% of portfolio
                "unrealized_pl": "-500.00",
                "unrealized_plpc": "-0.0625",  # -6.25% unrealized loss
                "cost_basis": "8500.00",
                "avg_entry_price": "212.50"
            }
        ])
    )

    # Try to add another risky position that would push heat over limit
    order_data = {
        "symbol": "NVDA",  # Another volatile tech stock
        "qty": 5.0,  # $400 * 5 = $2,000 more exposure
        "side": "buy"
    }

    response = await client.post("/api/trading/orders/market", json=order_data) 

    # Should return 200 with either rejected or modified status due to risk limits
    assert response.status_code == 200
    data = response.json()
    # Risk system should either reject or modify the order
    assert data["status"] in ["rejected", "submitted"]  # submitted means it was modified and accepted
    assert data["risk_decision"] in ["REJECTED", "MODIFIED"]
    
    # Should have risk-related reasons
    assert "reasons" in data
    assert len(data["reasons"]) > 0
@pytest.mark.asyncio
async def test_order_blocked_by_daily_loss_limit(client, alpaca_mock, alpaca_urls):
    """Test order blocked when daily loss limit hit."""
    # Mock account showing significant daily losses
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789", 
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "50000.00",
            "cash": "50000.00",
            "portfolio_value": "90000.00",  # Down $10k from yesterday
            "equity": "90000.00",
            "last_equity": "100000.00",  # Previous day equity - shows $10k loss
            "multiplying_power": "1",
            "regt_buying_power": "50000.00",
            "daytrading_buying_power": "50000.00",
            "daytrade_count": 0,  # Required field!
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

    # Mock positions showing unrealized losses
    alpaca_mock.get(alpaca_urls["positions"]).mock(
        return_value=httpx.Response(200, json=[
            {
                "symbol": "SPY",
                "qty": "200",
                "side": "long",
                "market_value": "40000.00",
                "unrealized_pl": "-5000.00",  # $5k unrealized loss
                "unrealized_plpc": "-0.111",
                "cost_basis": "45000.00",
                "avg_entry_price": "225.00"
            }
        ])
    )

    order_data = {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    
    # Should return 200 but with rejected status due to daily loss limit
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected" 
    assert data["risk_decision"] == "REJECTED"
    
    # Check that rejection mentions daily loss
    rejection_reasons = " ".join(data["reasons"]).lower()
    assert any(keyword in rejection_reasons for keyword in ["loss", "daily", "limit", "drawdown"])


@pytest.mark.asyncio
async def test_order_blocked_insufficient_buying_power(client, alpaca_mock, alpaca_urls):
    """Test order blocked due to insufficient buying power."""
    # Mock account with very low buying power
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "100.00",  # Only $100 buying power
            "cash": "100.00",
            "portfolio_value": "10000.00", 
            "equity": "10000.00",
            "last_equity": "10000.00",
            "multiplying_power": "1",
            "regt_buying_power": "100.00",
            "daytrading_buying_power": "100.00",
            "daytrade_count": 0,  # Required field!
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
    
    # Order that exceeds buying power
    order_data = {
        "symbol": "AAPL",  # $150 * 10 = $1,500 order with only $100 available
        "qty": 10.0,
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["risk_decision"] == "REJECTED"
    
    # Should mention buying power or funds
    rejection_reasons = " ".join(data["reasons"]).lower()
    assert any(keyword in rejection_reasons for keyword in ["buying power", "funds", "insufficient", "cash"])


@pytest.mark.asyncio
async def test_risk_service_prevents_alpaca_calls(client, alpaca_mock, alpaca_urls):
    """Test that when risk blocks order, no call to Alpaca POST /v2/orders occurs."""
    # Set up conditions that will definitely trigger risk rejection
    alpaca_mock.get(alpaca_urls["account"]).mock(
        return_value=httpx.Response(200, json={
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "100.00",  # Very low
            "cash": "100.00",
            "portfolio_value": "100.00",
            "equity": "100.00",
            "last_equity": "100.00",
            "multiplying_power": "1",
            "regt_buying_power": "100.00",
            "daytrading_buying_power": "100.00",
            "daytrade_count": 0,  # Required field!
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
    
    # Clear any default POST mocks to verify no calls
    alpaca_mock.post(alpaca_urls["orders"]).pass_through = True
    
    # Order that will definitely be rejected
    order_data = {
        "symbol": "AAPL",
        "qty": 100.0,  # $15,000 order on $100 portfolio
        "side": "buy"
    }
    
    response = await client.post("/api/trading/orders/market", json=order_data)
    
    # Verify risk rejection
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    
    # Verify no POST calls were made to Alpaca orders endpoint
    post_calls = [call for call in alpaca_mock.calls if call.request.method == "POST" and "orders" in str(call.request.url)]
    assert len(post_calls) == 0, f"Expected no POST calls to orders endpoint, but found: {post_calls}"
