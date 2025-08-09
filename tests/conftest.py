"""Test configuration and fixtures."""
import os
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
import respx
import httpx
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from adapters.alpaca_client import AlpacaClient
from backend.main import create_app
from services.broker_service import BrokerService
from services.risk_service import RiskService


@pytest.fixture
def settings_overrides(monkeypatch):
    """Override environment variables for testing."""
    test_env_vars = {
        # Alpaca Configuration
        "APCA_API_KEY_ID": "test_alpaca_key_12345",
        "APCA_API_SECRET_KEY": "test_alpaca_secret_67890",
        "APCA_API_BASE_URL": "https://paper-api.alpaca.markets",
        
        # Environment
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        
        # Risk Management - Deterministic values
        "MAX_POSITION_SIZE": "0.10",  # 10% max position
        "MAX_PORTFOLIO_HEAT": "0.02",  # 2% portfolio heat
        "DAILY_LOSS_LIMIT": "0.05",   # 5% daily loss limit
        "MAX_DRAWDOWN_LIMIT": "0.15", # 15% max drawdown
        
        # Trading Configuration
        "ENABLE_PAPER_TRADING": "true",
        "ENABLE_RISK_MANAGEMENT": "true",
        "ENABLE_AUTO_TRADING": "false",
        
        # Logging
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        
        # Performance
        "REQUEST_TIMEOUT": "30",
        "MAX_CONCURRENT_REQUESTS": "10"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_env_vars


@pytest_asyncio.fixture
async def test_app(settings_overrides):
    """Create a fresh FastAPI app instance for testing."""
    from services.broker_service import broker_service
    from services.risk_service import risk_service
    
    # Initialize services for testing
    await broker_service.initialize()
    
    app = create_app()
    
    yield app
    
    # Cleanup services after testing
    await broker_service.shutdown()


@pytest_asyncio.fixture
async def client(test_app):
    """Create async HTTP client for FastAPI testing."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client


@pytest.fixture
def alpaca_urls():
    """Alpaca API endpoint URLs for mocking."""
    base_url = "https://paper-api.alpaca.markets"
    return {
        "account": f"{base_url}/v2/account",
        "positions": f"{base_url}/v2/positions",
        "orders": f"{base_url}/v2/orders",
        "orders_by_status": f"{base_url}/v2/orders",
        "single_order": f"{base_url}/v2/orders/",  # append order_id
        "cancel_order": f"{base_url}/v2/orders/",  # append order_id  
        "assets": f"{base_url}/v2/assets",
        "market_data": f"{base_url}/v2/stocks"
    }


@pytest.fixture(autouse=True)
def alpaca_mock(alpaca_urls):
    """Auto-use fixture that provides default Alpaca API mocks."""
    with respx.mock:
        # Default account response
        respx.get(alpaca_urls["account"]).mock(
            return_value=httpx.Response(200, json={
                "account_number": "123456789",
                "status": "ACTIVE",
                "currency": "USD",
                "buying_power": "100000.00",
                "cash": "50000.00",
                "portfolio_value": "100000.00",
                "equity": "100000.00",
                "last_equity": "100000.00",
                "multiplying_power": "1",
                "regt_buying_power": "100000.00",
                "daytrading_buying_power": "100000.00",
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
        
        # Default positions response (empty)
        respx.get(alpaca_urls["positions"]).mock(
            return_value=httpx.Response(200, json=[])
        )
        
        # Default orders response (empty)
        respx.get(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(200, json=[])
        )
        
        # Default order submission response
        respx.post(alpaca_urls["orders"]).mock(
            return_value=httpx.Response(201, json={
                "id": "test_order_12345",
                "client_order_id": "test_client_order_67890",
                "created_at": "2024-08-08T10:00:00Z",
                "updated_at": "2024-08-08T10:00:00Z",
                "submitted_at": "2024-08-08T10:00:00Z",
                "filled_at": None,
                "expired_at": None,
                "canceled_at": None,
                "failed_at": None,
                "replaced_at": None,
                "replaced_by": None,
                "replaces": None,
                "asset_id": "test_asset_id",
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "notional": None,
                "qty": "10",
                "filled_qty": "0",
                "filled_avg_price": None,
                "order_class": "simple",
                "order_type": "market",
                "type": "market",
                "side": "buy",
                "time_in_force": "day",
                "limit_price": None,
                "stop_price": None,
                "status": "new",
                "extended_hours": False,
                "legs": None,
                "trail_percent": None,
                "trail_price": None,
                "hwm": None
            })
        )
        
        yield respx


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Mock(
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
        alpaca_base_url="https://paper-api.alpaca.markets",
        environment="test",
        max_position_size=10000.0,
        max_daily_loss=5000.0,
        max_portfolio_heat=0.02,
        log_level="INFO"
    )


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca client for testing."""
    client = AsyncMock(spec=AlpacaClient)
    client.get_account.return_value = {
        "account_number": "123456789",
        "status": "ACTIVE",
        "buying_power": "100000.0",
        "cash": "50000.0",
        "portfolio_value": "100000.0",
        "equity": "100000.0",
        "daytrade_count": 0,
        "pattern_day_trader": False
    }
    client.get_positions.return_value = []
    client.submit_order.return_value = {
        "id": "test_order_id",
        "status": "new",
        "symbol": "AAPL",
        "qty": "10",
        "side": "buy",
        "order_type": "market",
        "submitted_at": "2024-01-01T10:00:00Z"
    }
    client.get_order.return_value = {
        "id": "test_order_id",
        "status": "filled",
        "symbol": "AAPL",
        "qty": "10",
        "filled_qty": "10",
        "side": "buy",
        "order_type": "market",
        "filled_avg_price": "150.0",
        "submitted_at": "2024-01-01T10:00:00Z",
        "filled_at": "2024-01-01T10:00:05Z"
    }
    return client


@pytest.fixture
def mock_broker_service(mock_alpaca_client):
    """Mock broker service for testing."""
    service = AsyncMock(spec=BrokerService)
    service.alpaca_client = mock_alpaca_client
    return service


@pytest.fixture
def mock_risk_service():
    """Mock risk service for testing."""
    service = AsyncMock(spec=RiskService)
    service.check_trade_risk.return_value = {
        "approved": True,
        "reason": "Trade approved",
        "risk_score": 0.5,
        "position_size_limit": 10000.0,
        "portfolio_heat": 0.01
    }
    return service


@pytest_asyncio.fixture
async def app_client(test_app):
    """FastAPI async test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(test_app):
    """Synchronous test client for non-async tests."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def sample_market_order():
    """Sample market order data for testing."""
    return {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "buy",
        "oco_stop": None
    }


@pytest.fixture
def sample_limit_order():
    """Sample limit order data for testing."""
    return {
        "symbol": "TSLA",
        "qty": 5.0,
        "side": "sell",
        "limit_price": 200.0
    }


@pytest.fixture
def sample_stop_loss_order():
    """Sample stop-loss order data for testing."""
    return {
        "symbol": "MSFT",
        "qty": 8.0,
        "side": "sell",
        "stop_price": 300.0
    }


@pytest.fixture
def sample_account_data():
    """Sample account data for testing."""
    return {
        "account_number": "123456789",
        "status": "ACTIVE",
        "buying_power": "100000.00",
        "cash": "50000.00", 
        "portfolio_value": "100000.00",
        "equity": "100000.00",
        "daytrade_count": 0,
        "pattern_day_trader": False
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    return {
        "symbol": "AAPL",
        "qty": "100",
        "side": "long",
        "market_value": "15000.00",
        "cost_basis": "14500.00",
        "unrealized_pl": "500.00",
        "unrealized_plpc": "0.0345",
        "current_price": "150.00",
        "lastday_price": "148.00",
        "change_today": "0.0135"
    }
