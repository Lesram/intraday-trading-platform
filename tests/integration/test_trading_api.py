"""Integration tests for the FastAPI trading endpoints."""
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import create_app


class TestTradingEndpoints:
    """Integration tests for trading API endpoints."""

    @pytest.fixture
    async def app_client(self):
        """Create FastAPI test client."""
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client

    @pytest.fixture
    def mock_services(self):
        """Mock all services for integration testing."""
        with patch('api.routers.trading.broker_service') as mock_broker, \
             patch('api.routers.trading.risk_service') as mock_risk:

            # Configure mock risk service
            mock_risk.check_trade_risk.return_value = {
                "approved": True,
                "reason": "Trade approved",
                "risk_score": 0.3,
                "position_size_limit": 10000.0,
                "portfolio_heat": 0.015
            }

            # Configure mock broker service
            mock_broker.submit_order.return_value = AsyncMock(
                id="internal_123",
                symbol="AAPL",
                quantity=10,
                side="buy",
                state="SUBMITTED",
                alpaca_order_id="alpaca_456"
            )

            mock_broker.get_order_status.return_value = AsyncMock(
                id="internal_123",
                state="FILLED",
                filled_quantity=10,
                filled_price=150.25
            )

            mock_broker.cancel_order.return_value = True

            yield {
                "broker": mock_broker,
                "risk": mock_risk
            }

    async def test_submit_market_order_success(self, app_client, mock_services):
        """Test successful market order submission."""
        order_data = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day"
        }

        response = await app_client.post("/api/orders/market", json=order_data)

        assert response.status_code == 201
        data = response.json()

        assert data["order_id"] == "internal_123"
        assert data["symbol"] == "AAPL"
        assert data["quantity"] == 10
        assert data["status"] == "SUBMITTED"

        # Verify services were called
        mock_services["risk"].check_trade_risk.assert_called_once()
        mock_services["broker"].submit_order.assert_called_once()

    async def test_submit_order_risk_rejection(self, app_client, mock_services):
        """Test order rejection due to risk limits."""
        # Configure risk service to reject
        mock_services["risk"].check_trade_risk.return_value = {
            "approved": False,
            "reason": "Position size exceeds limit",
            "risk_score": 0.9,
            "position_size_limit": 1000.0,
            "portfolio_heat": 0.03
        }

        order_data = {
            "symbol": "AAPL",
            "quantity": 100,  # Large quantity
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day"
        }

        response = await app_client.post("/api/orders/market", json=order_data)

        assert response.status_code == 400
        data = response.json()

        assert "Position size exceeds limit" in data["detail"]

        # Verify broker service was not called
        mock_services["broker"].submit_order.assert_not_called()

    async def test_submit_limit_order_success(self, app_client, mock_services):
        """Test successful limit order submission."""
        order_data = {
            "symbol": "TSLA",
            "quantity": 5,
            "side": "sell",
            "order_type": "limit",
            "limit_price": 200.0,
            "time_in_force": "gtc"
        }

        response = await app_client.post("/api/orders/limit", json=order_data)

        assert response.status_code == 201
        data = response.json()

        assert data["order_id"] == "internal_123"
        assert data["symbol"] == "TSLA"

    async def test_get_order_status_success(self, app_client, mock_services):
        """Test successful order status retrieval."""
        order_id = "internal_123"

        response = await app_client.get(f"/api/orders/{order_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["order_id"] == order_id
        assert data["status"] == "FILLED"
        assert data["filled_quantity"] == 10
        assert data["filled_price"] == 150.25

    async def test_get_order_status_not_found(self, app_client, mock_services):
        """Test order status retrieval for non-existent order."""
        mock_services["broker"].get_order_status.side_effect = ValueError("Order not found")

        response = await app_client.get("/api/orders/non_existent")

        assert response.status_code == 404

    async def test_cancel_order_success(self, app_client, mock_services):
        """Test successful order cancellation."""
        order_id = "internal_123"

        response = await app_client.post(f"/api/orders/{order_id}/cancel")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["order_id"] == order_id

        mock_services["broker"].cancel_order.assert_called_once_with(order_id)

    async def test_cancel_order_not_found(self, app_client, mock_services):
        """Test cancelling non-existent order."""
        mock_services["broker"].cancel_order.side_effect = ValueError("Order not found")

        response = await app_client.post("/api/orders/non_existent/cancel")

        assert response.status_code == 404

    async def test_get_account_info(self, app_client, mock_services):
        """Test account information retrieval."""
        mock_services["broker"].get_account_info.return_value = {
            "buying_power": "100000.0",
            "cash": "50000.0",
            "portfolio_value": "120000.0",
            "equity": "120000.0"
        }

        response = await app_client.get("/api/account")

        assert response.status_code == 200
        data = response.json()

        assert data["buying_power"] == "100000.0"
        assert data["equity"] == "120000.0"

    async def test_list_orders(self, app_client, mock_services):
        """Test listing all orders."""
        mock_services["broker"].get_all_orders.return_value = [
            AsyncMock(
                id="order_1",
                symbol="AAPL",
                quantity=10,
                side="buy",
                state="FILLED"
            ),
            AsyncMock(
                id="order_2",
                symbol="TSLA",
                quantity=5,
                side="sell",
                state="SUBMITTED"
            )
        ]

        response = await app_client.get("/api/orders")

        assert response.status_code == 200
        data = response.json()

        assert len(data["orders"]) == 2
        assert data["orders"][0]["order_id"] == "order_1"
        assert data["orders"][1]["symbol"] == "TSLA"

    async def test_invalid_order_data(self, app_client, mock_services):
        """Test order submission with invalid data."""
        invalid_order = {
            "symbol": "",  # Invalid symbol
            "quantity": -1,  # Invalid quantity
            "side": "invalid_side",  # Invalid side
            "order_type": "market"
        }

        response = await app_client.post("/api/orders/market", json=invalid_order)

        assert response.status_code == 422  # Validation error

    async def test_stop_loss_order_submission(self, app_client, mock_services):
        """Test stop-loss order submission."""
        order_data = {
            "symbol": "MSFT",
            "quantity": 8,
            "side": "sell",
            "order_type": "stop",
            "stop_price": 300.0,
            "time_in_force": "day"
        }

        response = await app_client.post("/api/orders/stop", json=order_data)

        assert response.status_code == 201
        data = response.json()

        assert data["symbol"] == "MSFT"
        assert data["order_type"] == "stop"

    async def test_bracket_order_submission(self, app_client, mock_services):
        """Test bracket order (OCO) submission."""
        order_data = {
            "symbol": "GOOGL",
            "quantity": 3,
            "side": "buy",
            "order_type": "market",
            "take_profit": 2800.0,
            "stop_loss": 2500.0,
            "time_in_force": "day"
        }

        response = await app_client.post("/api/orders/bracket", json=order_data)

        assert response.status_code == 201
        data = response.json()

        assert data["symbol"] == "GOOGL"
        assert data["take_profit"] == 2800.0
        assert data["stop_loss"] == 2500.0

    async def test_cors_headers(self, app_client):
        """Test CORS headers are present."""
        response = await app_client.options("/api/orders/market")

        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    async def test_health_check(self, app_client):
        """Test health check endpoint."""
        response = await app_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data

    async def test_api_documentation(self, app_client):
        """Test API documentation is accessible."""
        response = await app_client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
