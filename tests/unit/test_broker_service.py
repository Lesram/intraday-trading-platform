"""Unit tests for the broker service."""
from unittest.mock import AsyncMock

import pytest

from adapters.errors import AlpacaError
from services.broker_service import BrokerService, OrderState


class TestBrokerService:
    """Test suite for BrokerService."""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client for testing."""
        client = AsyncMock()
        client.get_account.return_value = {
            "buying_power": "100000.0",
            "cash": "50000.0"
        }
        
        # Create a mock order object with an id attribute
        mock_order = AsyncMock()
        mock_order.id = "alpaca_order_123"
        mock_order.status = "new"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side = "buy"
        client.submit_order.return_value = mock_order
        
        # Return order details for status checks
        client.get_order.return_value = {
            "id": "alpaca_order_123", 
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "150.0"
        }
        return client

    @pytest.fixture  
    async def broker_service(self, mock_alpaca_client):
        """Create broker service instance."""
        service = BrokerService()
        # Mock the Alpaca client
        service.alpaca_client = mock_alpaca_client
        return service

    async def test_submit_market_order_success(self, broker_service, mock_alpaca_client):
        """Test successful market order submission."""
        
        internal_order = await broker_service.submit_market_order(
            symbol="AAPL",
            side="buy", 
            qty=10.0
        )
        
        assert internal_order.symbol == "AAPL"
        assert internal_order.qty == 10.0
        assert internal_order.side == "buy"
        assert internal_order.state == OrderState.ACCEPTED  # Order goes through full lifecycle
        assert internal_order.alpaca_order_id == "alpaca_order_123"

        # Check that Alpaca client was called correctly
        mock_alpaca_client.submit_order.assert_called_once()

    async def test_submit_limit_order_success(self, broker_service, mock_alpaca_client):
        """Test successful limit order submission."""
        order_request = {
            "symbol": "TSLA",
            "quantity": 5,
            "side": "sell",
            "order_type": "limit",
            "limit_price": 200.0
        }

        mock_alpaca_client.submit_order.return_value = {
            "id": "alpaca_limit_456",
            "status": "new",
            "symbol": "TSLA",
            "qty": "5",
            "side": "sell"
        }

        internal_order = await broker_service.submit_order(order_request)

        assert internal_order.symbol == "TSLA"
        assert internal_order.limit_price == 200.0
        assert internal_order.alpaca_order_id == "alpaca_limit_456"

    async def test_submit_order_alpaca_error(self, broker_service, mock_alpaca_client):
        """Test order submission with Alpaca API error."""
        order_request = {
            "symbol": "INVALID",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }

        mock_alpaca_client.submit_order.side_effect = AlpacaError("Invalid symbol")
        
        with pytest.raises(AlpacaError):
            await broker_service.submit_order(order_request)

    async def test_get_order_status_success(self, broker_service, mock_alpaca_client):
        """Test successful order status retrieval."""
        # First submit an order
        order_request = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }

        internal_order = await broker_service.submit_order(order_request)
        order_id = internal_order.id

        # Now check its status
        status = await broker_service.get_order_status(order_id)

        assert status.state == OrderState.SUBMITTED
        assert status.alpaca_order_id == "alpaca_order_123"

    async def test_get_order_status_not_found(self, broker_service):
        """Test order status retrieval for non-existent order."""
        with pytest.raises(ValueError, match="Order .* not found"):
            await broker_service.get_order_status("non_existent_id")

    async def test_cancel_order_success(self, broker_service, mock_alpaca_client):
        """Test successful order cancellation."""
        # Submit an order first
        order_request = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 140.0
        }

        internal_order = await broker_service.submit_order(order_request)
        order_id = internal_order.id

        mock_alpaca_client.cancel_order.return_value = True

        # Cancel the order
        result = await broker_service.cancel_order(order_id)

        assert result is True
        mock_alpaca_client.cancel_order.assert_called_once_with("alpaca_order_123")

        # Order state should be updated
        updated_order = await broker_service.get_order_status(order_id)
        assert updated_order.state == OrderState.CANCELLED

    async def test_cancel_order_not_found(self, broker_service):
        """Test cancelling non-existent order."""
        with pytest.raises(ValueError, match="Order .* not found"):
            await broker_service.cancel_order("non_existent_id")

    async def test_update_order_status_from_alpaca(self, broker_service, mock_alpaca_client):
        """Test updating order status from Alpaca."""
        # Submit an order
        order_request = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }

        internal_order = await broker_service.submit_order(order_request)
        order_id = internal_order.id

        # Mock Alpaca returning filled status
        mock_alpaca_client.get_order.return_value = {
            "id": "alpaca_order_123",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "151.50",
            "filled_at": "2024-01-01T10:00:05Z"
        }

        # Update status
        updated_order = await broker_service.update_order_status(order_id)

        assert updated_order.state == OrderState.FILLED
        assert updated_order.filled_quantity == 10
        assert updated_order.filled_price == 151.50
        assert updated_order.filled_at is not None

    async def test_order_state_transitions(self, broker_service):
        """Test order state transitions."""
        # Submit order (NEW -> SUBMITTED)
        order_request = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }

        internal_order = await broker_service.submit_order(order_request)
        assert internal_order.state == OrderState.SUBMITTED
        assert internal_order.submitted_at is not None

        # Check that order is tracked
        assert internal_order.id in broker_service.orders

    async def test_stop_loss_order_creation(self, broker_service, mock_alpaca_client):
        """Test stop-loss order creation."""
        order_request = {
            "symbol": "MSFT",
            "quantity": 8,
            "side": "sell",
            "order_type": "stop",
            "stop_price": 300.0
        }

        mock_alpaca_client.submit_order.return_value = {
            "id": "alpaca_stop_789",
            "status": "new",
            "symbol": "MSFT",
            "qty": "8",
            "side": "sell"
        }

        internal_order = await broker_service.submit_order(order_request)

        assert internal_order.order_type == "stop"
        assert internal_order.stop_price == 300.0
        assert internal_order.alpaca_order_id == "alpaca_stop_789"

    async def test_oco_bracket_order_creation(self, broker_service, mock_alpaca_client):
        """Test OCO bracket order creation."""
        order_request = {
            "symbol": "GOOGL",
            "quantity": 3,
            "side": "buy",
            "order_type": "market",
            "take_profit": 2800.0,
            "stop_loss": 2500.0
        }

        # Mock responses for main order and OCO orders
        mock_alpaca_client.submit_order.side_effect = [
            {"id": "main_order", "status": "new"},
            {"id": "tp_order", "status": "new"},
            {"id": "sl_order", "status": "new"}
        ]

        internal_order = await broker_service.submit_order(order_request)

        assert internal_order.take_profit_price == 2800.0
        assert internal_order.stop_loss_price == 2500.0
        assert mock_alpaca_client.submit_order.call_count == 3  # Main + TP + SL

    async def test_get_account_info(self, broker_service, mock_alpaca_client):
        """Test account information retrieval."""
        account_info = await broker_service.get_account_info()

        assert account_info["buying_power"] == "100000.0"
        assert account_info["cash"] == "50000.0"
        mock_alpaca_client.get_account.assert_called_once()

    def test_order_audit_logging(self, broker_service):
        """Test that order actions are properly logged for audit."""
        # This would test logging functionality
        # Implementation depends on your logging setup
        pass

    def test_order_id_generation(self, broker_service):
        """Test unique order ID generation."""
        # Test that internal order IDs are unique
        order_id1 = broker_service._generate_order_id()
        order_id2 = broker_service._generate_order_id()

        assert order_id1 != order_id2
        assert len(order_id1) > 0
        assert len(order_id2) > 0
