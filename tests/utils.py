"""Test utilities and helpers for the trading platform test suite."""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import AsyncClient


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_order_request(
        symbol: str = "AAPL",
        quantity: float = 10.0,
        side: str = "buy",
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """Create a standard order request."""
        request = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force
        }
        
        if limit_price:
            request["limit_price"] = limit_price
        if stop_price:
            request["stop_price"] = stop_price
            
        return request
    
    @staticmethod
    def create_account_data(
        buying_power: float = 100000.0,
        cash: float = 50000.0,
        equity: float = 100000.0,
        portfolio_value: float = 100000.0
    ) -> Dict[str, Any]:
        """Create mock account data."""
        return {
            "account_number": "123456789",
            "status": "ACTIVE",
            "buying_power": str(buying_power),
            "cash": str(cash),
            "equity": str(equity),
            "portfolio_value": str(portfolio_value),
            "daytrade_count": 0,
            "pattern_day_trader": False
        }
    
    @staticmethod
    def create_alpaca_order_response(
        order_id: str = None,
        symbol: str = "AAPL",
        qty: float = 10.0,
        side: str = "buy",
        status: str = "new"
    ) -> Mock:
        """Create mock Alpaca order response."""
        mock_order = Mock()
        mock_order.id = order_id or str(uuid.uuid4())
        mock_order.symbol = symbol
        mock_order.qty = str(qty)
        mock_order.side = side
        mock_order.status = status
        mock_order.order_type = "market"
        mock_order.submitted_at = datetime.utcnow()
        return mock_order


class MockAlpacaClient:
    """Enhanced mock Alpaca client for testing."""
    
    def __init__(self):
        self.orders = {}
        self.account_data = TestDataGenerator.create_account_data()
        self.should_fail = False
        self.failure_message = "Mock failure"
        self.call_count = 0
    
    def set_failure_mode(self, should_fail: bool, message: str = "Mock failure"):
        """Configure the mock to simulate failures."""
        self.should_fail = should_fail
        self.failure_message = message
    
    async def get_account(self):
        """Mock get account."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.failure_message)
        return self.account_data
    
    async def submit_order(self, **kwargs):
        """Mock order submission."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.failure_message)
        
        order = TestDataGenerator.create_alpaca_order_response(
            symbol=kwargs.get("symbol", "AAPL"),
            qty=kwargs.get("qty", 10.0),
            side=kwargs.get("side", "buy")
        )
        self.orders[order.id] = order
        return order
    
    async def get_order(self, order_id: str):
        """Mock get order status."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.failure_message)
        
        if order_id not in self.orders:
            raise Exception("Order not found")
        
        return self.orders[order_id]
    
    async def cancel_order(self, order_id: str):
        """Mock order cancellation."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.failure_message)
        
        if order_id in self.orders:
            self.orders[order_id].status = "canceled"
        return True


class WebSocketTestClient:
    """WebSocket test client for testing real-time features."""
    
    def __init__(self):
        self.connected = False
        self.messages = []
        self.subscriptions = set()
    
    async def connect(self, url: str):
        """Simulate WebSocket connection."""
        self.connected = True
        return True
    
    async def disconnect(self):
        """Simulate WebSocket disconnection."""
        self.connected = False
        self.messages.clear()
        self.subscriptions.clear()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if not self.connected:
            raise Exception("Not connected")
        
        # Handle subscription messages
        if message.get("type") == "subscribe":
            self.subscriptions.add(message.get("symbol"))
        elif message.get("type") == "unsubscribe":
            self.subscriptions.discard(message.get("symbol"))
    
    async def receive_message(self, timeout: float = 1.0) -> Dict[str, Any]:
        """Receive message from WebSocket."""
        if not self.connected:
            raise Exception("Not connected")
        
        # Simulate receiving a price update
        await asyncio.sleep(0.1)
        return {
            "type": "price_update",
            "symbol": "AAPL",
            "price": 150.25,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def simulate_price_update(self, symbol: str, price: float):
        """Simulate receiving a price update."""
        if symbol in self.subscriptions:
            message = {
                "type": "price_update",
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.messages.append(message)


class ConcurrencyTestHelper:
    """Helper for testing concurrent operations."""
    
    @staticmethod
    async def run_concurrent_operations(operations: List, max_concurrent: int = 10):
        """Run operations concurrently with a limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_operation(op):
            async with semaphore:
                return await op
        
        tasks = [limited_operation(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @staticmethod
    def create_concurrent_order_requests(count: int, base_symbol: str = "AAPL") -> List[Dict]:
        """Create multiple order requests for concurrency testing."""
        orders = []
        for i in range(count):
            orders.append(TestDataGenerator.create_order_request(
                symbol=f"{base_symbol}",
                quantity=10.0 + i,
                side="buy" if i % 2 == 0 else "sell"
            ))
        return orders


class RiskTestScenarios:
    """Pre-defined risk scenarios for testing."""
    
    @staticmethod
    def get_high_risk_scenario():
        """Scenario that should be blocked by risk management."""
        return {
            "symbol": "TSLA",
            "quantity": 1000.0,  # Very large quantity
            "side": "buy",
            "order_type": "market",
            "current_portfolio_heat": 0.95  # High portfolio heat
        }
    
    @staticmethod
    def get_low_risk_scenario():
        """Scenario that should pass risk management."""
        return {
            "symbol": "AAPL",
            "quantity": 5.0,  # Small quantity
            "side": "buy",
            "order_type": "limit",
            "limit_price": 150.0,
            "current_portfolio_heat": 0.01  # Low portfolio heat
        }
    
    @staticmethod
    def get_edge_case_scenarios():
        """Various edge case scenarios for thorough testing."""
        return [
            {
                "name": "zero_quantity",
                "data": TestDataGenerator.create_order_request(quantity=0.0)
            },
            {
                "name": "negative_quantity",
                "data": TestDataGenerator.create_order_request(quantity=-10.0)
            },
            {
                "name": "invalid_symbol",
                "data": TestDataGenerator.create_order_request(symbol="")
            },
            {
                "name": "invalid_side",
                "data": TestDataGenerator.create_order_request(side="invalid")
            },
            {
                "name": "limit_without_price",
                "data": {**TestDataGenerator.create_order_request(order_type="limit"), "limit_price": None}
            }
        ]


class PerformanceTestHelper:
    """Helper for performance testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start_timer(self):
        """Start performance timing."""
        self.start_time = datetime.utcnow()
    
    def end_timer(self) -> float:
        """End performance timing and return duration in seconds."""
        self.end_time = datetime.utcnow()
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            return duration
        return 0.0
    
    def record_metric(self, name: str, value: Any):
        """Record a performance metric."""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance test summary."""
        duration = self.end_timer()
        return {
            "duration_seconds": duration,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


# Pytest fixtures that can be used across test modules
@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def mock_alpaca_client():
    """Provide enhanced mock Alpaca client."""
    return MockAlpacaClient()


@pytest.fixture
def websocket_test_client():
    """Provide WebSocket test client."""
    return WebSocketTestClient()


@pytest.fixture
def concurrency_helper():
    """Provide concurrency test helper."""
    return ConcurrencyTestHelper()


@pytest.fixture
def risk_scenarios():
    """Provide risk test scenarios."""
    return RiskTestScenarios()


@pytest.fixture
def performance_helper():
    """Provide performance test helper."""
    return PerformanceTestHelper()


# Common assertions
def assert_order_response_valid(response_data: Dict[str, Any]):
    """Assert that an order response has valid structure."""
    required_fields = ["order_id", "symbol", "quantity", "side", "status"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert response_data["quantity"] > 0, "Quantity must be positive"
    assert response_data["side"] in ["buy", "sell"], "Side must be buy or sell"


def assert_error_response_valid(response_data: Dict[str, Any]):
    """Assert that an error response has valid structure."""
    assert "detail" in response_data, "Error response must have detail field"
    assert isinstance(response_data["detail"], str), "Detail must be a string"
    assert len(response_data["detail"]) > 0, "Detail must not be empty"


def assert_performance_acceptable(duration: float, max_duration: float = 1.0):
    """Assert that operation completed within acceptable time."""
    assert duration <= max_duration, f"Operation took {duration:.3f}s, expected <= {max_duration}s"
