#!/usr/bin/env python3
"""
🧪 METRICS INTEGRATION TESTS
Comprehensive tests for trading metrics observability
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import requests
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from infra.metrics import TradingMetrics, get_trading_metrics, MetricsTimer, measure_latency
from api.metrics_api import app
from services.strategy_runner import StrategyRunner, ExecutionMode, SignalEvent


class TestTradingMetrics:
    """Test core metrics functionality"""
    
    def setup_method(self):
        """Setup fresh metrics for each test"""
        self.metrics = TradingMetrics()
    
    def test_order_flow_metrics(self):
        """Test order submission, fill, and rejection tracking"""
        # Test order submissions
        self.metrics.record_order_submitted("AAPL", "buy", "momentum")
        self.metrics.record_order_submitted("TSLA", "sell", "mean_reversion")
        
        # Test order fills
        self.metrics.record_order_filled("AAPL", "buy")
        self.metrics.record_order_filled("MSFT", "sell")
        
        # Test order rejections
        self.metrics.record_order_rejected("GOOGL", "insufficient_buying_power")
        self.metrics.record_order_rejected("NVDA", "position_limit_exceeded")
        
        # Get summary
        summary = self.metrics.get_metrics_summary()
        
        assert summary['orders_submitted'] >= 2
        assert summary['orders_filled'] >= 2
        
        # Verify prometheus format contains our metrics
        prometheus_output = self.metrics.get_metrics()
        assert "orders_submitted_total" in prometheus_output
        assert "orders_filled_total" in prometheus_output
        assert "orders_rejected_total" in prometheus_output
        assert 'symbol="AAPL"' in prometheus_output
        assert 'side="buy"' in prometheus_output
        assert 'strategy="momentum"' in prometheus_output
    
    def test_risk_management_metrics(self):
        """Test risk block and modification tracking"""
        # Test risk blocks
        self.metrics.record_risk_block("position_limit_exceeded", "AAPL", "high")
        self.metrics.record_risk_block("daily_loss_limit", "TSLA", "critical")
        self.metrics.record_risk_block("concentration_limit", "MSFT", "medium")
        
        # Test risk modifications
        self.metrics.record_risk_modification("GOOGL", "quantity_reduced")
        self.metrics.record_risk_modification("NVDA", "price_adjusted")
        
        # Verify metrics
        prometheus_output = self.metrics.get_metrics()
        assert "risk_blocks_total" in prometheus_output
        assert "risk_modifications_total" in prometheus_output
        assert 'reason="position_limit_exceeded"' in prometheus_output
        assert 'severity="high"' in prometheus_output
        assert 'modification_type="quantity_reduced"' in prometheus_output
    
    def test_latency_metrics(self):
        """Test latency histogram tracking"""
        # Test order submission latency
        self.metrics.record_order_submit_latency("AAPL", "buy", 0.025)
        self.metrics.record_order_submit_latency("TSLA", "sell", 0.050)
        
        # Test risk check latency
        self.metrics.record_risk_check_latency("MSFT", 0.010)
        self.metrics.record_risk_check_latency("GOOGL", 0.015)
        
        # Test broker API latency
        self.metrics.record_broker_api_latency("/v2/orders", "POST", 0.150)
        self.metrics.record_broker_api_latency("/v2/positions", "GET", 0.080)
        
        # Verify histograms in prometheus output
        prometheus_output = self.metrics.get_metrics()
        assert "order_submit_latency_seconds" in prometheus_output
        assert "risk_check_latency_seconds" in prometheus_output
        assert "broker_api_latency_seconds" in prometheus_output
        
        # Should have bucket counts
        assert "_bucket{" in prometheus_output
        assert "_count " in prometheus_output
        assert "_sum " in prometheus_output
    
    def test_portfolio_metrics(self):
        """Test portfolio and position tracking"""
        # Update portfolio metrics
        portfolio_data = {
            'portfolio_value': 150000.0,
            'buying_power': 75000.0,
            'position_count': 8,
            'portfolio_heat': 0.25,
            'daily_pnl': -1500.0
        }
        self.metrics.update_portfolio_metrics(portfolio_data)
        
        # Update individual positions
        self.metrics.update_position_value("AAPL", 25000.0)
        self.metrics.update_position_value("TSLA", -5000.0)
        
        # Verify gauges
        prometheus_output = self.metrics.get_metrics()
        assert "portfolio_value_usd" in prometheus_output
        assert "buying_power_usd" in prometheus_output
        assert "position_count" in prometheus_output
        assert "portfolio_heat" in prometheus_output
        assert "daily_pnl_usd" in prometheus_output
        assert "position_value_usd" in prometheus_output
        
        # Verify values
        assert "150000" in prometheus_output
        assert "75000" in prometheus_output
        assert '8 ' in prometheus_output  # position count
        assert "0.25" in prometheus_output  # portfolio heat
    
    def test_system_health_metrics(self):
        """Test system health indicators"""
        # Set broker connection status
        self.metrics.set_broker_connection_status(True)
        self.metrics.set_broker_connection_status(False)
        
        # Set risk service status  
        self.metrics.set_risk_service_status(True)
        
        # Update market data lag
        self.metrics.update_market_data_lag("alpaca", 0.050)
        self.metrics.update_market_data_lag("polygon", 0.025)
        
        # Verify system health metrics
        prometheus_output = self.metrics.get_metrics()
        assert "broker_connection_status" in prometheus_output
        assert "risk_service_status" in prometheus_output
        assert "market_data_lag_seconds" in prometheus_output
        assert 'source="alpaca"' in prometheus_output
        assert 'source="polygon"' in prometheus_output
        
        # Current values should be reflected
        summary = self.metrics.get_metrics_summary()
        assert summary['broker_connected'] == False  # Last value set
    
    def test_metrics_timer_context_manager(self):
        """Test MetricsTimer context manager"""
        # Test order submit timer
        with MetricsTimer('order_submit', symbol='AAPL', side='buy'):
            time.sleep(0.01)  # Simulate 10ms operation
        
        # Test broker API timer
        with MetricsTimer('broker_api', endpoint='/v2/orders', method='POST'):
            time.sleep(0.005)  # Simulate 5ms operation
        
        # Verify latency was recorded
        prometheus_output = self.metrics.get_metrics()
        assert "order_submit_latency_seconds" in prometheus_output
        assert "broker_api_latency_seconds" in prometheus_output
        
        # Should have some measurement > 0
        lines = prometheus_output.split('\n')
        latency_lines = [line for line in lines if 'latency_seconds_count' in line]
        assert len(latency_lines) > 0
    
    @pytest.mark.asyncio
    async def test_measure_latency_decorator(self):
        """Test automatic latency measurement decorator"""
        
        @measure_latency('order_submit_latency_seconds', symbol='AAPL', side='buy')
        async def mock_submit_order():
            await asyncio.sleep(0.01)  # Simulate async operation
            return "order_123"
        
        # Execute decorated function
        result = await mock_submit_order()
        assert result == "order_123"
        
        # Verify latency was recorded
        prometheus_output = self.metrics.get_metrics()
        assert "order_submit_latency_seconds" in prometheus_output


class TestMetricsAPI:
    """Test FastAPI metrics endpoint"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "trading-metrics"
        assert "timestamp" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        # Generate some test metrics first
        self.client.get("/metrics/test")
        
        # Get metrics
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        assert "orders_submitted_total" in content
        assert "portfolio_value_usd" in content
        assert "risk_blocks_total" in content
        
        # Should be valid Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
    
    def test_metrics_summary_endpoint(self):
        """Test human-readable metrics summary"""
        response = self.client.get("/metrics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "metrics_summary" in data
        assert "endpoint_info" in data
        assert data["endpoint_info"]["prometheus_endpoint"] == "/metrics"
    
    def test_test_metrics_endpoint(self):
        """Test metrics generation endpoint"""
        response = self.client.get("/metrics/test")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "sample_metrics" in data
        assert "next_steps" in data
        
        # Should have generated some metrics
        metrics_summary = data["sample_metrics"]
        assert metrics_summary["orders_submitted"] > 0
    
    def test_root_endpoint(self):
        """Test root API information endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Trading Metrics API"
        assert "endpoints" in data
        assert "integration" in data
        assert "/metrics" in data["endpoints"]
        assert "prometheus_config" in data["integration"]


class TestMetricsIntegration:
    """Test metrics integration with strategy runner"""
    
    @pytest.mark.asyncio
    async def test_strategy_runner_metrics_integration(self):
        """Test that strategy runner properly records metrics"""
        # Create test strategy runner
        runner = StrategyRunner(ExecutionMode.BACKTEST)
        
        # Mock services
        mock_risk_service = AsyncMock()
        mock_risk_service.check_trade_risk.return_value = MagicMock(
            decision=MagicMock(value="APPROVED"),
            approved_qty=100,
            reasons=[]
        )
        
        mock_execution_service = AsyncMock()
        mock_execution_service.simulate_fill.return_value = MagicMock(
            executed_price=150.0,
            slippage=0.01,
            commission=1.0
        )
        
        runner.set_services(mock_risk_service, mock_execution_service)
        
        # Create test signal
        test_signal = SignalEvent(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            confidence=0.8,
            metadata={"strategy_name": "test_strategy"}
        )
        
        # Get metrics instance
        metrics = get_trading_metrics()
        initial_orders = metrics.get_metrics_summary()['orders_submitted']
        
        # Execute order through strategy runner
        with patch.object(runner, '_get_current_positions', return_value=[]):
            with patch.object(runner, '_get_account_info', return_value={'buying_power': 10000}):
                order_events = await runner.run_once("AAPL", {"price": 150.0})
        
        # Verify metrics were recorded
        final_orders = metrics.get_metrics_summary()['orders_submitted']
        assert final_orders > initial_orders
        
        # Check prometheus output includes our test
        prometheus_output = metrics.get_metrics()
        assert 'symbol="AAPL"' in prometheus_output
        assert 'strategy="test_strategy"' in prometheus_output
    
    def test_metrics_persistence(self):
        """Test that metrics persist across multiple operations"""
        metrics = get_trading_metrics()
        
        # Record multiple operations
        for i in range(5):
            metrics.record_order_submitted("AAPL", "buy", "test")
            metrics.record_order_filled("AAPL", "buy")
        
        # Verify cumulative counts
        summary = metrics.get_metrics_summary()
        assert summary['orders_submitted'] >= 5
        assert summary['orders_filled'] >= 5
        
        # Prometheus output should show cumulative counts
        prometheus_output = metrics.get_metrics()
        assert "orders_submitted_total" in prometheus_output
        
        # Extract counter value for AAPL buy orders
        lines = prometheus_output.split('\n')
        aapl_buy_lines = [line for line in lines if 'orders_submitted_total{' in line and 'symbol="AAPL"' in line and 'side="buy"' in line]
        assert len(aapl_buy_lines) > 0
        
        # Should show count >= 5
        for line in aapl_buy_lines:
            if 'strategy="test"' in line:
                value = float(line.split()[-1])
                assert value >= 5.0


# Integration test with actual metrics server
@pytest.mark.integration
class TestMetricsServer:
    """Integration tests for metrics server"""
    
    def test_server_startup_and_endpoints(self):
        """Test that metrics server starts and serves endpoints correctly"""
        # This would typically be run as a separate process
        # For now, just test the client directly
        with TestClient(app) as client:
            # Test all endpoints
            health_response = client.get("/health")
            assert health_response.status_code == 200
            
            metrics_response = client.get("/metrics")
            assert metrics_response.status_code == 200
            assert "text/plain" in metrics_response.headers["content-type"]
            
            summary_response = client.get("/metrics/summary")
            assert summary_response.status_code == 200
            
            test_response = client.get("/metrics/test")
            assert test_response.status_code == 200
            
            root_response = client.get("/")
            assert root_response.status_code == 200


def run_metrics_integration_tests():
    """Run all metrics integration tests"""
    print("🧪 Running metrics integration tests...")
    
    # Test basic metrics
    test_metrics = TestTradingMetrics()
    test_metrics.setup_method()
    test_metrics.test_order_flow_metrics()
    test_metrics.test_risk_management_metrics()
    test_metrics.test_latency_metrics()
    test_metrics.test_portfolio_metrics()
    test_metrics.test_system_health_metrics()
    print("✅ Core metrics tests passed")
    
    # Test API
    test_api = TestMetricsAPI()
    test_api.setup_method()
    test_api.test_health_endpoint()
    test_api.test_metrics_endpoint()
    test_api.test_metrics_summary_endpoint()
    test_api.test_test_metrics_endpoint()
    test_api.test_root_endpoint()
    print("✅ API tests passed")
    
    print("🎉 All metrics integration tests completed successfully!")
    
    # Show sample metrics output
    metrics = get_trading_metrics()
    metrics.record_order_submitted("TEST", "buy", "integration_test")
    metrics.record_order_filled("TEST", "buy")
    print(f"\n📊 Sample metrics output:\n{metrics.get_metrics()[:500]}...")


if __name__ == "__main__":
    run_metrics_integration_tests()
