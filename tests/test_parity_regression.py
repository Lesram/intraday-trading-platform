#!/usr/bin/env python3
"""
🔍 PARITY REGRESSION TESTS
Validates that backtest and live modes generate identical order intents and risk decisions
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from backend.services.strategy_runner import (
    StrategyRunner, create_strategy_runner, SignalEvent, OrderEvent, RiskEvent
)


class EventCollector:
    """Collects events for comparison between modes"""
    
    def __init__(self):
        self.signals: List[SignalEvent] = []
        self.orders: List[OrderEvent] = []
        self.risks: List[RiskEvent] = []
    
    def handle_event(self, event):
        """Event handler that sorts events by type"""
        if hasattr(event, 'type'):
            if event.type == 'signal':
                self.signals.append(event)
            elif event.type == 'order':
                self.orders.append(event)
            elif event.type == 'risk':
                self.risks.append(event)

    def reset(self):
        """Reset all collected events"""
        self.signals.clear()
        self.orders.clear()  
        self.risks.clear()

    def get_summary(self) -> Dict[str, int]:
        """Get event count summary"""
        return {
            'signals': len(self.signals),
            'orders': len(self.orders), 
            'risks': len(self.risks)
        }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    base_time = datetime.utcnow()
    return [
        {
            'timestamp': base_time - timedelta(minutes=5),
            'AAPL': {'price': 149.50, 'prev_price': 150.00, 'volume': 1000000},
            'MSFT': {'price': 299.80, 'prev_price': 300.50, 'volume': 800000}
        },
        {
            'timestamp': base_time - timedelta(minutes=4),
            'AAPL': {'price': 150.20, 'prev_price': 149.50, 'volume': 1200000},
            'MSFT': {'price': 301.10, 'prev_price': 299.80, 'volume': 900000}
        },
        {
            'timestamp': base_time - timedelta(minutes=3),
            'AAPL': {'price': 151.00, 'prev_price': 150.20, 'volume': 1100000},
            'MSFT': {'price': 302.50, 'prev_price': 301.10, 'volume': 950000}
        }
    ]


@pytest.fixture 
def mock_risk_service():
    """Mock risk service with deterministic responses"""
    risk_service = AsyncMock()
    
    # Create mock risk decision that's identical for both modes
    from services.risk_service import RiskDecision, RiskDecisionType
    
    async def mock_check_risk(symbol, side, qty, price, positions, account):
        # Deterministic risk decision based on inputs
        decision_key = f"{symbol}_{side}_{qty}_{price}"
        
        # Simple deterministic logic for testing
        if qty > 50:  # Large orders get reduced
            return RiskDecision(
                decision=RiskDecisionType.MODIFIED,
                approved_qty=qty * 0.8,  # Reduce by 20%
                reasons=[f"Position size reduced: {qty} -> {qty * 0.8}"],
                portfolio_heat_before=0.05,
                portfolio_heat_after=0.08
            )
        else:  # Small orders approved
            return RiskDecision(
                decision=RiskDecisionType.APPROVED,
                approved_qty=qty,
                reasons=["All risk checks passed"],
                portfolio_heat_before=0.03,
                portfolio_heat_after=0.05
            )
    
    risk_service.check_trade_risk = mock_check_risk
    return risk_service


@pytest.fixture
def mock_broker_service():
    """Mock broker service for live mode testing"""
    broker_service = AsyncMock()
    
    # Mock order object
    mock_order = MagicMock()
    mock_order.id = "broker_order_123"
    mock_order.alpaca_order_id = "alpaca_123"
    
    broker_service.submit_market_order.return_value = mock_order
    return broker_service


@pytest.fixture  
def mock_fill_simulator():
    """Mock fill simulator for backtest mode testing"""
    from backend.adapters.fill_simulator import SimulatedFill
    
    fill_simulator = AsyncMock()
    
    async def mock_simulate_fill(symbol, side, qty, price, **kwargs):
        return SimulatedFill(
            order_id=f"sim_fill_{symbol}_{side}",
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            executed_price=price + (0.01 if side == 'buy' else -0.01),  # 1 cent slippage
            slippage=0.01,
            commission=1.00,
            total_cost=2.00,
            timestamp=datetime.utcnow(),
            metadata={'simulation': True}
        )
    
    fill_simulator.simulate_fill = mock_simulate_fill
    return fill_simulator


class TestParityRegression:
    """Test parity between backtest and live strategy execution"""

    @pytest.mark.asyncio
    async def test_signal_generation_parity(self, sample_market_data, mock_risk_service, 
                                          mock_broker_service, mock_fill_simulator):
        """
        Test that signal generation is identical between backtest and live modes
        """
        # Create runners for both modes
        backtest_runner = StrategyRunner("backtest")
        live_runner = StrategyRunner("live")
        
        # Set up services (mocked to ensure deterministic behavior)
        backtest_runner.set_services(mock_risk_service, mock_fill_simulator)
        live_runner.set_services(mock_risk_service, mock_broker_service)
        
        # Event collectors
        backtest_collector = EventCollector()
        live_collector = EventCollector()
        
        backtest_runner.subscribe_to_events(backtest_collector.handle_event)
        live_runner.subscribe_to_events(live_collector.handle_event)
        
        # Test signal generation for each data point
        for data_point in sample_market_data:
            for symbol in ['AAPL', 'MSFT']:
                market_data = data_point[symbol]
                
                # Generate signals in both modes
                backtest_signals = await backtest_runner.generate_signals(symbol, market_data)
                live_signals = await live_runner.generate_signals(symbol, market_data)
                
                # Signals should be identical (same logic, same data)
                assert len(backtest_signals) == len(live_signals), f"Signal count mismatch for {symbol}"
                
                for bt_signal, live_signal in zip(backtest_signals, live_signals):
                    assert bt_signal.symbol == live_signal.symbol
                    assert bt_signal.side == live_signal.side
                    assert bt_signal.quantity == live_signal.quantity
                    assert bt_signal.confidence == live_signal.confidence
                    assert bt_signal.strategy == live_signal.strategy
                    # Note: timestamps and IDs will differ, which is expected
        
        print("✅ Signal generation parity verified")

    @pytest.mark.asyncio
    async def test_risk_decision_parity(self, sample_market_data, mock_risk_service,
                                      mock_broker_service, mock_fill_simulator):
        """
        Test that risk decisions are identical between backtest and live modes
        """
        # Create runners
        backtest_runner = StrategyRunner("backtest")
        live_runner = StrategyRunner("live") 
        
        backtest_runner.set_services(mock_risk_service, mock_fill_simulator)
        live_runner.set_services(mock_risk_service, mock_broker_service)
        
        # Event collectors
        backtest_collector = EventCollector()
        live_collector = EventCollector()
        
        backtest_runner.subscribe_to_events(backtest_collector.handle_event)
        live_runner.subscribe_to_events(live_collector.handle_event)
        
        # Run strategy cycles on same data
        test_data = sample_market_data[1]  # Use middle data point
        
        for symbol in ['AAPL', 'MSFT']:
            market_data = test_data[symbol]
            
            # Reset collectors
            backtest_collector.reset()
            live_collector.reset()
            
            # Run full cycle (signals -> risk -> execution)
            await backtest_runner.run_once(symbol, market_data)
            await live_runner.run_once(symbol, market_data)
            
            # Compare risk decisions
            bt_risks = backtest_collector.risks
            live_risks = live_collector.risks
            
            assert len(bt_risks) == len(live_risks), f"Risk decision count mismatch for {symbol}"
            
            for bt_risk, live_risk in zip(bt_risks, live_risks):
                assert bt_risk.symbol == live_risk.symbol
                assert bt_risk.decision == live_risk.decision
                assert bt_risk.original_qty == live_risk.original_qty
                assert abs(bt_risk.approved_qty - live_risk.approved_qty) < 0.001  # Float precision
                assert bt_risk.reasons == live_risk.reasons
        
        print("✅ Risk decision parity verified")

    @pytest.mark.asyncio
    async def test_order_intent_parity(self, sample_market_data, mock_risk_service,
                                     mock_broker_service, mock_fill_simulator):
        """
        Test that order intents (before execution) are identical between modes
        """
        backtest_runner = StrategyRunner("backtest")
        live_runner = StrategyRunner("live")
        
        backtest_runner.set_services(mock_risk_service, mock_fill_simulator) 
        live_runner.set_services(mock_risk_service, mock_broker_service)
        
        backtest_collector = EventCollector()
        live_collector = EventCollector()
        
        backtest_runner.subscribe_to_events(backtest_collector.handle_event)
        live_runner.subscribe_to_events(live_collector.handle_event)
        
        # Test multiple market conditions
        for i, data_point in enumerate(sample_market_data):
            print(f"Testing data point {i+1}/{len(sample_market_data)}")
            
            for symbol in ['AAPL', 'MSFT']:
                market_data = data_point[symbol]
                
                backtest_collector.reset()
                live_collector.reset()
                
                # Execute full strategy cycle
                backtest_orders = await backtest_runner.run_once(symbol, market_data)
                live_orders = await live_runner.run_once(symbol, market_data)
                
                # Compare order intents (symbol, side, quantity should be identical)
                assert len(backtest_orders) == len(live_orders), f"Order count mismatch for {symbol}"
                
                for bt_order, live_order in zip(backtest_orders, live_orders):
                    # Order intents should be identical
                    assert bt_order.symbol == live_order.symbol
                    assert bt_order.side == live_order.side  
                    assert abs(bt_order.quantity - live_order.quantity) < 0.001
                    
                    # Execution details will differ (prices, IDs, etc.) which is expected
                    # Backtest gets filled immediately, live gets submitted
                    
        print("✅ Order intent parity verified")

    @pytest.mark.asyncio
    async def test_end_to_end_parity_regression(self, sample_market_data, mock_risk_service,
                                               mock_broker_service, mock_fill_simulator):
        """
        Full end-to-end parity test over multiple time periods
        """
        print("🔍 Running comprehensive parity regression test...")
        
        # Create strategy runners
        backtest_runner = StrategyRunner("backtest")
        live_runner = StrategyRunner("live")
        
        backtest_runner.set_services(mock_risk_service, mock_fill_simulator)
        live_runner.set_services(mock_risk_service, mock_broker_service)
        
        # Comprehensive event tracking
        backtest_collector = EventCollector()
        live_collector = EventCollector()
        
        backtest_runner.subscribe_to_events(backtest_collector.handle_event)
        live_runner.subscribe_to_events(live_collector.handle_event)
        
        # Test over all time periods and symbols
        symbols = ['AAPL', 'MSFT']
        total_signals_bt = 0
        total_signals_live = 0
        total_orders_bt = 0
        total_orders_live = 0
        
        for i, data_point in enumerate(sample_market_data):
            for symbol in symbols:
                market_data = data_point[symbol]
                
                # Execute strategy
                bt_orders = await backtest_runner.run_once(symbol, market_data)
                live_orders = await live_runner.run_once(symbol, market_data)
                
                total_orders_bt += len(bt_orders)
                total_orders_live += len(live_orders)
        
        # Get final event counts
        bt_summary = backtest_collector.get_summary()
        live_summary = live_collector.get_summary()
        
        print(f"📊 Backtest Summary: {bt_summary}")
        print(f"📊 Live Summary: {live_summary}")
        
        # Assert parity in decision making
        assert bt_summary['signals'] == live_summary['signals'], "Signal generation count mismatch"
        assert bt_summary['risks'] == live_summary['risks'], "Risk decision count mismatch"
        assert total_orders_bt == total_orders_live, "Order intent count mismatch"
        
        # Verify we actually generated some activity
        assert bt_summary['signals'] > 0, "No signals generated in test"
        assert bt_summary['risks'] > 0, "No risk decisions made in test"
        
        print("✅ End-to-end parity regression test PASSED")
        print(f"   Generated {bt_summary['signals']} signals, {bt_summary['risks']} risk decisions")
        print(f"   Executed {total_orders_bt} order intents across {len(sample_market_data)} time periods")

    @pytest.mark.asyncio
    async def test_strategy_statistics_consistency(self, mock_risk_service,
                                                 mock_broker_service, mock_fill_simulator):
        """Test that strategy statistics are consistent between modes"""
        
        backtest_runner = StrategyRunner("backtest")
        live_runner = StrategyRunner("live")
        
        backtest_runner.set_services(mock_risk_service, mock_fill_simulator)
        live_runner.set_services(mock_risk_service, mock_broker_service)
        
        # Get statistics
        bt_stats = backtest_runner.get_stats()
        live_stats = live_runner.get_stats()
        
        # Both should have services configured
        assert bt_stats['has_risk_service'] == live_stats['has_risk_service'] == True
        assert bt_stats['has_execution_service'] == live_stats['has_execution_service'] == True
        
        # Mode should be different
        assert bt_stats['mode'] == 'backtest'
        assert live_stats['mode'] == 'live'
        
        # Both should start as not running
        assert bt_stats['running'] == live_stats['running'] == False
        
        print("✅ Strategy statistics consistency verified")


# Integration test with actual market data simulation
@pytest.mark.asyncio
async def test_toy_strategy_parity():
    """
    Integration test: run a toy strategy over N bars and validate parity
    
    This test simulates a more realistic scenario with a simple momentum strategy
    """
    print("🧸 Running toy strategy parity test...")
    
    # Generate synthetic market data (N bars)
    n_bars = 10
    base_price = 150.0
    market_data_sequence = []
    
    for i in range(n_bars):
        # Simulate price movement
        price_change = (i % 3 - 1) * 0.5  # -0.5, 0, 0.5 pattern
        current_price = base_price + price_change
        prev_price = base_price if i == 0 else market_data_sequence[i-1]['AAPL']['price']
        
        market_data_sequence.append({
            'AAPL': {
                'price': current_price,
                'prev_price': prev_price, 
                'volume': 1000000 + (i * 50000)  # Increasing volume
            }
        })
    
    # Mock services with consistent behavior
    from services.risk_service import RiskDecision, RiskDecisionType
    
    mock_risk = AsyncMock()
    async def deterministic_risk_check(symbol, side, qty, price, positions, account):
        return RiskDecision(
            decision=RiskDecisionType.APPROVED,
            approved_qty=qty,
            reasons=["Toy strategy - approved"],
            portfolio_heat_before=0.01,
            portfolio_heat_after=0.02
        )
    mock_risk.check_trade_risk = deterministic_risk_check
    
    # Mock execution services
    mock_broker = AsyncMock()
    mock_broker.submit_market_order.return_value = MagicMock(
        id="toy_order", alpaca_order_id="toy_alpaca"
    )
    
    from backend.adapters.fill_simulator import FillSimulatorAdapter
    mock_fill_sim = FillSimulatorAdapter()
    
    # Create runners
    backtest_runner = StrategyRunner("backtest") 
    live_runner = StrategyRunner("live")
    
    backtest_runner.set_services(mock_risk, mock_fill_sim)
    live_runner.set_services(mock_risk, mock_broker)
    
    # Track all decisions
    bt_decisions = []
    live_decisions = []
    
    def bt_tracker(event):
        if hasattr(event, 'type'):
            bt_decisions.append((event.type, getattr(event, 'symbol', ''), 
                               getattr(event, 'side', ''), getattr(event, 'quantity', 0)))
    
    def live_tracker(event):
        if hasattr(event, 'type'):
            live_decisions.append((event.type, getattr(event, 'symbol', ''),
                                 getattr(event, 'side', ''), getattr(event, 'quantity', 0)))
    
    backtest_runner.subscribe_to_events(bt_tracker)
    live_runner.subscribe_to_events(live_tracker)
    
    # Run toy strategy over all bars
    for i, market_data in enumerate(market_data_sequence):
        print(f"Processing bar {i+1}/{n_bars}: AAPL @ {market_data['AAPL']['price']}")
        
        await backtest_runner.run_once('AAPL', market_data['AAPL'])
        await live_runner.run_once('AAPL', market_data['AAPL'])
    
    # Validate identical decision sequences
    print(f"Backtest decisions: {len(bt_decisions)}")
    print(f"Live decisions: {len(live_decisions)}")
    
    assert len(bt_decisions) == len(live_decisions), "Decision sequence length mismatch"
    
    # Compare decision intents (type, symbol, side, quantity)
    for i, (bt_decision, live_decision) in enumerate(zip(bt_decisions, live_decisions)):
        assert bt_decision == live_decision, f"Decision {i} mismatch: {bt_decision} != {live_decision}"
    
    print(f"✅ Toy strategy parity validated over {n_bars} bars")
    print(f"   Total decisions: {len(bt_decisions)} (identical between modes)")


# Run all tests
if __name__ == "__main__":
    print("🔍 Running parity regression tests...")
    
    # Run the toy strategy test directly 
    asyncio.run(test_toy_strategy_parity())
    
    print("✅ Parity regression tests completed!")
    print("🎯 Strategy runner demonstrates perfect backtest/live parity")
