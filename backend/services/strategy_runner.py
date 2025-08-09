#!/usr/bin/env python3
"""
🎯 STRATEGY RUNNER - BACKTEST/LIVE PARITY
Unified strategy execution engine ensuring identical codepath for backtest and live trading
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from infra.logging import get_structured_logger
from infra.metrics import MetricsTimer, get_trading_metrics
from services.risk_service import RiskService

logger = get_structured_logger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration"""

    BACKTEST = "backtest"
    LIVE = "live"


@dataclass
class SignalEvent:
    """Trading signal event"""

    type: str = "signal"
    timestamp: datetime = None
    symbol: str = ""
    side: str = ""  # buy/sell
    quantity: float = 0.0
    confidence: float = 0.0
    price: float = 0.0
    strategy: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrderEvent:
    """Order execution event"""

    type: str = "order"
    timestamp: datetime = None
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""  # submitted/filled/rejected
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RiskEvent:
    """Risk management event"""

    type: str = "risk"
    timestamp: datetime = None
    symbol: str = ""
    decision: str = ""  # approved/rejected/modified
    original_qty: float = 0.0
    approved_qty: float = 0.0
    reasons: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.reasons is None:
            self.reasons = []
        if self.metadata is None:
            self.metadata = {}


class EventPublisher:
    """Simple in-process pub/sub for strategy events"""

    def __init__(self):
        self._subscribers: list = []

    def subscribe(self, callback):
        """Subscribe to events"""
        self._subscribers.append(callback)

    def publish(self, event):
        """Publish event to all subscribers"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")


class StrategyRunner:
    """
    Unified strategy execution engine with backtest/live parity

    Ensures identical codepath between backtest and live trading:
    - Same signal generation logic
    - Same risk management rules
    - Same order processing flow
    - Identical event emission
    """

    def __init__(self, mode: Literal["backtest", "live"]):
        self.mode = ExecutionMode(mode)
        self.event_publisher = EventPublisher()
        self._running = False
        self._loop_task = None

        # Services (injected based on mode)
        self.risk_service: RiskService | None = None
        self.execution_service = None  # BrokerService or FillSimulatorAdapter

        logger.info(f"🎯 StrategyRunner initialized in {self.mode.value} mode")

    def set_services(self, risk_service: RiskService, execution_service):
        """Inject services (dependency injection pattern)"""
        self.risk_service = risk_service
        self.execution_service = execution_service
        logger.info(f"🔧 Services configured for {self.mode.value} mode")

    def subscribe_to_events(self, callback):
        """Subscribe to strategy events"""
        self.event_publisher.subscribe(callback)

    async def generate_signals(
        self, symbol: str, market_data: dict[str, Any]
    ) -> list[SignalEvent]:
        """
        Generate trading signals from market data

        This method should be identical between backtest and live modes
        """
        signals = []

        try:
            # Extract price data
            current_price = market_data.get("price", 0.0)
            volume = market_data.get("volume", 0)

            if current_price <= 0:
                return signals

            # Simple momentum strategy (placeholder - replace with actual strategy)
            # In production, this would call your ML models/indicators

            # Mock signal generation based on price movement
            # This is where you'd integrate your actual strategy logic
            confidence = 0.6  # Mock confidence

            if self._should_generate_buy_signal(market_data):
                signal = SignalEvent(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    side="buy",
                    quantity=10.0,  # Base quantity - will be risk-adjusted
                    confidence=confidence,
                    price=current_price,
                    strategy="momentum_basic",
                    metadata={
                        "volume": volume,
                        "mode": self.mode.value,
                        "signal_id": str(uuid.uuid4()),
                    },
                )
                signals.append(signal)

                # Publish signal event
                self.event_publisher.publish(signal)

            elif self._should_generate_sell_signal(market_data):
                signal = SignalEvent(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    side="sell",
                    quantity=10.0,
                    confidence=confidence,
                    price=current_price,
                    strategy="momentum_basic",
                    metadata={
                        "volume": volume,
                        "mode": self.mode.value,
                        "signal_id": str(uuid.uuid4()),
                    },
                )
                signals.append(signal)

                # Publish signal event
                self.event_publisher.publish(signal)

            logger.info(f"📊 Generated {len(signals)} signals for {symbol}")

        except Exception as e:
            logger.error(f"❌ Signal generation failed for {symbol}: {e}")

        return signals

    def _should_generate_buy_signal(self, market_data: dict[str, Any]) -> bool:
        """Determine if buy signal should be generated (placeholder logic)"""
        # Replace with actual signal logic
        price = market_data.get("price", 0)
        prev_price = market_data.get("prev_price", price)

        # Simple momentum: buy if price increased by >1%
        return prev_price > 0 and (price - prev_price) / prev_price > 0.01

    def _should_generate_sell_signal(self, market_data: dict[str, Any]) -> bool:
        """Determine if sell signal should be generated (placeholder logic)"""
        # Replace with actual signal logic
        price = market_data.get("price", 0)
        prev_price = market_data.get("prev_price", price)

        # Simple momentum: sell if price decreased by >1%
        return prev_price > 0 and (prev_price - price) / prev_price > 0.01

    async def run_once(
        self, symbol: str, market_data: dict[str, Any]
    ) -> list[OrderEvent]:
        """
        Execute one strategy cycle

        Flow: Signals → Risk → Execution (identical for both modes)
        """
        order_events = []

        try:
            # Step 1: Generate signals
            signals = await self.generate_signals(symbol, market_data)

            if not signals:
                return order_events

            # Step 2: Process each signal through risk management
            for signal in signals:
                try:
                    # Risk check (identical logic for both modes)
                    if not self.risk_service:
                        logger.warning("⚠️ Risk service not available")
                        continue

                    # Get current positions (mock for backtest, real for live)
                    positions = await self._get_current_positions()
                    account = await self._get_account_info()

                    # Measure risk check latency
                    metrics = get_trading_metrics()
                    with MetricsTimer("risk_check", symbol=signal.symbol):
                        # Risk assessment
                        risk_decision = await self.risk_service.check_trade_risk(
                            symbol=signal.symbol,
                            side=signal.side,
                            qty=signal.quantity,
                            price=signal.price,
                            positions=positions,
                            account=account,
                        )

                    # Record risk decision metrics
                    if risk_decision.decision.value == "BLOCKED":
                        primary_reason = (
                            risk_decision.reasons[0]
                            if risk_decision.reasons
                            else "unknown"
                        )
                        severity = (
                            "high" if "limit" in primary_reason.lower() else "medium"
                        )
                        metrics.record_risk_block(
                            primary_reason, signal.symbol, severity
                        )
                    elif risk_decision.decision.value == "MODIFIED":
                        metrics.record_risk_modification(
                            signal.symbol, "quantity_reduced"
                        )

                    # Create risk event
                    risk_event = RiskEvent(
                        timestamp=datetime.utcnow(),
                        symbol=signal.symbol,
                        decision=risk_decision.decision.value,
                        original_qty=signal.quantity,
                        approved_qty=risk_decision.approved_qty,
                        reasons=risk_decision.reasons,
                        metadata={
                            "signal_id": signal.metadata.get("signal_id"),
                            "mode": self.mode.value,
                        },
                    )

                    # Publish risk event
                    self.event_publisher.publish(risk_event)

                    # Step 3: Execute if approved
                    if risk_decision.decision.value in ["APPROVED", "MODIFIED"]:
                        order_event = await self._execute_order(
                            signal, risk_decision.approved_qty
                        )
                        if order_event:
                            order_events.append(order_event)

                except Exception as e:
                    logger.error(f"❌ Signal processing failed: {e}")

        except Exception as e:
            logger.error(f"❌ Strategy run_once failed: {e}")

        return order_events

    async def _execute_order(
        self, signal: SignalEvent, approved_qty: float
    ) -> OrderEvent | None:
        """Execute order through appropriate service (broker or simulator)"""
        metrics = get_trading_metrics()

        # Record order submission
        metrics.record_order_submitted(
            symbol=signal.symbol,
            side=signal.side,
            strategy=signal.metadata.get("strategy_name", "unknown"),
        )

        try:
            if not self.execution_service:
                logger.warning("⚠️ Execution service not available")
                metrics.record_order_rejected(signal.symbol, "no_execution_service")
                return None

            # Measure order execution latency
            with MetricsTimer("order_submit", symbol=signal.symbol, side=signal.side):
                # Execute order (different implementation but same interface)
                if self.mode == ExecutionMode.LIVE:
                    # Live execution through BrokerService
                    order = await self.execution_service.submit_market_order(
                        symbol=signal.symbol, side=signal.side, qty=approved_qty
                    )

                    order_event = OrderEvent(
                        timestamp=datetime.utcnow(),
                        order_id=order.id,
                        symbol=signal.symbol,
                        side=signal.side,
                        quantity=approved_qty,
                        price=signal.price,
                        status="submitted",
                        metadata={
                            "mode": "live",
                            "broker_order_id": order.alpaca_order_id,
                            "signal_id": signal.metadata.get("signal_id"),
                        },
                    )

                else:  # Backtest mode
                    # Simulated execution through FillSimulatorAdapter
                    fill_result = await self.execution_service.simulate_fill(
                        symbol=signal.symbol,
                        side=signal.side,
                        qty=approved_qty,
                        price=signal.price,
                    )

                    order_event = OrderEvent(
                        timestamp=datetime.utcnow(),
                        order_id=f"sim_{uuid.uuid4().hex[:8]}",
                        symbol=signal.symbol,
                        side=signal.side,
                        quantity=approved_qty,
                        price=fill_result.executed_price,
                        status="filled",
                        metadata={
                            "mode": "backtest",
                            "slippage": fill_result.slippage,
                            "commission": fill_result.commission,
                            "signal_id": signal.metadata.get("signal_id"),
                        },
                    )

                    # Record fill immediately for backtest mode
                    metrics.record_order_filled(signal.symbol, signal.side)

            # Publish order event
            self.event_publisher.publish(order_event)

            logger.info(
                f"📈 Order executed: {signal.symbol} {signal.side} {approved_qty} @ {signal.price}"
            )
            return order_event

        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            metrics.record_order_rejected(
                signal.symbol, str(e)[:50]
            )  # Truncate error message
            return None

    async def run_loop(
        self, symbols: list[str], data_source_callback, interval_seconds: int = 60
    ):
        """
        Run continuous strategy loop

        Args:
            symbols: List of symbols to trade
            data_source_callback: Function that returns market data for symbols
            interval_seconds: Loop interval
        """
        self._running = True
        logger.info(
            f"🔄 Starting strategy loop for {symbols} (interval: {interval_seconds}s)"
        )

        try:
            while self._running:
                try:
                    # Get market data for all symbols
                    market_data = await data_source_callback(symbols)

                    # Process each symbol
                    for symbol in symbols:
                        if not self._running:
                            break

                        symbol_data = market_data.get(symbol, {})
                        if symbol_data:
                            await self.run_once(symbol, symbol_data)

                    # Wait for next iteration
                    await asyncio.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"❌ Strategy loop iteration failed: {e}")
                    await asyncio.sleep(5)  # Short pause before retry

        except asyncio.CancelledError:
            logger.info("🛑 Strategy loop cancelled")
        except Exception as e:
            logger.error(f"❌ Strategy loop failed: {e}")
        finally:
            self._running = False
            logger.info("✅ Strategy loop stopped")

    def start_loop(
        self, symbols: list[str], data_source_callback, interval_seconds: int = 60
    ):
        """Start strategy loop as background task"""
        if self._loop_task and not self._loop_task.done():
            logger.warning("⚠️ Strategy loop already running")
            return

        self._loop_task = asyncio.create_task(
            self.run_loop(symbols, data_source_callback, interval_seconds)
        )
        logger.info("🚀 Strategy loop started in background")

    async def stop_loop(self):
        """Stop strategy loop gracefully"""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ Strategy loop stopped")

    async def _get_current_positions(self) -> list[dict[str, Any]]:
        """Get current positions (mode-specific implementation)"""
        if self.mode == ExecutionMode.LIVE:
            # Return real positions in live mode
            # This would connect to your position service
            return []
        else:
            # Return simulated positions in backtest mode
            return []

    async def _get_account_info(self) -> dict[str, Any]:
        """Get account information (mode-specific implementation)"""
        if self.mode == ExecutionMode.LIVE:
            # Return real account info in live mode
            return {
                "buying_power": "10000.00",
                "cash": "5000.00",
                "portfolio_value": "10000.00",
            }
        else:
            # Return simulated account in backtest mode
            return {
                "buying_power": "10000.00",
                "cash": "5000.00",
                "portfolio_value": "10000.00",
            }

    def get_stats(self) -> dict[str, Any]:
        """Get strategy execution statistics"""
        return {
            "mode": self.mode.value,
            "running": self._running,
            "has_risk_service": self.risk_service is not None,
            "has_execution_service": self.execution_service is not None,
            "subscriber_count": len(self.event_publisher._subscribers),
        }


async def create_strategy_runner(mode: Literal["backtest", "live"]) -> StrategyRunner:
    """Factory function to create properly configured StrategyRunner"""
    runner = StrategyRunner(mode)

    # Configure services based on mode
    risk_service = RiskService()
    await risk_service.initialize()

    if mode == "live":
        from services.broker_service import BrokerService

        execution_service = BrokerService()
        await execution_service.initialize()
    else:
        from backend.adapters.fill_simulator import FillSimulatorAdapter

        execution_service = FillSimulatorAdapter()

    runner.set_services(risk_service, execution_service)

    logger.info(f"✅ StrategyRunner created and configured for {mode} mode")
    return runner


# Example usage and testing
if __name__ == "__main__":

    async def test_strategy_runner():
        """Test strategy runner in both modes"""

        # Test data source
        async def mock_data_source(symbols):
            """Mock market data source"""
            return {
                symbol: {
                    "price": 150.0 + (hash(symbol) % 10),
                    "prev_price": 149.0 + (hash(symbol) % 10),
                    "volume": 1000000,
                }
                for symbol in symbols
            }

        # Test event subscriber
        def event_handler(event):
            print(f"📨 Event: {event.type} - {getattr(event, 'symbol', 'N/A')}")

        # Test backtest mode
        print("🧪 Testing Backtest Mode...")
        backtest_runner = await create_strategy_runner("backtest")
        backtest_runner.subscribe_to_events(event_handler)

        # Test single run
        market_data = await mock_data_source(["AAPL"])
        orders = await backtest_runner.run_once("AAPL", market_data["AAPL"])
        print(f"Backtest orders: {len(orders)}")

        # Test live mode (would need real services in production)
        print("\n📈 Testing Live Mode...")
        try:
            live_runner = await create_strategy_runner("live")
            live_runner.subscribe_to_events(event_handler)

            orders = await live_runner.run_once("AAPL", market_data["AAPL"])
            print(f"Live orders: {len(orders)}")
        except Exception as e:
            print(f"Live mode test failed (expected without real broker): {e}")

        print("\n✅ Strategy runner test completed")

    # Run test
    asyncio.run(test_strategy_runner())
