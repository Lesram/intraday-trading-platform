#!/usr/bin/env python3
"""
🎮 FILL SIMULATOR ADAPTER
Wraps existing FillSimulator to provide broker-compatible interface
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from infra.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class SimulatedFill:
    """Simulated fill result matching broker fill format"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    executed_price: float
    slippage: float
    commission: float
    total_cost: float
    timestamp: datetime
    metadata: Dict[str, Any]


class SlippageModel:
    """Realistic slippage modeling"""
    
    def __init__(self):
        # Slippage parameters by symbol type
        self.base_slippage = {
            'large_cap': 0.0001,    # 1 bps
            'mid_cap': 0.0002,      # 2 bps  
            'small_cap': 0.0005,    # 5 bps
            'default': 0.0002       # 2 bps
        }
        
        # Volume impact parameters
        self.volume_impact_factor = 0.01  # 1% of daily volume = 1bp extra slippage

    def estimate_slippage(self, symbol: str, quantity: float, price: float, 
                         daily_volume: float = 1000000) -> float:
        """Estimate realistic slippage based on trade characteristics"""
        try:
            # Base slippage by market cap (simplified categorization)
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
                base_slippage = self.base_slippage['large_cap']
            elif price > 50:
                base_slippage = self.base_slippage['mid_cap']
            else:
                base_slippage = self.base_slippage['small_cap']
            
            # Volume impact
            trade_notional = quantity * price
            volume_participation = trade_notional / (daily_volume * price)
            volume_impact = volume_participation * self.volume_impact_factor
            
            # Total slippage
            total_slippage_pct = base_slippage + volume_impact
            slippage_dollars = price * total_slippage_pct
            
            return slippage_dollars
            
        except Exception as e:
            logger.warning(f"Slippage calculation failed: {e}")
            return price * 0.0002  # Default 2bps


class CommissionModel:
    """Realistic commission modeling"""
    
    def __init__(self):
        # Commission structure (similar to major brokers)
        self.per_share_commission = 0.0  # Zero commission (like most modern brokers)
        self.min_commission = 0.0
        self.max_commission = 0.0
        
        # SEC fees and other regulatory costs
        self.sec_fee_rate = 0.0000231  # $0.0000231 per dollar sold
        self.taf_fee = 0.000145  # TAF fee for NASDAQ
        
    def calculate_commission(self, symbol: str, quantity: float, price: float, 
                           side: str) -> float:
        """Calculate realistic commission and fees"""
        try:
            notional = quantity * price
            
            # Base commission (most retail brokers are zero)
            commission = max(
                self.min_commission,
                min(self.max_commission, quantity * self.per_share_commission)
            )
            
            # Regulatory fees (only on sells for SEC fees)
            if side.lower() == 'sell':
                sec_fee = notional * self.sec_fee_rate
                commission += sec_fee
            
            # TAF fees for NASDAQ stocks (simplified)
            if symbol not in ['SPY', 'QQQ']:  # Assume non-ETFs pay TAF
                taf_fee = min(quantity * self.taf_fee, 5.95)  # Capped at $5.95
                commission += taf_fee
            
            return round(commission, 2)
            
        except Exception as e:
            logger.warning(f"Commission calculation failed: {e}")
            return 1.0  # Default $1 commission


class FillSimulatorAdapter:
    """
    Adapter that wraps existing FillSimulator to provide broker-compatible interface
    
    This adapter ensures backtest fills have the same shape and behavior as
    real broker fills, enabling perfect parity between backtest and live trading.
    """
    
    def __init__(self):
        self.slippage_model = SlippageModel()
        self.commission_model = CommissionModel()
        
        # Try to import and use existing FillSimulator
        self.execution_simulator = None
        try:
            from institutional_backtest_engine import ExecutionSimulator
            self.execution_simulator = ExecutionSimulator()
            logger.info("✅ Using existing ExecutionSimulator")
        except ImportError:
            logger.warning("⚠️ ExecutionSimulator not available, using built-in simulation")
        
        logger.info("🎮 FillSimulatorAdapter initialized")

    async def simulate_fill(self, symbol: str, side: str, qty: float, 
                           price: float, **kwargs) -> SimulatedFill:
        """
        Simulate order fill with realistic execution characteristics
        
        Returns same format as broker fills for perfect parity
        """
        try:
            order_id = f"sim_{uuid.uuid4().hex[:8]}"
            
            # Get market parameters
            volume = kwargs.get('volume', 1000000)
            spread_pct = kwargs.get('spread_pct', 0.001)
            
            # Use existing ExecutionSimulator if available
            if self.execution_simulator:
                trade_result = self.execution_simulator.simulate_trade_execution(
                    signal=side.upper(),
                    symbol=symbol,
                    quantity=qty,
                    price=price,
                    volume=volume,
                    spread_pct=spread_pct
                )
                
                # Convert to our format
                fill = SimulatedFill(
                    order_id=order_id,
                    symbol=symbol,
                    side=side.lower(),
                    quantity=qty,
                    price=price,
                    executed_price=trade_result.executed_price,
                    slippage=trade_result.slippage,
                    commission=trade_result.commission,
                    total_cost=trade_result.total_cost,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'simulation_method': 'execution_simulator',
                        'signal_confidence': getattr(trade_result, 'signal_confidence', 0.0),
                        'risk_metrics': getattr(trade_result, 'risk_metrics', {})
                    }
                )
            else:
                # Built-in simulation
                fill = await self._simulate_fill_builtin(
                    order_id, symbol, side, qty, price, volume, spread_pct
                )
            
            logger.info(f"🎮 Simulated fill: {symbol} {side} {qty} @ {fill.executed_price:.4f}")
            return fill
            
        except Exception as e:
            logger.error(f"❌ Fill simulation failed: {e}")
            # Return basic fill to prevent failures
            return SimulatedFill(
                order_id=f"sim_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side.lower(),
                quantity=qty,
                price=price,
                executed_price=price,
                slippage=0.0,
                commission=1.0,
                total_cost=1.0,
                timestamp=datetime.utcnow(),
                metadata={'error': str(e)}
            )

    async def _simulate_fill_builtin(self, order_id: str, symbol: str, side: str,
                                   qty: float, price: float, volume: float,
                                   spread_pct: float) -> SimulatedFill:
        """Built-in fill simulation when ExecutionSimulator is not available"""
        
        # Calculate slippage
        slippage = self.slippage_model.estimate_slippage(symbol, qty, price, volume)
        
        # Determine execution price  
        if side.lower() == 'buy':
            executed_price = price + slippage
        else:
            executed_price = max(0.01, price - slippage)  # Prevent negative prices
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(symbol, qty, price, side)
        
        # Calculate total cost
        notional = qty * executed_price
        spread_cost = notional * spread_pct
        total_cost = commission + spread_cost + abs(slippage * qty)
        
        return SimulatedFill(
            order_id=order_id,
            symbol=symbol,
            side=side.lower(),
            quantity=qty,
            price=price,
            executed_price=executed_price,
            slippage=slippage,
            commission=commission,
            total_cost=total_cost,
            timestamp=datetime.utcnow(),
            metadata={
                'simulation_method': 'builtin',
                'spread_cost': spread_cost,
                'notional': notional
            }
        )

    async def simulate_partial_fill(self, symbol: str, side: str, requested_qty: float,
                                  price: float, fill_ratio: float = 0.8, **kwargs) -> SimulatedFill:
        """Simulate partial fill scenario"""
        filled_qty = requested_qty * fill_ratio
        return await self.simulate_fill(symbol, side, filled_qty, price, **kwargs)

    async def simulate_fill_with_delay(self, symbol: str, side: str, qty: float,
                                     price: float, delay_seconds: float = 0.1, **kwargs) -> SimulatedFill:
        """Simulate fill with execution delay"""
        import asyncio
        await asyncio.sleep(delay_seconds)
        return await self.simulate_fill(symbol, side, qty, price, **kwargs)

    def get_fill_statistics(self) -> Dict[str, Any]:
        """Get fill simulation statistics"""
        return {
            'adapter_type': 'fill_simulator',
            'has_execution_simulator': self.execution_simulator is not None,
            'slippage_model': {
                'base_slippage_bps': {k: v * 10000 for k, v in self.slippage_model.base_slippage.items()},
                'volume_impact_factor': self.slippage_model.volume_impact_factor
            },
            'commission_model': {
                'per_share': self.commission_model.per_share_commission,
                'sec_fee_rate': self.commission_model.sec_fee_rate,
                'taf_fee': self.commission_model.taf_fee
            }
        }

    async def get_market_impact_estimate(self, symbol: str, qty: float, price: float,
                                       daily_volume: float = 1000000) -> Dict[str, float]:
        """Get market impact estimates for order sizing"""
        slippage = self.slippage_model.estimate_slippage(symbol, qty, price, daily_volume)
        commission = self.commission_model.calculate_commission(symbol, qty, price, 'buy')
        
        notional = qty * price
        total_cost_pct = (slippage + commission) / notional * 100
        
        return {
            'slippage_dollars': slippage,
            'slippage_bps': (slippage / price) * 10000,
            'commission': commission,
            'total_cost_pct': total_cost_pct,
            'price_impact_pct': (slippage / price) * 100,
            'volume_participation_pct': (notional / (daily_volume * price)) * 100
        }


# Testing and example usage
async def test_fill_simulator_adapter():
    """Test the fill simulator adapter"""
    print("🧪 Testing FillSimulatorAdapter...")
    
    adapter = FillSimulatorAdapter()
    
    # Test basic fill
    fill = await adapter.simulate_fill(
        symbol="AAPL",
        side="buy", 
        qty=100.0,
        price=150.0,
        volume=5000000
    )
    
    print(f"✅ Basic fill: {fill.symbol} {fill.side} {fill.quantity} @ {fill.executed_price:.4f}")
    print(f"   Slippage: ${fill.slippage:.4f}, Commission: ${fill.commission:.2f}")
    
    # Test market impact estimate
    impact = await adapter.get_market_impact_estimate("AAPL", 1000, 150.0)
    print(f"✅ Market impact: {impact['slippage_bps']:.1f}bps, Cost: {impact['total_cost_pct']:.3f}%")
    
    # Test partial fill
    partial = await adapter.simulate_partial_fill("TSLA", "sell", 50, 900.0, 0.6)
    print(f"✅ Partial fill: {partial.quantity}/{50} shares filled")
    
    # Get statistics
    stats = adapter.get_fill_statistics()
    print(f"✅ Statistics: {stats['adapter_type']}, ExecutionSim: {stats['has_execution_simulator']}")
    
    print("🎮 FillSimulatorAdapter test completed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fill_simulator_adapter())
