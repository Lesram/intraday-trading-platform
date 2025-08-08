#!/usr/bin/env python3
"""
🎯 STRATEGY STOP-LOSS INTEGRATIONS
Phase 1 Optimization: Implementing unified stop-losses across all strategies
Part of Audit Item 4: Trading Strategy Reevaluation
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

# Add current directory to Python path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our unified risk manager
from unified_risk_manager import get_risk_manager, PositionRisk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Types of stop-loss mechanisms"""
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted" 
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    ATR_BASED = "atr_based"

@dataclass
class StopLossOrder:
    """Stop-loss order details"""
    symbol: str
    stop_price: float
    order_type: StopLossType
    created_at: datetime
    strategy: str
    original_entry_price: float
    current_price: Optional[float] = None
    triggered: bool = False
    trigger_time: Optional[datetime] = None

class StrategyStopLossManager:
    """Manages stop-losses for all trading strategies with unified risk management"""
    
    def __init__(self):
        self.risk_manager = get_risk_manager()
        self.active_stops = {}  # symbol -> StopLossOrder
        self.stop_loss_history = []
        self.strategy_configs = self._initialize_strategy_configs()
        
        logger.info("🎯 Strategy Stop-Loss Manager initialized with unified risk management")
    
    def _initialize_strategy_configs(self) -> Dict[str, Dict]:
        """Initialize stop-loss configurations for each strategy"""
        return {
            "automated_signal_trading": {
                "primary_type": StopLossType.VOLATILITY_ADJUSTED,
                "secondary_type": StopLossType.TIME_BASED,
                "base_stop_pct": 0.03,           # 3% base stop
                "max_hold_hours": 24,            # Exit after 24 hours regardless
                "trailing_activation": 0.02,     # Activate trailing at 2% profit
                "confidence_adjustment": True     # Adjust based on signal confidence
            },
            "momentum_strategy": {
                "primary_type": StopLossType.TRAILING_STOP,
                "secondary_type": StopLossType.VOLATILITY_ADJUSTED,
                "base_stop_pct": 0.02,           # 2% base stop (tighter for momentum)
                "trailing_distance": 0.015,      # 1.5% trailing distance
                "momentum_acceleration": True,    # Tighten stops when momentum weakens
                "max_hold_days": 5               # Momentum trades are shorter-term
            },
            "mean_reversion_strategy": {
                "primary_type": StopLossType.FIXED_PERCENTAGE,
                "secondary_type": StopLossType.TIME_BASED,
                "base_stop_pct": 0.05,           # 5% wider stops for mean reversion
                "reversion_timeout": 72,         # Exit after 3 days if no reversion
                "support_level_stops": True,     # Use technical support levels
                "patience_multiplier": 1.5       # More patient with reversions
            },
            "portfolio_rebalancing": {
                "primary_type": StopLossType.ATR_BASED,
                "secondary_type": StopLossType.TIME_BASED,
                "base_stop_pct": 0.08,           # 8% very wide stops for long-term
                "rebalance_frequency": 30,       # Check stops monthly
                "fundamental_override": True,    # Don't stop if fundamentals strong
                "correlation_adjustment": True   # Adjust based on portfolio correlation
            }
        }
    
    async def create_stop_loss(self, symbol: str, entry_price: float, strategy: str, 
                             confidence: Optional[float] = None, volatility: Optional[float] = None) -> StopLossOrder:
        """Create a stop-loss order using unified risk management"""
        
        # Get strategy configuration
        config = self.strategy_configs.get(strategy, self.strategy_configs["automated_signal_trading"])
        
        # Use unified risk manager to calculate optimal stop-loss
        stop_price = self.risk_manager.calculate_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            volatility=volatility or 0.02,  # Default 2% volatility if not provided
            confidence=confidence or 0.5,   # Default 50% confidence if not provided
            strategy_type=strategy
        )
        
        # Create stop-loss order
        stop_order = StopLossOrder(
            symbol=symbol,
            stop_price=stop_price,
            order_type=config["primary_type"],
            created_at=datetime.now(),
            strategy=strategy,
            original_entry_price=entry_price
        )
        
        # Store active stop
        self.active_stops[symbol] = stop_order
        
        # Log the creation
        stop_pct = ((entry_price - stop_price) / entry_price) * 100
        logger.info(f"🎯 Created {config['primary_type'].value} stop-loss for {symbol}: "
                   f"${stop_price:.2f} ({stop_pct:.1f}% from entry ${entry_price:.2f})")
        
        return stop_order
    
    async def update_stop_loss(self, symbol: str, current_price: float) -> Optional[StopLossOrder]:
        """Update stop-loss based on current price and strategy rules"""
        
        if symbol not in self.active_stops:
            return None
        
        stop_order = self.active_stops[symbol]
        stop_order.current_price = current_price
        
        # Get strategy configuration
        config = self.strategy_configs[stop_order.strategy]
        
        # Handle different stop-loss types
        if stop_order.order_type == StopLossType.TRAILING_STOP:
            await self._update_trailing_stop(stop_order, current_price, config)
        elif stop_order.order_type == StopLossType.TIME_BASED:
            await self._check_time_based_stop(stop_order, config)
        elif stop_order.order_type == StopLossType.VOLATILITY_ADJUSTED:
            await self._update_volatility_stop(stop_order, current_price, config)
        
        return stop_order
    
    async def _update_trailing_stop(self, stop_order: StopLossOrder, current_price: float, config: Dict):
        """Update trailing stop-loss"""
        
        profit_pct = (current_price - stop_order.original_entry_price) / stop_order.original_entry_price
        
        # Only trail if in profit and above activation threshold
        activation_threshold = config.get("trailing_activation", 0.01)
        
        if profit_pct > activation_threshold:
            trailing_distance = config.get("trailing_distance", 0.015)
            new_stop_price = current_price * (1 - trailing_distance)
            
            # Only move stop up, never down
            if new_stop_price > stop_order.stop_price:
                old_stop = stop_order.stop_price
                stop_order.stop_price = new_stop_price
                logger.info(f"📈 Trailing stop updated for {stop_order.symbol}: "
                           f"${old_stop:.2f} → ${new_stop_price:.2f}")
    
    async def _check_time_based_stop(self, stop_order: StopLossOrder, config: Dict):
        """Check time-based stop conditions"""
        
        time_elapsed = datetime.now() - stop_order.created_at
        
        # Check maximum hold time
        if "max_hold_hours" in config:
            max_hours = config["max_hold_hours"]
            if time_elapsed.total_seconds() / 3600 > max_hours:
                await self._trigger_stop_loss(stop_order, f"Time-based exit: held for {max_hours} hours")
        
        elif "max_hold_days" in config:
            max_days = config["max_hold_days"]
            if time_elapsed.days > max_days:
                await self._trigger_stop_loss(stop_order, f"Time-based exit: held for {max_days} days")
        
        elif "reversion_timeout" in config:
            timeout_hours = config["reversion_timeout"]
            if time_elapsed.total_seconds() / 3600 > timeout_hours:
                await self._trigger_stop_loss(stop_order, f"Mean reversion timeout: {timeout_hours} hours")
    
    async def _update_volatility_stop(self, stop_order: StopLossOrder, current_price: float, config: Dict):
        """Update volatility-adjusted stop-loss"""
        
        # This would typically use recent volatility data to adjust the stop
        # For now, we'll use a simple volatility estimate
        volatility = abs(current_price - stop_order.original_entry_price) / stop_order.original_entry_price
        
        if volatility > 0.05:  # If volatility is high, slightly widen the stop
            adjustment_factor = min(1.2, 1 + volatility)
            base_stop_distance = stop_order.original_entry_price - stop_order.stop_price
            new_stop_distance = base_stop_distance * adjustment_factor
            new_stop_price = stop_order.original_entry_price - new_stop_distance
            
            # Only widen the stop if it helps (move it down for long positions)
            if new_stop_price < stop_order.stop_price:
                stop_order.stop_price = new_stop_price
                logger.info(f"📊 Volatility-adjusted stop for {stop_order.symbol}: ${new_stop_price:.2f}")
    
    async def check_stop_triggers(self, market_data: Dict[str, float]) -> List[StopLossOrder]:
        """Check all active stops for trigger conditions"""
        
        triggered_stops = []
        
        for symbol, stop_order in list(self.active_stops.items()):
            if symbol in market_data:
                current_price = market_data[symbol]
                
                # Update the stop first
                await self.update_stop_loss(symbol, current_price)
                
                # Check if stop is triggered
                if current_price <= stop_order.stop_price and not stop_order.triggered:
                    await self._trigger_stop_loss(stop_order, f"Price trigger: ${current_price:.2f} <= ${stop_order.stop_price:.2f}")
                    triggered_stops.append(stop_order)
        
        return triggered_stops
    
    async def _trigger_stop_loss(self, stop_order: StopLossOrder, reason: str):
        """Trigger a stop-loss order"""
        
        stop_order.triggered = True
        stop_order.trigger_time = datetime.now()
        
        # Calculate the loss
        if stop_order.current_price:
            loss_pct = ((stop_order.original_entry_price - stop_order.current_price) / stop_order.original_entry_price) * 100
            loss_amount = stop_order.original_entry_price - stop_order.current_price
        else:
            loss_pct = ((stop_order.original_entry_price - stop_order.stop_price) / stop_order.original_entry_price) * 100
            loss_amount = stop_order.original_entry_price - stop_order.stop_price
        
        logger.warning(f"🛑 STOP-LOSS TRIGGERED for {stop_order.symbol}: {reason}")
        logger.warning(f"   Entry: ${stop_order.original_entry_price:.2f}, Exit: ${stop_order.stop_price:.2f}")
        logger.warning(f"   Loss: {loss_pct:.1f}% (${loss_amount:.2f} per share)")
        
        # Move to history and remove from active stops
        self.stop_loss_history.append(stop_order)
        del self.active_stops[stop_order.symbol]
        
        # Here you would integrate with your actual trading system to execute the stop-loss order
        await self._execute_stop_loss_order(stop_order)
    
    async def _execute_stop_loss_order(self, stop_order: StopLossOrder):
        """Execute the actual stop-loss order (integration point with trading system)"""
        
        logger.info(f"🔄 Executing stop-loss order for {stop_order.symbol} at ${stop_order.stop_price:.2f}")
        
        # This would integrate with your actual trading API
        # For now, we'll just log the intended action
        order_details = {
            "symbol": stop_order.symbol,
            "action": "SELL",  # Assuming long positions
            "order_type": "MARKET",  # or "STOP_MARKET"
            "trigger_price": stop_order.stop_price,
            "strategy": stop_order.strategy,
            "reason": "stop_loss_triggered"
        }
        
        logger.info(f"📤 Stop-loss order details: {order_details}")
        
        # TODO: Integrate with actual trading system
        # await trading_api.place_order(order_details)
    
    def get_strategy_stop_loss_stats(self) -> Dict[str, Any]:
        """Get statistics about stop-losses by strategy"""
        
        stats = {}
        
        for strategy in self.strategy_configs.keys():
            strategy_stops = [s for s in self.stop_loss_history if s.strategy == strategy]
            
            if strategy_stops:
                total_stops = len(strategy_stops)
                avg_loss = sum((s.original_entry_price - s.stop_price) / s.original_entry_price 
                             for s in strategy_stops) / total_stops * 100
                
                stats[strategy] = {
                    "total_stops_triggered": total_stops,
                    "average_loss_pct": f"{avg_loss:.1f}%",
                    "active_stops": len([s for s in self.active_stops.values() if s.strategy == strategy])
                }
            else:
                stats[strategy] = {
                    "total_stops_triggered": 0,
                    "average_loss_pct": "N/A",
                    "active_stops": len([s for s in self.active_stops.values() if s.strategy == strategy])
                }
        
        return stats
    
    def get_stop_loss_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive stop-loss dashboard"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_stops": len(self.active_stops),
            "triggered_today": len([s for s in self.stop_loss_history 
                                   if s.trigger_time and s.trigger_time.date() == datetime.now().date()]),
            "strategy_stats": self.get_strategy_stop_loss_stats(),
            "active_positions": [
                {
                    "symbol": stop.symbol,
                    "strategy": stop.strategy,
                    "entry_price": stop.original_entry_price,
                    "stop_price": stop.stop_price,
                    "stop_type": stop.order_type.value,
                    "age_hours": (datetime.now() - stop.created_at).total_seconds() / 3600
                }
                for stop in self.active_stops.values()
            ]
        }

# Global stop-loss manager instance
stop_loss_manager = StrategyStopLossManager()

def get_stop_loss_manager() -> StrategyStopLossManager:
    """Get the global stop-loss manager instance"""
    return stop_loss_manager

# Integration functions for each strategy
class AutomatedSignalTradingIntegration:
    """Stop-loss integration for automated signal trading strategy"""
    
    @staticmethod
    async def create_signal_stop_loss(symbol: str, entry_price: float, signal_confidence: float) -> StopLossOrder:
        """Create stop-loss for automated signal trading"""
        return await stop_loss_manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            strategy="automated_signal_trading",
            confidence=signal_confidence,
            volatility=0.025  # Default volatility for automated signals
        )

class MomentumStrategyIntegration:
    """Stop-loss integration for momentum strategy"""
    
    @staticmethod
    async def create_momentum_stop_loss(symbol: str, entry_price: float, momentum_strength: float) -> StopLossOrder:
        """Create stop-loss for momentum strategy with trailing capabilities"""
        return await stop_loss_manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            strategy="momentum_strategy",
            confidence=momentum_strength,
            volatility=0.03  # Momentum stocks tend to be more volatile
        )

class MeanReversionStrategyIntegration:
    """Stop-loss integration for mean reversion strategy"""
    
    @staticmethod
    async def create_reversion_stop_loss(symbol: str, entry_price: float, support_level: float) -> StopLossOrder:
        """Create stop-loss for mean reversion with support level consideration"""
        return await stop_loss_manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            strategy="mean_reversion_strategy",
            confidence=0.6,  # Moderate confidence for mean reversion
            volatility=0.02  # Lower volatility assumption for mean reversion
        )

class PortfolioRebalancingIntegration:
    """Stop-loss integration for portfolio rebalancing strategy"""
    
    @staticmethod
    async def create_rebalancing_stop_loss(symbol: str, entry_price: float, portfolio_weight: float) -> StopLossOrder:
        """Create stop-loss for portfolio rebalancing positions"""
        return await stop_loss_manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            strategy="portfolio_rebalancing",
            confidence=0.7,  # Higher confidence for diversified positions
            volatility=0.018  # Lower volatility for diversified portfolio positions
        )

if __name__ == "__main__":
    # Test the stop-loss integration system
    logger.info("🧪 Testing Strategy Stop-Loss Integration System...")
    
    async def test_stop_loss_system():
        # Test creating stop-losses for different strategies
        
        # Automated signal trading
        signal_stop = await AutomatedSignalTradingIntegration.create_signal_stop_loss("AAPL", 150.0, 0.8)
        print(f"Signal Trading Stop: {signal_stop.symbol} @ ${signal_stop.stop_price:.2f}")
        
        # Momentum strategy
        momentum_stop = await MomentumStrategyIntegration.create_momentum_stop_loss("TSLA", 900.0, 0.75)
        print(f"Momentum Stop: {momentum_stop.symbol} @ ${momentum_stop.stop_price:.2f}")
        
        # Mean reversion
        reversion_stop = await MeanReversionStrategyIntegration.create_reversion_stop_loss("MSFT", 380.0, 370.0)
        print(f"Mean Reversion Stop: {reversion_stop.symbol} @ ${reversion_stop.stop_price:.2f}")
        
        # Portfolio rebalancing
        portfolio_stop = await PortfolioRebalancingIntegration.create_rebalancing_stop_loss("SPY", 450.0, 0.25)
        print(f"Portfolio Stop: {portfolio_stop.symbol} @ ${portfolio_stop.stop_price:.2f}")
        
        # Test stop-loss triggers
        market_data = {"AAPL": 145.0, "TSLA": 850.0, "MSFT": 375.0, "SPY": 445.0}
        triggered = await stop_loss_manager.check_stop_triggers(market_data)
        
        print(f"\nTriggered stops: {len(triggered)}")
        for stop in triggered:
            print(f"  {stop.symbol}: {stop.order_type.value}")
        
        # Get dashboard
        dashboard = stop_loss_manager.get_stop_loss_dashboard()
        print(f"\nActive stops: {dashboard['active_stops']}")
        print(f"Triggered today: {dashboard['triggered_today']}")
        
        print("\n✅ Strategy Stop-Loss Integration test completed!")
    
    # Run the test
    asyncio.run(test_stop_loss_system())
