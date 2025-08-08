#!/usr/bin/env python3
"""
🛡️ DYNAMIC STOP-LOSS & TAKE-PROFIT MODULE
ATR-based dynamic stops with trailing functionality
Institutional-grade risk management for live trading
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
import alpaca_trade_api as tradeapi
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StopType(Enum):
    INITIAL_STOP = "initial"
    TRAILING_STOP = "trailing"
    TAKE_PROFIT = "take_profit"
    VOLATILITY_STOP = "volatility"

@dataclass
class StopOrder:
    """Represents a dynamic stop order"""
    symbol: str
    order_id: Optional[str]
    stop_type: StopType
    stop_price: float
    original_stop: float
    position_side: str  # 'long' or 'short'
    entry_price: float
    atr_multiplier: float
    created_at: datetime
    last_updated: datetime
    is_active: bool = True

class DynamicStopManager:
    """
    Manages dynamic stop-loss and take-profit orders with ATR-based calculations
    
    Features:
    - Initial stop: 1.5x ATR from entry
    - Trailing stop: 1.0x ATR from highest/lowest point
    - Take profit: 2.5x ATR target (risk/reward = 1.67)
    - Volatility-based adjustments
    - Real-time monitoring and updates
    """
    
    def __init__(self, api_client: tradeapi.REST):
        self.api = api_client
        self.active_stops = {}  # symbol -> StopOrder
        self.position_highs = {}  # symbol -> highest price since entry (for longs)
        self.position_lows = {}   # symbol -> lowest price since entry (for shorts)
        
        # ATR multipliers (backtested optimal values)
        self.initial_stop_multiplier = 1.5    # Initial stop distance
        self.trailing_stop_multiplier = 1.0   # Trailing stop distance
        self.take_profit_multiplier = 2.5     # Take profit target
        self.min_trail_distance = 0.005       # Minimum 0.5% trail distance
        
        # Update frequencies
        self.last_update = {}  # symbol -> datetime
        self.update_interval = 60  # Update stops every 60 seconds
        
        logger.info("🛡️ Dynamic stop manager initialized")
    
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range for stop distance"""
        try:
            # Get enough data for ATR calculation
            bars = self.api.get_bars(symbol, "5Min", limit=period + 10)
            if not bars or len(bars) < period:
                logger.warning(f"Insufficient data for {symbol} ATR, using default")
                return 0.02  # 2% default ATR
            
            # Calculate True Range for each bar
            true_ranges = []
            for i in range(1, len(bars)):
                high = float(bars[i].h)
                low = float(bars[i].l)
                prev_close = float(bars[i-1].c)
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr / float(bars[i].c))  # Normalize by price
            
            # Simple moving average of True Ranges
            atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)
            
            logger.debug(f"📊 {symbol} ATR: {atr:.4f} ({atr*100:.2f}%)")
            return max(0.005, atr)  # Minimum 0.5% ATR
            
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.02
    
    def create_initial_stops(
        self, 
        symbol: str, 
        entry_price: float, 
        position_side: str, 
        quantity: float
    ) -> Dict[str, StopOrder]:
        """Create initial stop-loss and take-profit orders for a new position"""
        try:
            atr = self.calculate_atr(symbol)
            current_time = datetime.now()
            
            stops_created = {}
            
            if position_side.lower() == 'long':
                # Long position stops
                initial_stop_price = entry_price * (1 - atr * self.initial_stop_multiplier)
                take_profit_price = entry_price * (1 + atr * self.take_profit_multiplier)
                
                # Create stop-loss order
                try:
                    stop_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='stop',
                        time_in_force='gtc',
                        stop_price=round(initial_stop_price, 2)
                    )
                    
                    stops_created['stop_loss'] = StopOrder(
                        symbol=symbol,
                        order_id=stop_order.id,
                        stop_type=StopType.INITIAL_STOP,
                        stop_price=initial_stop_price,
                        original_stop=initial_stop_price,
                        position_side='long',
                        entry_price=entry_price,
                        atr_multiplier=self.initial_stop_multiplier,
                        created_at=current_time,
                        last_updated=current_time
                    )
                    
                    logger.info(f"🛑 Created stop-loss for {symbol}: ${initial_stop_price:.2f} (ATR: {atr:.3f})")
                    
                except Exception as e:
                    logger.error(f"Failed to create stop-loss for {symbol}: {e}")
                
                # Create take-profit order (limit order)
                try:
                    tp_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='limit',
                        time_in_force='gtc',
                        limit_price=round(take_profit_price, 2)
                    )
                    
                    stops_created['take_profit'] = StopOrder(
                        symbol=symbol,
                        order_id=tp_order.id,
                        stop_type=StopType.TAKE_PROFIT,
                        stop_price=take_profit_price,
                        original_stop=take_profit_price,
                        position_side='long',
                        entry_price=entry_price,
                        atr_multiplier=self.take_profit_multiplier,
                        created_at=current_time,
                        last_updated=current_time
                    )
                    
                    logger.info(f"🎯 Created take-profit for {symbol}: ${take_profit_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to create take-profit for {symbol}: {e}")
            
            else:  # Short position
                # Short position stops
                initial_stop_price = entry_price * (1 + atr * self.initial_stop_multiplier)
                take_profit_price = entry_price * (1 - atr * self.take_profit_multiplier)
                
                # Create stop-loss order (buy to cover)
                try:
                    stop_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='stop',
                        time_in_force='gtc',
                        stop_price=round(initial_stop_price, 2)
                    )
                    
                    stops_created['stop_loss'] = StopOrder(
                        symbol=symbol,
                        order_id=stop_order.id,
                        stop_type=StopType.INITIAL_STOP,
                        stop_price=initial_stop_price,
                        original_stop=initial_stop_price,
                        position_side='short',
                        entry_price=entry_price,
                        atr_multiplier=self.initial_stop_multiplier,
                        created_at=current_time,
                        last_updated=current_time
                    )
                    
                    logger.info(f"🛑 Created stop-loss for {symbol} SHORT: ${initial_stop_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to create stop-loss for {symbol} SHORT: {e}")
                
                # Create take-profit order
                try:
                    tp_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='limit',
                        time_in_force='gtc',
                        limit_price=round(take_profit_price, 2)
                    )
                    
                    stops_created['take_profit'] = StopOrder(
                        symbol=symbol,
                        order_id=tp_order.id,
                        stop_type=StopType.TAKE_PROFIT,
                        stop_price=take_profit_price,
                        original_stop=take_profit_price,
                        position_side='short',
                        entry_price=entry_price,
                        atr_multiplier=self.take_profit_multiplier,
                        created_at=current_time,
                        last_updated=current_time
                    )
                    
                    logger.info(f"🎯 Created take-profit for {symbol} SHORT: ${take_profit_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to create take-profit for {symbol} SHORT: {e}")
            
            # Store active stops
            self.active_stops[symbol] = stops_created
            self.position_highs[symbol] = entry_price
            self.position_lows[symbol] = entry_price
            self.last_update[symbol] = current_time
            
            return stops_created
            
        except Exception as e:
            logger.error(f"Error creating initial stops for {symbol}: {e}")
            return {}
    
    def update_trailing_stops(self, symbol: str, current_price: float) -> bool:
        """Update trailing stops based on current price movement"""
        try:
            if symbol not in self.active_stops:
                return False
            
            current_time = datetime.now()
            
            # Throttle updates (don't spam the API)
            if symbol in self.last_update:
                time_since_update = (current_time - self.last_update[symbol]).seconds
                if time_since_update < self.update_interval:
                    return False
            
            stops = self.active_stops[symbol]
            if 'stop_loss' not in stops or not stops['stop_loss'].is_active:
                return False
            
            stop_order = stops['stop_loss']
            atr = self.calculate_atr(symbol)
            updated = False
            
            if stop_order.position_side == 'long':
                # Update position high
                if current_price > self.position_highs[symbol]:
                    self.position_highs[symbol] = current_price
                
                # Calculate new trailing stop
                trailing_stop_price = self.position_highs[symbol] * (1 - atr * self.trailing_stop_multiplier)
                
                # Only move stop up (for longs)
                if trailing_stop_price > stop_order.stop_price:
                    # Ensure minimum trail distance
                    min_stop_price = current_price * (1 - self.min_trail_distance)
                    final_stop_price = min(trailing_stop_price, min_stop_price)
                    
                    # Update the stop order
                    try:
                        # Cancel old stop
                        self.api.cancel_order(stop_order.order_id)
                        
                        # Create new trailing stop
                        position = self.api.get_position(symbol)
                        new_stop_order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(float(position.qty)),
                            side='sell',
                            type='stop',
                            time_in_force='gtc',
                            stop_price=round(final_stop_price, 2)
                        )
                        
                        # Update stop record
                        stop_order.order_id = new_stop_order.id
                        stop_order.stop_price = final_stop_price
                        stop_order.stop_type = StopType.TRAILING_STOP
                        stop_order.last_updated = current_time
                        
                        logger.info(f"📈 Updated trailing stop for {symbol}: ${final_stop_price:.2f} (High: ${self.position_highs[symbol]:.2f})")
                        updated = True
                        
                    except Exception as e:
                        logger.error(f"Failed to update trailing stop for {symbol}: {e}")
            
            else:  # Short position
                # Update position low
                if current_price < self.position_lows[symbol]:
                    self.position_lows[symbol] = current_price
                
                # Calculate new trailing stop
                trailing_stop_price = self.position_lows[symbol] * (1 + atr * self.trailing_stop_multiplier)
                
                # Only move stop down (for shorts)
                if trailing_stop_price < stop_order.stop_price:
                    # Ensure minimum trail distance
                    max_stop_price = current_price * (1 + self.min_trail_distance)
                    final_stop_price = max(trailing_stop_price, max_stop_price)
                    
                    # Update the stop order
                    try:
                        # Cancel old stop
                        self.api.cancel_order(stop_order.order_id)
                        
                        # Create new trailing stop
                        position = self.api.get_position(symbol)
                        new_stop_order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(float(position.qty)),
                            side='buy',
                            type='stop',
                            time_in_force='gtc',
                            stop_price=round(final_stop_price, 2)
                        )
                        
                        # Update stop record
                        stop_order.order_id = new_stop_order.id
                        stop_order.stop_price = final_stop_price
                        stop_order.stop_type = StopType.TRAILING_STOP
                        stop_order.last_updated = current_time
                        
                        logger.info(f"📉 Updated trailing stop for {symbol} SHORT: ${final_stop_price:.2f} (Low: ${self.position_lows[symbol]:.2f})")
                        updated = True
                        
                    except Exception as e:
                        logger.error(f"Failed to update trailing stop for {symbol} SHORT: {e}")
            
            if updated:
                self.last_update[symbol] = current_time
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating trailing stops for {symbol}: {e}")
            return False
    
    def cleanup_position_stops(self, symbol: str) -> bool:
        """Clean up stops when position is closed"""
        try:
            if symbol not in self.active_stops:
                return True
            
            stops = self.active_stops[symbol]
            
            # Cancel any remaining active orders
            for stop_type, stop_order in stops.items():
                if stop_order.is_active and stop_order.order_id:
                    try:
                        self.api.cancel_order(stop_order.order_id)
                        logger.info(f"🗑️ Cancelled {stop_type} order for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel {stop_type} for {symbol}: {e}")
            
            # Clean up tracking data
            del self.active_stops[symbol]
            if symbol in self.position_highs:
                del self.position_highs[symbol]
            if symbol in self.position_lows:
                del self.position_lows[symbol]
            if symbol in self.last_update:
                del self.last_update[symbol]
            
            logger.info(f"🧹 Cleaned up stops for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up stops for {symbol}: {e}")
            return False
    
    def monitor_all_stops(self) -> Dict[str, Dict]:
        """Monitor and update all active stops"""
        try:
            positions = self.api.list_positions()
            position_symbols = {pos.symbol for pos in positions}
            
            # Clean up stops for closed positions
            for symbol in list(self.active_stops.keys()):
                if symbol not in position_symbols:
                    self.cleanup_position_stops(symbol)
            
            # Update trailing stops for open positions
            updates = {}
            for position in positions:
                symbol = position.symbol
                current_price = float(position.current_price)
                
                # Update trailing stops
                updated = self.update_trailing_stops(symbol, current_price)
                if updated:
                    updates[symbol] = {
                        'current_price': current_price,
                        'stop_price': self.active_stops[symbol]['stop_loss'].stop_price,
                        'position_high': self.position_highs.get(symbol),
                        'position_low': self.position_lows.get(symbol)
                    }
            
            return updates
            
        except Exception as e:
            logger.error(f"Error monitoring stops: {e}")
            return {}
    
    def get_stop_status(self, symbol: str) -> Dict:
        """Get current stop status for a symbol"""
        if symbol not in self.active_stops:
            return {'has_stops': False}
        
        stops = self.active_stops[symbol]
        stop_loss = stops.get('stop_loss')
        take_profit = stops.get('take_profit')
        
        return {
            'has_stops': True,
            'stop_loss': {
                'price': stop_loss.stop_price if stop_loss else None,
                'type': stop_loss.stop_type.value if stop_loss else None,
                'order_id': stop_loss.order_id if stop_loss else None
            } if stop_loss else None,
            'take_profit': {
                'price': take_profit.stop_price if take_profit else None,
                'order_id': take_profit.order_id if take_profit else None
            } if take_profit else None,
            'position_high': self.position_highs.get(symbol),
            'position_low': self.position_lows.get(symbol),
            'last_updated': self.last_update.get(symbol)
        }

# Global instance
dynamic_stop_manager = None

def initialize_dynamic_stops(api_client: tradeapi.REST):
    """Initialize the global dynamic stop manager"""
    global dynamic_stop_manager
    dynamic_stop_manager = DynamicStopManager(api_client)
    logger.info("🛡️ Dynamic stop manager initialized")

def create_stops_for_position(symbol: str, entry_price: float, side: str, quantity: float):
    """Convenience function to create stops for a new position"""
    if dynamic_stop_manager is None:
        raise RuntimeError("Dynamic stop manager not initialized")
    
    return dynamic_stop_manager.create_initial_stops(symbol, entry_price, side, quantity)

def monitor_stops():
    """Convenience function to monitor all stops"""
    if dynamic_stop_manager is None:
        return {}
    
    return dynamic_stop_manager.monitor_all_stops()
