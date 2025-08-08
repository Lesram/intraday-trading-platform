#!/usr/bin/env python3
"""
🛡️ ADVANCED DYNAMIC STOP-LOSS OPTIMIZATION MODULE
Enhanced volatility-based stops with ATR scaling, ML-based placement, and time-decay components
Priority 2A implementation based on AI advisor feedback
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import alpaca_trade_api as tradeapi
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class StopConfiguration:
    """Configuration for dynamic stop-loss parameters"""
    initial_stop_pct: float = 0.02  # 2% initial stop
    max_stop_pct: float = 0.05      # 5% maximum stop
    min_stop_pct: float = 0.008     # 0.8% minimum stop
    atr_multiplier: float = 2.0     # ATR scaling factor
    time_decay_hours: int = 24      # Hours for time decay
    volatility_lookback: int = 20   # Days for volatility calculation
    profit_trailing_ratio: float = 0.5  # Trail at 50% of profit
    
@dataclass
class StopLossData:
    """Data structure for stop-loss information"""
    symbol: str
    entry_price: float
    current_stop_price: float
    initial_stop_price: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    last_update: datetime
    stop_type: str  # 'fixed', 'trailing', 'volatility_adjusted'
    atr_at_entry: float
    profit_high_water_mark: float
    time_decay_adjustment: float
    ml_adjustment_factor: float

class AdvancedDynamicStopOptimizer:
    """
    Advanced dynamic stop-loss optimization system with:
    - Volatility-based stop adjustment
    - ATR scaling with regime detection
    - ML-based stop placement optimization
    - Time-decay components
    - Profit trailing with high-water mark
    """
    
    def __init__(self, api_client: tradeapi.REST, config: StopConfiguration = None):
        self.api = api_client
        self.config = config or StopConfiguration()
        self.active_stops: Dict[str, StopLossData] = {}
        self.volatility_cache = {}
        self.atr_cache = {}
        self.ml_stop_cache = {}
        
        logger.info("🛡️ Advanced Dynamic Stop Optimizer initialized")
    
    def calculate_atr_scaled_stop(self, symbol: str, entry_price: float, 
                                 side: str) -> Tuple[float, float]:
        """
        Calculate ATR-scaled stop-loss distance
        
        Returns:
            Tuple of (stop_price, atr_value)
        """
        try:
            # Get ATR data
            atr = self._get_atr(symbol)
            
            # Calculate base stop distance using ATR
            atr_stop_distance = atr * self.config.atr_multiplier
            
            # Convert to percentage
            atr_stop_pct = atr_stop_distance / entry_price
            
            # Apply bounds
            stop_pct = max(self.config.min_stop_pct, 
                          min(self.config.max_stop_pct, atr_stop_pct))
            
            # Calculate stop price
            if side == 'long':
                stop_price = entry_price * (1 - stop_pct)
            else:  # short
                stop_price = entry_price * (1 + stop_pct)
            
            logger.info(f"📊 ATR stop for {symbol}: ATR=${atr:.3f}, "
                       f"Distance={stop_pct:.1%}, Stop=${stop_price:.2f}")
            
            return stop_price, atr
            
        except Exception as e:
            logger.warning(f"Error calculating ATR stop for {symbol}: {e}")
            # Fallback to fixed percentage
            fallback_pct = self.config.initial_stop_pct
            if side == 'long':
                return entry_price * (1 - fallback_pct), 0.01
            else:
                return entry_price * (1 + fallback_pct), 0.01
    
    def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement"""
        try:
            # Check cache
            cache_key = f"{symbol}_{period}"
            if cache_key in self.atr_cache:
                cached_data = self.atr_cache[cache_key]
                if (datetime.now() - cached_data['updated']).seconds < 900:  # 15 min cache
                    return cached_data['atr']
            
            # Get market data
            bars = self.api.get_bars(symbol, '1Day', limit=period + 5, adjustment='raw')
            if not bars or len(bars) < period:
                # Use 1% of recent price as fallback
                recent_bars = self.api.get_bars(symbol, '1Day', limit=1, adjustment='raw')
                if recent_bars and len(recent_bars) > 0:
                    recent_price = float(recent_bars[-1].c)
                    return recent_price * 0.01
                return 1.0  # Absolute fallback
            
            # Calculate True Range
            highs = [float(bar.h) for bar in bars]
            lows = [float(bar.l) for bar in bars]
            closes = [float(bar.c) for bar in bars]
            
            true_ranges = []
            for i in range(1, len(bars)):
                tr1 = highs[i] - lows[i]  # Current high - current low
                tr2 = abs(highs[i] - closes[i-1])  # Current high - previous close
                tr3 = abs(lows[i] - closes[i-1])   # Current low - previous close
                true_ranges.append(max(tr1, tr2, tr3))
            
            # Calculate ATR (average of true ranges)
            atr = sum(true_ranges[-period:]) / min(period, len(true_ranges))
            
            # Update cache
            self.atr_cache[cache_key] = {
                'atr': atr,
                'updated': datetime.now()
            }
            
            return atr
            
        except Exception as e:
            logger.warning(f"Error calculating ATR for {symbol}: {e}")
            return 1.0  # Default fallback
    
    def calculate_ml_optimized_stop(self, symbol: str, entry_price: float, 
                                   side: str, confidence: float = 0.7) -> float:
        """
        Use ML-based approach to optimize stop placement
        Based on historical price action and volatility patterns
        """
        try:
            # Get recent price data for pattern analysis
            bars = self.api.get_bars(symbol, '1Hour', limit=100, adjustment='raw')
            if not bars or len(bars) < 50:
                return 1.0  # Default adjustment
            
            prices = [float(bar.c) for bar in bars]
            volumes = [int(bar.v) for bar in bars]
            
            # Calculate volatility features
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = np.std(returns) * np.sqrt(24)  # Hourly to daily volatility
            
            # Volume profile analysis
            avg_volume = np.mean(volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum analysis
            momentum_5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0
            momentum_20 = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0
            
            # ML-based adjustment factors
            # High volatility -> wider stops
            volatility_factor = max(0.8, min(1.5, 1 + (volatility - 0.02) * 10))
            
            # High volume -> tighter stops (more conviction)
            volume_factor = max(0.9, min(1.1, 2 - volume_ratio * 0.1))
            
            # Strong momentum -> wider stops (trend following)
            momentum_factor = 1.0
            if abs(momentum_5) > 0.02:  # Strong 5-hour momentum
                momentum_factor = 1.2 if momentum_5 * (1 if side == 'long' else -1) > 0 else 0.9
            
            # Confidence adjustment
            confidence_factor = max(0.9, min(1.1, confidence))
            
            # Combined ML adjustment
            ml_adjustment = volatility_factor * volume_factor * momentum_factor * confidence_factor
            
            # Cache the result
            self.ml_stop_cache[symbol] = {
                'adjustment': ml_adjustment,
                'features': {
                    'volatility': volatility,
                    'volume_ratio': volume_ratio,
                    'momentum_5h': momentum_5,
                    'momentum_20h': momentum_20
                },
                'updated': datetime.now()
            }
            
            logger.info(f"🤖 ML stop adjustment for {symbol}: {ml_adjustment:.2f} "
                       f"(vol={volatility_factor:.2f}, volume={volume_factor:.2f}, "
                       f"momentum={momentum_factor:.2f})")
            
            return ml_adjustment
            
        except Exception as e:
            logger.warning(f"Error calculating ML stop for {symbol}: {e}")
            return 1.0
    
    def calculate_time_decay_adjustment(self, entry_time: datetime) -> float:
        """
        Calculate time-based adjustment to stop-loss
        Gradually widen stops as position ages (time decay)
        """
        try:
            hours_since_entry = (datetime.now() - entry_time).total_seconds() / 3600
            
            # Time decay curve: gradually widen stops over 24 hours
            max_decay_hours = self.config.time_decay_hours
            
            if hours_since_entry <= 1:
                # Tight stops in first hour
                return 0.8
            elif hours_since_entry <= 4:
                # Gradual widening 1-4 hours
                return 0.8 + (hours_since_entry - 1) * 0.1 / 3
            elif hours_since_entry <= max_decay_hours:
                # Continue widening up to max decay
                decay_progress = (hours_since_entry - 4) / (max_decay_hours - 4)
                return 0.9 + decay_progress * 0.3  # Up to 1.2x adjustment
            else:
                # Max decay reached
                return 1.2
                
        except Exception as e:
            logger.warning(f"Error calculating time decay: {e}")
            return 1.0
    
    def create_optimized_stop(self, symbol: str, entry_price: float, 
                             side: str, quantity: float, 
                             confidence: float = 0.7) -> Optional[StopLossData]:
        """
        Create an optimized stop-loss using all advanced methods
        """
        try:
            entry_time = datetime.now()
            
            # 1. Calculate ATR-based stop
            atr_stop_price, atr_value = self.calculate_atr_scaled_stop(
                symbol, entry_price, side)
            
            # 2. Get ML optimization adjustment
            ml_adjustment = self.calculate_ml_optimized_stop(
                symbol, entry_price, side, confidence)
            
            # 3. Apply time decay (starts at entry)
            time_decay = self.calculate_time_decay_adjustment(entry_time)
            
            # 4. Combine all adjustments
            base_stop_distance = abs(atr_stop_price - entry_price)
            adjusted_stop_distance = base_stop_distance * ml_adjustment * time_decay
            
            # 5. Calculate final stop price with bounds checking
            if side == 'long':
                final_stop_price = entry_price - adjusted_stop_distance
                # Ensure stop is below entry
                final_stop_price = min(final_stop_price, entry_price * 0.95)
            else:  # short
                final_stop_price = entry_price + adjusted_stop_distance
                # Ensure stop is above entry
                final_stop_price = max(final_stop_price, entry_price * 1.05)
            
            # Create stop-loss data structure
            stop_data = StopLossData(
                symbol=symbol,
                entry_price=entry_price,
                current_stop_price=final_stop_price,
                initial_stop_price=final_stop_price,
                side=side,
                entry_time=entry_time,
                last_update=entry_time,
                stop_type='volatility_adjusted',
                atr_at_entry=atr_value,
                profit_high_water_mark=0.0,
                time_decay_adjustment=time_decay,
                ml_adjustment_factor=ml_adjustment
            )
            
            # Store active stop
            self.active_stops[symbol] = stop_data
            
            logger.info(f"🎯 Optimized stop created for {symbol}: "
                       f"Entry=${entry_price:.2f}, Stop=${final_stop_price:.2f}, "
                       f"Distance={abs(final_stop_price-entry_price)/entry_price:.1%}")
            
            return stop_data
            
        except Exception as e:
            logger.error(f"Error creating optimized stop for {symbol}: {e}")
            return None
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Update trailing stop based on profit high-water mark
        """
        try:
            if symbol not in self.active_stops:
                return None
            
            stop_data = self.active_stops[symbol]
            
            # Calculate current profit
            if stop_data.side == 'long':
                current_profit = (current_price - stop_data.entry_price) / stop_data.entry_price
                profit_target = current_price > stop_data.entry_price
            else:  # short
                current_profit = (stop_data.entry_price - current_price) / stop_data.entry_price
                profit_target = current_price < stop_data.entry_price
            
            # Update high-water mark if profit increased
            if current_profit > stop_data.profit_high_water_mark:
                stop_data.profit_high_water_mark = current_profit
                
                # Trail the stop at a percentage of the high-water mark
                if profit_target and current_profit > 0.01:  # Only trail if in profit > 1%
                    trailing_distance = stop_data.profit_high_water_mark * self.config.profit_trailing_ratio
                    
                    if stop_data.side == 'long':
                        new_stop_price = stop_data.entry_price * (1 + trailing_distance)
                        # Only move stop up (more favorable)
                        if new_stop_price > stop_data.current_stop_price:
                            stop_data.current_stop_price = new_stop_price
                            stop_data.stop_type = 'trailing'
                    else:  # short
                        new_stop_price = stop_data.entry_price * (1 - trailing_distance)
                        # Only move stop down (more favorable)
                        if new_stop_price < stop_data.current_stop_price:
                            stop_data.current_stop_price = new_stop_price
                            stop_data.stop_type = 'trailing'
                    
                    stop_data.last_update = datetime.now()
                    
                    logger.info(f"📈 Trailing stop updated for {symbol}: "
                               f"Profit={current_profit:.1%}, New stop=${stop_data.current_stop_price:.2f}")
            
            return stop_data.current_stop_price
            
        except Exception as e:
            logger.warning(f"Error updating trailing stop for {symbol}: {e}")
            return None
    
    def update_time_decay_stops(self) -> List[str]:
        """
        Update all stops for time decay adjustment
        """
        updated_symbols = []
        
        try:
            for symbol, stop_data in self.active_stops.items():
                # Calculate new time decay adjustment
                new_time_decay = self.calculate_time_decay_adjustment(stop_data.entry_time)
                
                # Only update if decay has changed significantly
                if abs(new_time_decay - stop_data.time_decay_adjustment) > 0.1:
                    # Recalculate stop with new time decay
                    base_distance = abs(stop_data.initial_stop_price - stop_data.entry_price)
                    adjusted_distance = base_distance * stop_data.ml_adjustment_factor * new_time_decay
                    
                    if stop_data.side == 'long':
                        new_stop_price = stop_data.entry_price - adjusted_distance
                    else:
                        new_stop_price = stop_data.entry_price + adjusted_distance
                    
                    # Only widen stops (more conservative), never tighten due to time decay
                    stop_moved = False
                    if stop_data.side == 'long' and new_stop_price < stop_data.current_stop_price:
                        stop_data.current_stop_price = new_stop_price
                        stop_moved = True
                    elif stop_data.side == 'short' and new_stop_price > stop_data.current_stop_price:
                        stop_data.current_stop_price = new_stop_price
                        stop_moved = True
                    
                    if stop_moved:
                        stop_data.time_decay_adjustment = new_time_decay
                        stop_data.last_update = datetime.now()
                        updated_symbols.append(symbol)
                        
                        logger.info(f"⏰ Time decay stop update for {symbol}: "
                                   f"New decay={new_time_decay:.2f}, Stop=${stop_data.current_stop_price:.2f}")
            
            return updated_symbols
            
        except Exception as e:
            logger.error(f"Error updating time decay stops: {e}")
            return []
    
    def get_stop_status(self, symbol: str) -> Dict:
        """Get current stop status for a symbol"""
        if symbol not in self.active_stops:
            return {"status": "no_stop", "symbol": symbol}
        
        stop_data = self.active_stops[symbol]
        
        return {
            "status": "active",
            "symbol": symbol,
            "entry_price": stop_data.entry_price,
            "current_stop_price": stop_data.current_stop_price,
            "initial_stop_price": stop_data.initial_stop_price,
            "side": stop_data.side,
            "stop_type": stop_data.stop_type,
            "entry_time": stop_data.entry_time.isoformat(),
            "last_update": stop_data.last_update.isoformat(),
            "profit_high_water_mark": stop_data.profit_high_water_mark,
            "atr_at_entry": stop_data.atr_at_entry,
            "time_decay_adjustment": stop_data.time_decay_adjustment,
            "ml_adjustment_factor": stop_data.ml_adjustment_factor,
            "stop_distance_pct": abs(stop_data.current_stop_price - stop_data.entry_price) / stop_data.entry_price
        }
    
    def monitor_and_update_all_stops(self, current_positions: List[Dict]) -> Dict[str, str]:
        """
        Monitor and update all active stops
        Returns dict of symbol -> update_type
        """
        updates = {}
        
        try:
            # Update time decay for all stops
            time_decay_updates = self.update_time_decay_stops()
            for symbol in time_decay_updates:
                updates[symbol] = "time_decay"
            
            # Update trailing stops based on current prices
            for position in current_positions:
                symbol = position.get('symbol')
                current_price = float(position.get('current_price', 0))
                
                if symbol and current_price > 0:
                    trailing_update = self.update_trailing_stop(symbol, current_price)
                    if trailing_update and symbol not in updates:
                        updates[symbol] = "trailing"
            
            # Clean up stops for closed positions
            active_symbols = {pos.get('symbol') for pos in current_positions}
            closed_symbols = set(self.active_stops.keys()) - active_symbols
            
            for symbol in closed_symbols:
                del self.active_stops[symbol]
                updates[symbol] = "position_closed"
            
            logger.info(f"🔄 Stop monitoring complete: {len(updates)} updates")
            return updates
            
        except Exception as e:
            logger.error(f"Error monitoring stops: {e}")
            return {}

# Global instance
advanced_stop_optimizer = None

def initialize_advanced_stop_optimizer(api_client: tradeapi.REST):
    """Initialize the advanced stop optimizer"""
    global advanced_stop_optimizer
    advanced_stop_optimizer = AdvancedDynamicStopOptimizer(api_client)
    logger.info("✅ Advanced Dynamic Stop Optimizer initialized")

def create_optimized_stops_for_position(symbol: str, entry_price: float, 
                                       side: str, quantity: float, 
                                       confidence: float = 0.7) -> bool:
    """Create optimized stops for a position"""
    if advanced_stop_optimizer is None:
        logger.warning("⚠️ Advanced stop optimizer not initialized")
        return False
    
    stop_data = advanced_stop_optimizer.create_optimized_stop(
        symbol, entry_price, side, quantity, confidence)
    
    return stop_data is not None

def monitor_advanced_stops(current_positions: List[Dict] = None) -> Dict[str, str]:
    """Monitor and update all advanced stops"""
    if advanced_stop_optimizer is None:
        return {}
    
    return advanced_stop_optimizer.monitor_and_update_all_stops(current_positions or [])

def get_advanced_stop_status(symbol: str) -> Dict:
    """Get advanced stop status for a symbol"""
    if advanced_stop_optimizer is None:
        return {"status": "optimizer_not_available"}
    
    return advanced_stop_optimizer.get_stop_status(symbol)
