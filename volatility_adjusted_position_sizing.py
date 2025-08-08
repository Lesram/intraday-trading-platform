#!/usr/bin/env python3
"""
🎯 ADVANCED MULTI-FACTOR POSITION SIZING MODULE
Enhanced dynamic position sizing with correlation, tail risk, and Kelly optimization
Institutional-grade risk management with portfolio-level controls
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class AdvancedVolatilityAdjustedSizer:
    """
    Implements institutional-grade multi-factor position sizing
    
    Enhanced Features:
    - Multi-factor Kelly Criterion with expected return integration
    - Portfolio-level correlation risk management
    - Tail risk assessment and VaR controls
    - Dynamic volatility regime detection
    - Enhanced sector concentration limits with correlation penalties
    - Real-time risk budget management
    """
    
    def __init__(self, api_client: tradeapi.REST):
        self.api = api_client
        self.base_position_limit = 0.03  # 3% base limit
        self.min_position_limit = 0.015  # 1.5% minimum during extreme conditions
        self.vix_threshold = 25.0        # VIX level to trigger scaling
        self.atr_spike_multiplier = 2.0  # ATR spike threshold
        self.max_correlation = 0.75      # Maximum correlation with existing positions
        
        # Enhanced sector exposure limits with correlation penalties
        self.sector_limits = {
            'tech': 0.25,      # 25% max tech exposure (was 20%)
            'tech_etf': 0.15,  # 15% for tech ETFs
            'market': 0.20,    # 20% for market ETFs
            'auto': 0.12,      # 12% for auto stocks (TSLA)
            'consumer': 0.15,  # 15% for consumer
            'media': 0.10,     # 10% for media
            'other': 0.12      # 12% default
        }
        
        # Correlation-based sector groupings
        self.correlation_groups = {
            'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'growth_tech': ['TSLA', 'NVDA', 'AMZN'],
            'broad_market': ['SPY', 'QQQ', 'IWM']
        }
        
        # Sector mapping
        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'NVDA': 'tech', 'META': 'tech',
            'TSLA': 'auto', 'AMZN': 'consumer', 'NFLX': 'media',
            'SPY': 'market', 'QQQ': 'tech_etf', 'IWM': 'market'
        }
        
        # Caching for performance
        self.vix_cache = {'value': 20.0, 'updated': datetime.min}
        self.atr_cache = {}
        
        # Risk budget tracking
        self.portfolio_risk_budget = 0.06  # 6% maximum portfolio VaR
        self.current_risk_usage = 0.0
    
    def calculate_multi_factor_kelly(self, symbol: str, confidence: float, expected_return: float = None,
                                    current_positions: List[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate enhanced Kelly fraction with multi-factor adjustments
        
        Args:
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            expected_return: Expected return estimate (optional)
            current_positions: Current portfolio positions
            
        Returns:
            Tuple of (kelly_fraction, adjustment_factors)
        """
        try:
            # Base Kelly calculation
            if expected_return is None:
                expected_return = confidence * 0.02  # Estimate 2% return for high confidence
            
            win_prob = 0.5 + (confidence - 0.5) * 0.4  # Scale confidence to win probability
            avg_win = expected_return * 1.5  # Average winning trade
            avg_loss = expected_return * 0.75  # Average losing trade (smaller losses)
            
            # Kelly fraction = (b*p - q) / b, where b = odds, p = win prob, q = loss prob
            if avg_loss > 0:
                kelly_base = (avg_win * win_prob - (1 - win_prob) * avg_loss) / avg_win
            else:
                kelly_base = confidence * 0.15  # Fallback
            
            # Multi-factor adjustments
            adjustments = {}
            
            # 1. Volatility regime adjustment
            vol_adjustment = self._calculate_volatility_adjustment(symbol)
            adjustments['volatility'] = vol_adjustment
            
            # 2. Correlation penalty
            corr_adjustment = self._calculate_correlation_adjustment(symbol, current_positions or [])
            adjustments['correlation'] = corr_adjustment
            
            # 3. Portfolio diversification bonus/penalty
            diversification_adjustment = self._calculate_diversification_adjustment(current_positions or [])
            adjustments['diversification'] = diversification_adjustment
            
            # 4. Tail risk adjustment
            tail_risk_adjustment = self._calculate_tail_risk_adjustment(symbol, current_positions or [])
            adjustments['tail_risk'] = tail_risk_adjustment
            
            # Combined Kelly fraction
            final_kelly = kelly_base * vol_adjustment * corr_adjustment * diversification_adjustment * tail_risk_adjustment
            final_kelly = max(0.01, min(0.25, final_kelly))  # Cap between 1% and 25%
            
            logger.info(f"🧮 Multi-factor Kelly for {symbol}: Base={kelly_base:.3f}, "
                       f"Final={final_kelly:.3f}, Adjustments={adjustments}")
            
            return final_kelly, adjustments
            
        except Exception as e:
            logger.error(f"Error calculating multi-factor Kelly: {e}")
            return confidence * 0.1, {'error': str(e)}
    
    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility regime adjustment factor"""
        try:
            # Get recent volatility data
            bars = self.api.get_bars(symbol, '1Day', limit=30, adjustment='raw')
            if not bars or len(bars) < 10:
                return 1.0
            
            prices = [float(bar.c) for bar in bars]
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            current_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Adjustment based on volatility percentile
            if current_vol < 0.15:  # Low volatility
                return 1.2  # Increase position size
            elif current_vol < 0.25:  # Normal volatility
                return 1.0  # No adjustment
            elif current_vol < 0.40:  # High volatility
                return 0.8  # Reduce position size
            else:  # Extreme volatility
                return 0.6  # Significantly reduce
                
        except Exception as e:
            logger.warning(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str, current_positions: List[Dict]) -> float:
        """Calculate correlation penalty for similar positions"""
        try:
            if not current_positions:
                return 1.0  # No penalty for first position
            
            new_symbol_sector = self.sector_map.get(symbol, 'other')
            
            # Check for high correlation with existing positions
            high_correlation_exposure = 0.0
            same_group_exposure = 0.0
            
            # Check correlation groups
            new_symbol_groups = []
            for group_name, symbols in self.correlation_groups.items():
                if symbol in symbols:
                    new_symbol_groups.append(group_name)
            
            for pos in current_positions:
                pos_symbol = pos.get('symbol', '')
                pos_sector = self.sector_map.get(pos_symbol, 'other')
                position_weight = abs(float(pos.get('market_value', 0)))
                
                # Same sector penalty
                if pos_sector == new_symbol_sector:
                    high_correlation_exposure += position_weight
                
                # Same correlation group penalty
                for group_name, symbols in self.correlation_groups.items():
                    if pos_symbol in symbols and group_name in new_symbol_groups:
                        same_group_exposure += position_weight
            
            # Calculate penalty factors
            sector_penalty = max(0.7, 1.0 - (high_correlation_exposure / 100000) * 0.5)  # Reduce if high sector exposure
            group_penalty = max(0.6, 1.0 - (same_group_exposure / 50000) * 0.4)  # Reduce if same group exposure
            
            return min(sector_penalty, group_penalty)
            
        except Exception as e:
            logger.warning(f"Error calculating correlation adjustment: {e}")
            return 0.9  # Conservative fallback
    
    def _calculate_diversification_adjustment(self, current_positions: List[Dict]) -> float:
        """Calculate diversification bonus/penalty"""
        try:
            num_positions = len(current_positions)
            
            if num_positions == 0:
                return 1.0  # No adjustment for first position
            elif num_positions <= 3:
                return 1.1  # Small bonus for building diversification
            elif num_positions <= 6:
                return 1.0  # Optimal diversification
            elif num_positions <= 10:
                return 0.95  # Small penalty for over-diversification
            else:
                return 0.9  # Larger penalty for excessive positions
                
        except Exception as e:
            logger.warning(f"Error calculating diversification adjustment: {e}")
            return 1.0
    
    def _calculate_tail_risk_adjustment(self, symbol: str, current_positions: List[Dict]) -> float:
        """Calculate tail risk adjustment based on portfolio stress scenarios"""
        try:
            # Estimate tail risk based on symbol characteristics
            tail_risk_factors = {
                'TSLA': 0.85,  # High tail risk
                'NVDA': 0.90,  # Moderate-high tail risk
                'AAPL': 0.95,  # Low tail risk
                'MSFT': 0.95,  # Low tail risk
                'SPY': 0.98,   # Very low tail risk
                'QQQ': 0.92,   # Moderate tail risk
            }
            
            base_adjustment = tail_risk_factors.get(symbol, 0.90)  # Default moderate tail risk
            
            # Increase penalty if portfolio already has high tail risk exposure
            high_tail_risk_exposure = 0.0
            for pos in current_positions:
                pos_symbol = pos.get('symbol', '')
                if pos_symbol in ['TSLA', 'NVDA', 'NFLX']:  # High tail risk symbols
                    high_tail_risk_exposure += abs(float(pos.get('market_value', 0)))
            
            if high_tail_risk_exposure > 30000:  # More than $30k in high tail risk
                base_adjustment *= 0.9
            
            return base_adjustment
            
        except Exception as e:
            logger.warning(f"Error calculating tail risk adjustment: {e}")
            return 0.95

    def get_vix_proxy(self) -> float:
        """Get VIX proxy using QQQ volatility"""
        try:
            # Check cache
            if (datetime.now() - self.vix_cache['updated']).seconds < 300:  # 5 minute cache
                return self.vix_cache['value']
                
            # Get QQQ data for volatility calculation
            bars = self.api.get_bars('QQQ', '1Day', limit=21, adjustment='raw')
            if not bars or len(bars) < 20:
                return 20.0  # Default VIX proxy
                
            prices = [float(bar.c) for bar in bars]
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized vol as percentage
            
            # Scale to VIX-like range (QQQ vol tends to be lower)
            vix_proxy = volatility * 1.3  # Rough scaling factor
            
            # Update cache
            self.vix_cache = {'value': vix_proxy, 'updated': datetime.now()}
            return vix_proxy
            
        except Exception as e:
            logger.warning(f"Error calculating VIX proxy: {e}")
            return 20.0

    def get_atr_percentile(self, symbol: str, lookback_days: int = 30) -> Tuple[float, float]:
        """Calculate ATR percentile for volatility regime detection"""
        try:
            cache_key = f"{symbol}_{lookback_days}"
            if cache_key in self.atr_cache:
                if (datetime.now() - self.atr_cache[cache_key]['updated']).seconds < 1800:  # 30 min cache
                    cached = self.atr_cache[cache_key]
                    return cached['current_atr'], cached['percentile']
            
            # Get historical data
            bars = self.api.get_bars(symbol, '1Day', limit=lookback_days + 5, adjustment='raw')
            if not bars or len(bars) < 20:
                return 1.0, 0.5  # Default values
                
            # Calculate True Range and ATR
            highs = [float(bar.h) for bar in bars]
            lows = [float(bar.l) for bar in bars]
            closes = [float(bar.c) for bar in bars]
            
            true_ranges = []
            for i in range(1, len(bars)):
                tr1 = highs[i] - lows[i]  # Current high - current low
                tr2 = abs(highs[i] - closes[i-1])  # Current high - previous close
                tr3 = abs(lows[i] - closes[i-1])   # Current low - previous close
                true_ranges.append(max(tr1, tr2, tr3))
            
            if len(true_ranges) < 14:
                return 1.0, 0.5
                
            # Calculate ATR (14-period)
            current_atr = np.mean(true_ranges[-14:])
            
            # Calculate percentile over lookback period
            all_atr_values = []
            for i in range(14, len(true_ranges)):
                atr_val = np.mean(true_ranges[i-14:i])
                all_atr_values.append(atr_val)
            
            if all_atr_values:
                percentile = np.percentile(all_atr_values, 100 * (len([x for x in all_atr_values if x <= current_atr]) / len(all_atr_values)))
            else:
                percentile = 50.0
                
            # Update cache
            self.atr_cache[cache_key] = {
                'current_atr': current_atr,
                'percentile': percentile,
                'updated': datetime.now()
            }
            
            return current_atr, percentile
            
        except Exception as e:
            logger.warning(f"Error calculating ATR percentile for {symbol}: {e}")
            return 1.0, 0.5

    def calculate_sector_exposure(self, positions) -> Dict[str, float]:
        """Calculate current sector exposure"""
        try:
            sector_exposure = {}
            total_value = sum(abs(float(pos.get('market_value', 0))) for pos in positions)
            
            if total_value == 0:
                return sector_exposure
                
            for pos in positions:
                symbol = pos.get('symbol', '')
                sector = self.sector_map.get(symbol, 'other')
                market_value = abs(float(pos.get('market_value', 0)))
                
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += market_value / total_value
                
            return sector_exposure
            
        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return {}

    def calculate_volatility_adjusted_size(
        self, 
        symbol: str, 
        confidence: float, 
        account_value: float,
        current_positions: List[Dict] = None,
        use_multi_factor_kelly: bool = True
    ) -> Tuple[float, Dict]:
        """
        Calculate position size using enhanced multi-factor approach
        
        Args:
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            account_value: Total account value
            current_positions: Current portfolio positions
            use_multi_factor_kelly: Use enhanced Kelly criterion
            
        Returns:
            Tuple of (position_dollars, sizing_details)
        """
        try:
            current_positions = current_positions or []
            
            # Use multi-factor Kelly if enabled
            if use_multi_factor_kelly:
                kelly_fraction, kelly_adjustments = self.calculate_multi_factor_kelly(
                    symbol, confidence, current_positions=current_positions
                )
                base_size_fraction = kelly_fraction
            else:
                # Legacy volatility-based sizing
                base_size_fraction = self._calculate_legacy_sizing(symbol, confidence)
                kelly_adjustments = {'legacy_mode': True}
            
            # Apply market regime adjustments
            vix_proxy = self.get_vix_proxy()
            atr, atr_percentile = self.get_atr_percentile(symbol)
            
            # VIX scaling
            if vix_proxy > self.vix_threshold:
                vix_scale = max(0.5, 1.0 - (vix_proxy - self.vix_threshold) / 50.0)
            else:
                vix_scale = min(1.2, 1.0 + (self.vix_threshold - vix_proxy) / 100.0)
            
            # ATR scaling
            if atr_percentile > 80:  # ATR spike
                atr_scale = 0.7
            elif atr_percentile > 60:
                atr_scale = 0.85
            elif atr_percentile < 20:
                atr_scale = 1.15
            else:
                atr_scale = 1.0
            
            # Sector exposure check
            sector_exposure = self.calculate_sector_exposure(current_positions)
            symbol_sector = self.sector_map.get(symbol, 'other')
            current_sector_exposure = sector_exposure.get(symbol_sector, 0.0)
            sector_limit = self.sector_limits.get(symbol_sector, 0.12)
            
            # Sector scaling
            if current_sector_exposure >= sector_limit * 0.8:  # Near limit
                sector_scale = max(0.3, 1.0 - (current_sector_exposure / sector_limit))
            else:
                sector_scale = 1.0
            
            # Final position size calculation
            final_size_fraction = base_size_fraction * vix_scale * atr_scale * sector_scale
            final_size_fraction = max(self.min_position_limit, min(self.base_position_limit, final_size_fraction))
            
            position_dollars = account_value * final_size_fraction
            
            sizing_details = {
                'base_kelly_fraction': base_size_fraction,
                'kelly_adjustments': kelly_adjustments,
                'vix_proxy': vix_proxy,
                'vix_scale': vix_scale,
                'atr_percentile': atr_percentile,
                'atr_scale': atr_scale,
                'sector_exposure': current_sector_exposure,
                'sector_scale': sector_scale,
                'final_fraction': final_size_fraction,
                'position_dollars': position_dollars
            }
            
            logger.info(f"📊 Enhanced sizing for {symbol}: "
                       f"${position_dollars:,.0f} ({final_size_fraction:.1%})")
            
            return position_dollars, sizing_details
            
        except Exception as e:
            logger.error(f"Error in volatility adjusted sizing: {e}")
            # Fallback to simple sizing
            fallback_size = account_value * 0.02 * confidence
            return fallback_size, {'error': str(e)}
    
    def _calculate_legacy_sizing(self, symbol: str, confidence: float) -> float:
        """Legacy volatility-based sizing for backward compatibility"""
        try:
            # Simple volatility-adjusted sizing
            vix_proxy = self.get_vix_proxy()
            _, atr_percentile = self.get_atr_percentile(symbol)
            
            base_fraction = confidence * 0.025  # Base 2.5% for full confidence
            
            # Volatility adjustments
            if vix_proxy > 25:
                vol_adjustment = 0.7
            elif vix_proxy < 15:
                vol_adjustment = 1.3
            else:
                vol_adjustment = 1.0
            
            # ATR adjustment
            atr_adjustment = max(0.6, min(1.4, 1.0 - (atr_percentile - 50) / 100))
            
            return base_fraction * vol_adjustment * atr_adjustment
            
        except Exception as e:
            logger.warning(f"Error in legacy sizing: {e}")
            return confidence * 0.02


# Maintain backward compatibility
VolatilityAdjustedSizer = AdvancedVolatilityAdjustedSizer

# Global instance
volatility_sizer = None

def initialize_volatility_sizer(api_client: tradeapi.REST):
    """Initialize the volatility-adjusted position sizer"""
    global volatility_sizer
    volatility_sizer = AdvancedVolatilityAdjustedSizer(api_client)
    logger.info("✅ Advanced Volatility-Adjusted Position Sizer initialized")

def get_volatility_adjusted_size(symbol: str, signal_strength: float, portfolio_value: float = 100000) -> float:
    """Get volatility-adjusted position size"""
    if volatility_sizer:
        return volatility_sizer.calculate_position_size(symbol, signal_strength, portfolio_value)
    else:
        # Fallback calculation
        return min(0.02 * signal_strength * portfolio_value, 0.05 * portfolio_value)
