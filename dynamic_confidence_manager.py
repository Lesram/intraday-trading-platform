#!/usr/bin/env python3
"""
📈 PHASE 2 OPTIMIZATION: DYNAMIC CONFIDENCE THRESHOLDS & VIX INTEGRATION
Part of Audit Item 4: Trading Strategy Reevaluation - Phase 2 Implementation
Market-Adaptive Trading with Volatility-Based Confidence Adjustments
"""

import sys
import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Phase 1 systems
from unified_risk_manager import get_risk_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications based on volatility"""
    VERY_LOW_VOL = "very_low_volatility"    # VIX < 15
    LOW_VOL = "low_volatility"              # VIX 15-20
    NORMAL_VOL = "normal_volatility"        # VIX 20-30
    HIGH_VOL = "high_volatility"            # VIX 30-40
    EXTREME_VOL = "extreme_volatility"      # VIX > 40

@dataclass
class MarketConditions:
    """Current market conditions for adaptive trading"""
    vix: float
    spx_change_1d: float
    spx_change_5d: float
    market_regime: MarketRegime
    trend_strength: float
    correlation_breakdown: bool
    fear_greed_index: Optional[float] = None

@dataclass
class DynamicThresholds:
    """Dynamic confidence thresholds based on market conditions"""
    base_confidence_threshold: float
    adjusted_confidence_threshold: float
    position_size_multiplier: float
    stop_loss_adjustment: float
    time_decay_factor: float
    regime_specific_params: Dict[str, float]

class VixDataProvider:
    """Provides VIX and market volatility data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_current_vix(self) -> float:
        """Get current VIX value (with caching)"""
        
        now = datetime.now()
        
        # Check cache first
        if 'vix' in self.cache and 'vix' in self.cache_expiry:
            if now < self.cache_expiry['vix']:
                return self.cache['vix']
        
        try:
            # In production, this would connect to a real data provider
            # For now, we'll simulate VIX based on time of day and some randomness
            vix_value = self._simulate_vix()
            
            # Cache the result
            self.cache['vix'] = vix_value
            self.cache_expiry['vix'] = now + timedelta(seconds=self.cache_duration)
            
            logger.info(f"📊 Current VIX: {vix_value:.2f}")
            return vix_value
            
        except Exception as e:
            logger.warning(f"⚠️ VIX data unavailable, using default: {str(e)}")
            return 20.0  # Default VIX
    
    def _simulate_vix(self) -> float:
        """Simulate VIX based on market hours and random factors"""
        
        # Base VIX around normal levels
        base_vix = 22.0
        
        # Time-based adjustments (higher volatility around open/close)
        hour = datetime.now().hour
        if 9 <= hour <= 10 or 15 <= hour <= 16:  # Market open/close
            time_adjustment = np.random.normal(3.0, 2.0)
        elif 11 <= hour <= 14:  # Mid-day calm
            time_adjustment = np.random.normal(-2.0, 1.5)
        else:  # After hours
            time_adjustment = np.random.normal(0.0, 1.0)
        
        # Random market stress factor
        stress_factor = np.random.normal(0.0, 4.0)
        
        # Calculate final VIX
        simulated_vix = base_vix + time_adjustment + stress_factor
        
        # Keep within reasonable bounds
        return max(10.0, min(80.0, simulated_vix))
    
    async def get_market_conditions(self) -> MarketConditions:
        """Get comprehensive market conditions"""
        
        vix = await self.get_current_vix()
        
        # Simulate other market metrics
        spx_1d = np.random.normal(0.002, 0.015)  # Daily S&P change
        spx_5d = np.random.normal(0.01, 0.035)   # 5-day S&P change
        
        # Determine market regime
        if vix < 15:
            regime = MarketRegime.VERY_LOW_VOL
        elif vix < 20:
            regime = MarketRegime.LOW_VOL
        elif vix < 30:
            regime = MarketRegime.NORMAL_VOL
        elif vix < 40:
            regime = MarketRegime.HIGH_VOL
        else:
            regime = MarketRegime.EXTREME_VOL
        
        # Calculate trend strength (0-1)
        trend_strength = min(1.0, abs(spx_5d) / 0.05)
        
        # Detect correlation breakdown (high VIX + mixed signals)
        correlation_breakdown = vix > 25 and abs(spx_1d - spx_5d) > 0.02
        
        conditions = MarketConditions(
            vix=vix,
            spx_change_1d=spx_1d,
            spx_change_5d=spx_5d,
            market_regime=regime,
            trend_strength=trend_strength,
            correlation_breakdown=correlation_breakdown,
            fear_greed_index=None  # Would come from CNN Fear & Greed Index
        )
        
        logger.info(f"🌍 Market Regime: {regime.value}, VIX: {vix:.1f}, Trend: {trend_strength:.2f}")
        return conditions

class DynamicConfidenceManager:
    """Manages dynamic confidence thresholds based on market conditions"""
    
    def __init__(self):
        self.vix_provider = VixDataProvider()
        self.risk_manager = get_risk_manager()
        
        # Base confidence thresholds by strategy
        self.base_thresholds = {
            "automated_signal_trading": 0.65,    # Require 65% confidence
            "momentum_strategy": 0.60,           # 60% for momentum
            "mean_reversion_strategy": 0.55,     # 55% for mean reversion  
            "portfolio_rebalancing": 0.45        # 45% for rebalancing
        }
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            MarketRegime.VERY_LOW_VOL: {
                "confidence_multiplier": 0.9,    # Lower bar in calm markets
                "position_size_multiplier": 1.2, # Larger positions
                "stop_loss_adjustment": 0.8      # Tighter stops
            },
            MarketRegime.LOW_VOL: {
                "confidence_multiplier": 0.95,
                "position_size_multiplier": 1.1,
                "stop_loss_adjustment": 0.9
            },
            MarketRegime.NORMAL_VOL: {
                "confidence_multiplier": 1.0,    # Standard thresholds
                "position_size_multiplier": 1.0,
                "stop_loss_adjustment": 1.0
            },
            MarketRegime.HIGH_VOL: {
                "confidence_multiplier": 1.15,   # Higher bar in volatile markets
                "position_size_multiplier": 0.8, # Smaller positions
                "stop_loss_adjustment": 1.3     # Wider stops
            },
            MarketRegime.EXTREME_VOL: {
                "confidence_multiplier": 1.4,    # Much higher bar
                "position_size_multiplier": 0.5, # Much smaller positions
                "stop_loss_adjustment": 1.5     # Much wider stops
            }
        }
        
        logger.info("📈 Dynamic Confidence Manager initialized")
    
    async def calculate_dynamic_thresholds(self, strategy: str) -> DynamicThresholds:
        """Calculate dynamic confidence thresholds for a strategy"""
        
        # Get current market conditions
        conditions = await self.vix_provider.get_market_conditions()
        
        # Get base threshold for strategy
        base_threshold = self.base_thresholds.get(strategy, 0.60)
        
        # Get regime adjustments
        regime_params = self.regime_adjustments[conditions.market_regime]
        
        # Calculate adjusted threshold
        adjusted_threshold = base_threshold * regime_params["confidence_multiplier"]
        
        # Additional adjustments based on market conditions
        trend_adjustment = 1.0
        if conditions.trend_strength > 0.7:  # Strong trend
            if strategy in ["momentum_strategy"]:
                trend_adjustment = 0.95  # Easier for momentum in trends
            elif strategy in ["mean_reversion_strategy"]:
                trend_adjustment = 1.1   # Harder for mean reversion in trends
        
        # Correlation breakdown adjustment
        correlation_adjustment = 1.0
        if conditions.correlation_breakdown:
            correlation_adjustment = 1.2  # Higher bar when correlations break down
        
        # Final adjusted threshold
        final_threshold = adjusted_threshold * trend_adjustment * correlation_adjustment
        final_threshold = max(0.3, min(0.95, final_threshold))  # Reasonable bounds
        
        # Time decay factor (reduce confidence over time without new signals)
        time_decay = 0.95  # 5% decay per period without updates
        
        thresholds = DynamicThresholds(
            base_confidence_threshold=base_threshold,
            adjusted_confidence_threshold=final_threshold,
            position_size_multiplier=regime_params["position_size_multiplier"],
            stop_loss_adjustment=regime_params["stop_loss_adjustment"],
            time_decay_factor=time_decay,
            regime_specific_params=regime_params
        )
        
        logger.info(f"🎯 {strategy} thresholds: {base_threshold:.2f} → {final_threshold:.2f} "
                   f"(regime: {conditions.market_regime.value})")
        
        return thresholds
    
    async def should_execute_trade(self, strategy: str, signal_confidence: float, 
                                 symbol: str, additional_context: Dict = None) -> Tuple[bool, str, float]:
        """Determine if a trade should be executed based on dynamic thresholds"""
        
        # Get dynamic thresholds
        thresholds = await self.calculate_dynamic_thresholds(strategy)
        
        # Check basic confidence threshold
        if signal_confidence < thresholds.adjusted_confidence_threshold:
            reason = f"Confidence {signal_confidence:.2f} below dynamic threshold {thresholds.adjusted_confidence_threshold:.2f}"
            return False, reason, 0.0
        
        # Get current market conditions for additional checks
        conditions = await self.vix_provider.get_market_conditions()
        
        # Strategy-specific regime filters
        if strategy == "momentum_strategy" and conditions.market_regime == MarketRegime.EXTREME_VOL:
            if signal_confidence < 0.8:  # Extra high bar for momentum in extreme volatility
                return False, "Momentum strategy requires 80%+ confidence in extreme volatility", 0.0
        
        if strategy == "mean_reversion_strategy" and conditions.trend_strength > 0.8:
            if signal_confidence < 0.7:  # Mean reversion harder in strong trends
                return False, "Mean reversion requires 70%+ confidence in strong trends", 0.0
        
        # Calculate adjusted position size
        base_size = 1.0  # Would come from normal position sizing
        adjusted_size = base_size * thresholds.position_size_multiplier
        
        # Additional size adjustments based on confidence level
        confidence_bonus = min(1.5, signal_confidence / thresholds.base_confidence_threshold)
        final_size = adjusted_size * confidence_bonus
        
        logger.info(f"✅ Trade approved for {symbol} ({strategy}): "
                   f"Confidence {signal_confidence:.2f}, Size multiplier {final_size:.2f}")
        
        return True, "Trade approved with dynamic adjustments", final_size
    
    async def get_market_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of current market adaptations"""
        
        conditions = await self.vix_provider.get_market_conditions()
        
        # Calculate thresholds for all strategies
        strategy_thresholds = {}
        for strategy in self.base_thresholds.keys():
            thresholds = await self.calculate_dynamic_thresholds(strategy)
            strategy_thresholds[strategy] = {
                "base_threshold": thresholds.base_confidence_threshold,
                "adjusted_threshold": thresholds.adjusted_confidence_threshold,
                "position_multiplier": thresholds.position_size_multiplier,
                "stop_adjustment": thresholds.stop_loss_adjustment
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_conditions": {
                "vix": conditions.vix,
                "regime": conditions.market_regime.value,
                "spx_1d_change": f"{conditions.spx_change_1d:.2%}",
                "spx_5d_change": f"{conditions.spx_change_5d:.2%}",
                "trend_strength": conditions.trend_strength,
                "correlation_breakdown": conditions.correlation_breakdown
            },
            "strategy_adaptations": strategy_thresholds,
            "market_impact": {
                "trading_difficulty": self._assess_trading_difficulty(conditions),
                "recommended_exposure": self._recommend_exposure_level(conditions),
                "primary_risks": self._identify_primary_risks(conditions)
            }
        }
    
    def _assess_trading_difficulty(self, conditions: MarketConditions) -> str:
        """Assess overall trading difficulty"""
        
        if conditions.market_regime in [MarketRegime.EXTREME_VOL]:
            return "VERY HIGH"
        elif conditions.market_regime == MarketRegime.HIGH_VOL:
            return "HIGH"
        elif conditions.correlation_breakdown:
            return "ELEVATED"
        elif conditions.market_regime == MarketRegime.VERY_LOW_VOL:
            return "LOW"
        else:
            return "NORMAL"
    
    def _recommend_exposure_level(self, conditions: MarketConditions) -> str:
        """Recommend portfolio exposure level"""
        
        if conditions.market_regime == MarketRegime.EXTREME_VOL:
            return "DEFENSIVE (25-50%)"
        elif conditions.market_regime == MarketRegime.HIGH_VOL:
            return "CAUTIOUS (50-75%)"
        elif conditions.correlation_breakdown:
            return "REDUCED (60-80%)"
        elif conditions.market_regime == MarketRegime.VERY_LOW_VOL:
            return "AGGRESSIVE (90-120%)"
        else:
            return "NORMAL (75-100%)"
    
    def _identify_primary_risks(self, conditions: MarketConditions) -> List[str]:
        """Identify primary market risks"""
        
        risks = []
        
        if conditions.vix > 30:
            risks.append("High volatility regime")
        
        if conditions.correlation_breakdown:
            risks.append("Asset correlation breakdown")
        
        if abs(conditions.spx_change_5d) > 0.05:
            risks.append("High trend momentum")
        
        if conditions.trend_strength < 0.3:
            risks.append("Choppy, directionless market")
        
        if not risks:
            risks.append("Normal market conditions")
        
        return risks

# Global dynamic confidence manager
dynamic_manager = DynamicConfidenceManager()

def get_dynamic_confidence_manager() -> DynamicConfidenceManager:
    """Get the global dynamic confidence manager"""
    return dynamic_manager

# Integration classes for existing strategies
class AdaptiveSignalTrading:
    """Automated signal trading with dynamic confidence thresholds"""
    
    @staticmethod
    async def should_trade_signal(symbol: str, base_confidence: float, 
                                signal_strength: float) -> Tuple[bool, float]:
        """Determine if signal should be traded with dynamic thresholds"""
        
        # Enhanced confidence calculation
        enhanced_confidence = base_confidence * (1 + signal_strength * 0.2)
        enhanced_confidence = min(0.95, enhanced_confidence)
        
        # Check dynamic thresholds
        approved, reason, size_multiplier = await dynamic_manager.should_execute_trade(
            "automated_signal_trading", enhanced_confidence, symbol
        )
        
        if approved:
            logger.info(f"📊 Signal trade approved: {symbol} confidence {enhanced_confidence:.2f}")
        else:
            logger.info(f"🚫 Signal trade rejected: {symbol} - {reason}")
        
        return approved, size_multiplier

class AdaptiveMomentumTrading:
    """Momentum trading with volatility-based adaptations"""
    
    @staticmethod
    async def should_trade_momentum(symbol: str, momentum_score: float, 
                                  volume_confirmation: float) -> Tuple[bool, float]:
        """Determine if momentum trade should be executed"""
        
        # Combine momentum and volume for confidence
        confidence = (momentum_score * 0.7) + (volume_confirmation * 0.3)
        
        approved, reason, size_multiplier = await dynamic_manager.should_execute_trade(
            "momentum_strategy", confidence, symbol
        )
        
        if approved:
            logger.info(f"🚀 Momentum trade approved: {symbol} momentum {momentum_score:.2f}")
        else:
            logger.info(f"🚫 Momentum trade rejected: {symbol} - {reason}")
        
        return approved, size_multiplier

class AdaptiveMeanReversion:
    """Mean reversion with trend-aware adaptations"""
    
    @staticmethod
    async def should_trade_reversion(symbol: str, reversion_probability: float,
                                   support_level_strength: float) -> Tuple[bool, float]:
        """Determine if mean reversion trade should be executed"""
        
        # Weight reversion probability and technical support
        confidence = (reversion_probability * 0.6) + (support_level_strength * 0.4)
        
        approved, reason, size_multiplier = await dynamic_manager.should_execute_trade(
            "mean_reversion_strategy", confidence, symbol
        )
        
        if approved:
            logger.info(f"↩️  Reversion trade approved: {symbol} probability {reversion_probability:.2f}")
        else:
            logger.info(f"🚫 Reversion trade rejected: {symbol} - {reason}")
        
        return approved, size_multiplier

if __name__ == "__main__":
    # Test the dynamic confidence system
    logger.info("🧪 Testing Dynamic Confidence Threshold System...")
    
    async def test_dynamic_system():
        print("📈 DYNAMIC CONFIDENCE THRESHOLD TEST")
        print("=" * 50)
        
        # Test market conditions
        conditions = await dynamic_manager.vix_provider.get_market_conditions()
        print(f"Market Regime: {conditions.market_regime.value}")
        print(f"VIX: {conditions.vix:.1f}")
        print(f"Trend Strength: {conditions.trend_strength:.2f}")
        
        # Test dynamic thresholds for each strategy
        strategies = ["automated_signal_trading", "momentum_strategy", "mean_reversion_strategy", "portfolio_rebalancing"]
        
        print(f"\n🎯 Dynamic Thresholds by Strategy:")
        for strategy in strategies:
            thresholds = await dynamic_manager.calculate_dynamic_thresholds(strategy)
            print(f"  {strategy}:")
            print(f"    Base: {thresholds.base_confidence_threshold:.2f}")
            print(f"    Adjusted: {thresholds.adjusted_confidence_threshold:.2f}")
            print(f"    Position Multiplier: {thresholds.position_size_multiplier:.2f}")
        
        # Test trade decisions
        print(f"\n📊 Trade Decision Tests:")
        
        test_signals = [
            ("AAPL", "automated_signal_trading", 0.72, "Strong signal"),
            ("TSLA", "momentum_strategy", 0.68, "Moderate momentum"),
            ("MSFT", "mean_reversion_strategy", 0.58, "Weak reversion"),
            ("SPY", "portfolio_rebalancing", 0.50, "Rebalancing need")
        ]
        
        for symbol, strategy, confidence, description in test_signals:
            approved, reason, size = await dynamic_manager.should_execute_trade(
                strategy, confidence, symbol
            )
            
            status = "✅ APPROVED" if approved else "🚫 REJECTED"
            print(f"  {symbol} ({description}): {status}")
            print(f"    Confidence: {confidence:.2f}, Size: {size:.2f}, Reason: {reason}")
        
        # Test adaptation summary
        print(f"\n🌍 Market Adaptation Summary:")
        summary = await dynamic_manager.get_market_adaptation_summary()
        
        print(f"  Trading Difficulty: {summary['market_impact']['trading_difficulty']}")
        print(f"  Recommended Exposure: {summary['market_impact']['recommended_exposure']}")
        print(f"  Primary Risks: {', '.join(summary['market_impact']['primary_risks'])}")
        
        print("\n✅ Dynamic Confidence System test completed!")
    
    # Run the test
    asyncio.run(test_dynamic_system())
