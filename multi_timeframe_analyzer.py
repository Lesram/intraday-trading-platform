#!/usr/bin/env python3
"""
🔄 MULTI-TIMEFRAME SIGNAL CONFIRMATION SYSTEM
Validates signals across multiple timeframes for higher accuracy
"""

import asyncio
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    def __init__(self, alpaca_client):
        self.alpaca_client = alpaca_client
        self.timeframes = ["1Min", "5Min", "15Min", "30Min"]
        
    async def get_multi_timeframe_confirmation(self, symbol: str, primary_signal: str) -> Dict:
        """Get signal confirmation across multiple timeframes"""
        confirmations = {}
        
        for timeframe in self.timeframes:
            try:
                bars = self.alpaca_client.get_market_data(symbol, timeframe=timeframe, limit=20)
                if not bars or len(bars) < 10:
                    continue
                    
                # Calculate trend direction for this timeframe
                prices = [float(bar.c) for bar in bars[-10:]]
                short_ma = sum(prices[-5:]) / 5  # 5-period MA
                long_ma = sum(prices[-10:]) / 10  # 10-period MA
                
                # Trend determination
                trend = "BULLISH" if short_ma > long_ma * 1.001 else "BEARISH" if short_ma < long_ma * 0.999 else "NEUTRAL"
                
                # Momentum calculation
                momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
                
                confirmations[timeframe] = {
                    "trend": trend,
                    "momentum": momentum,
                    "confirms_signal": (
                        (primary_signal == "BUY" and trend in ["BULLISH", "NEUTRAL"] and momentum > -0.01) or
                        (primary_signal == "SELL" and trend in ["BEARISH", "NEUTRAL"] and momentum < 0.01)
                    )
                }
                
            except Exception as e:
                logger.warning(f"Failed to get {timeframe} data for {symbol}: {e}")
                continue
        
        # Calculate confirmation score
        total_confirmations = len([tf for tf in confirmations.values() if tf["confirms_signal"]])
        confirmation_score = total_confirmations / len(confirmations) if confirmations else 0
        
        return {
            "confirmations": confirmations,
            "confirmation_score": confirmation_score,
            "is_confirmed": confirmation_score >= 0.6  # 60% of timeframes must confirm
        }

    async def calculate_regime_adjusted_confidence(self, symbol: str, base_confidence: float) -> float:
        """Adjust confidence based on market regime"""
        try:
            # Get SPY data for market regime detection
            spy_bars = self.alpaca_client.get_market_data("SPY", timeframe="15Min", limit=50)
            if not spy_bars or len(spy_bars) < 20:
                return base_confidence
            
            spy_prices = [float(bar.c) for bar in spy_bars]
            spy_volumes = [int(bar.v) for bar in spy_bars]
            
            # Volatility regime
            returns = [(spy_prices[i] / spy_prices[i-1] - 1) for i in range(1, len(spy_prices))]
            volatility = (sum(r**2 for r in returns[-20:]) / 20)**0.5
            
            # Volume regime
            avg_volume = sum(spy_volumes[-20:]) / 20
            current_volume = spy_volumes[-1]
            volume_ratio = current_volume / avg_volume
            
            # Trend regime
            short_term_trend = (spy_prices[-1] - spy_prices[-10]) / spy_prices[-10]
            medium_term_trend = (spy_prices[-1] - spy_prices[-20]) / spy_prices[-20]
            
            # Regime scoring
            regime_score = 1.0
            
            # High volatility reduces confidence
            if volatility > 0.02:  # > 2% intraday volatility
                regime_score *= 0.85
            
            # Low volume reduces confidence
            if volume_ratio < 0.8:
                regime_score *= 0.9
            
            # Conflicting trends reduce confidence
            if (short_term_trend > 0) != (medium_term_trend > 0):
                regime_score *= 0.9
            
            adjusted_confidence = base_confidence * regime_score
            
            logger.info(f"📈 Regime analysis for {symbol}: Vol={volatility:.4f}, "
                       f"Volume ratio={volume_ratio:.2f}, Regime score={regime_score:.3f}")
            
            return min(0.95, adjusted_confidence)
            
        except Exception as e:
            logger.error(f"Regime analysis failed for {symbol}: {e}")
            return base_confidence

# Global analyzer instance
multi_tf_analyzer = None

def initialize_multi_timeframe_analyzer(alpaca_client):
    global multi_tf_analyzer
    multi_tf_analyzer = MultiTimeframeAnalyzer(alpaca_client)

if __name__ == "__main__":
    print("🔄 Multi-Timeframe Confirmation System Ready")
    print("📊 Validates signals across 1Min, 5Min, 15Min, 30Min timeframes")
