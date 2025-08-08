#!/usr/bin/env python3
"""
🛡️ PORTFOLIO-LEVEL RISK MANAGER
Manages aggregate portfolio risk across all positions
Prevents risk accumulation beyond limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioRiskManager:
    def __init__(self, max_portfolio_risk=0.06, max_correlation=0.7):
        """
        Initialize portfolio risk manager
        
        Args:
            max_portfolio_risk: Maximum portfolio Value-at-Risk (default 6%)
            max_correlation: Maximum correlation between positions
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.position_risks = {}
        self.correlation_matrix = {}
        
    def calculate_portfolio_var(self, positions: List[Dict], market_data: Dict) -> float:
        """Calculate portfolio Value-at-Risk using correlations"""
        try:
            if not positions or len(positions) < 2:
                # Single position or no positions - use individual risk
                if positions:
                    return abs(float(positions[0].get('unrealized_pnl_percent', 0))) / 100
                return 0.0
            
            # Calculate individual position risks
            position_risks = []
            symbols = []
            
            for pos in positions:
                symbol = pos['symbol']
                symbols.append(symbol)
                
                # Risk as percentage of portfolio
                portfolio_weight = abs(float(pos.get('market_value', 0))) / sum(abs(float(p.get('market_value', 0))) for p in positions)
                volatility = market_data.get(symbol, {}).get('volatility', 0.02)  # Default 2% daily vol
                
                position_risk = portfolio_weight * volatility
                position_risks.append(position_risk)
                
                logger.info(f"📊 {symbol}: Weight={portfolio_weight:.1%}, Vol={volatility:.1%}, Risk={position_risk:.1%}")
            
            # Calculate portfolio VaR with correlations
            portfolio_var = self._calculate_diversified_var(position_risks, symbols)
            
            logger.info(f"🎯 Portfolio VaR: {portfolio_var:.2%} (Limit: {self.max_portfolio_risk:.1%})")
            return portfolio_var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            # Fallback: Sum individual risks (conservative)
            total_risk = sum(abs(float(pos.get('unrealized_pnl_percent', 0))) for pos in positions) / 100
            return min(total_risk, 0.20)  # Cap at 20%
    
    def _calculate_diversified_var(self, position_risks: List[float], symbols: List[str]) -> float:
        """Calculate diversified VaR using correlation estimates"""
        try:
            n_positions = len(position_risks)
            if n_positions == 1:
                return position_risks[0]
            
            # Estimate correlations (simplified sector-based approach)
            correlations = self._estimate_correlations(symbols)
            
            # Calculate portfolio variance
            portfolio_variance = 0.0
            
            for i in range(n_positions):
                for j in range(n_positions):
                    correlation = correlations.get(f"{symbols[i]}-{symbols[j]}", 0.3)  # Default moderate correlation
                    portfolio_variance += position_risks[i] * position_risks[j] * correlation
            
            # VaR is square root of variance (assuming normal distribution)
            portfolio_var = np.sqrt(max(portfolio_variance, 0))
            
            return min(portfolio_var, sum(position_risks))  # Cap at sum of individual risks
            
        except Exception as e:
            logger.error(f"Error in diversified VaR calculation: {e}")
            return sum(position_risks) * 0.8  # Conservative estimate with some diversification
    
    def _estimate_correlations(self, symbols: List[str]) -> Dict[str, float]:
        """Estimate correlations between symbols based on sector classification"""
        correlations = {}
        
        # Sector classifications
        sector_map = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN'],
            'etf': ['SPY', 'QQQ', 'IWM'],
            'auto': ['TSLA'],
            'media': ['NFLX']
        }
        
        # Create reverse mapping
        symbol_to_sector = {}
        for sector, sector_symbols in sector_map.items():
            for symbol in sector_symbols:
                symbol_to_sector[symbol] = sector
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    correlations[f"{sym1}-{sym2}"] = 1.0
                else:
                    sector1 = symbol_to_sector.get(sym1, 'other')
                    sector2 = symbol_to_sector.get(sym2, 'other')
                    
                    if sector1 == sector2:
                        # Same sector - high correlation
                        if sector1 == 'tech':
                            correlation = 0.75
                        elif sector1 == 'etf':
                            correlation = 0.85
                        else:
                            correlation = 0.60
                    else:
                        # Different sectors - moderate correlation
                        correlation = 0.25
                    
                    correlations[f"{sym1}-{sym2}"] = correlation
                    correlations[f"{sym2}-{sym1}"] = correlation
        
        return correlations
    
    def check_correlation_risk(self, positions: List[Dict]) -> Dict:
        """Check for high correlation risk in portfolio"""
        try:
            if len(positions) < 2:
                return {"high_correlation": False, "max_correlation": 0.0, "correlated_pairs": []}
            
            symbols = [pos['symbol'] for pos in positions]
            correlations = self._estimate_correlations(symbols)
            
            # Find highest correlations
            high_corr_pairs = []
            max_correlation = 0.0
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    corr = correlations.get(f"{sym1}-{sym2}", 0.0)
                    if corr > self.max_correlation:
                        high_corr_pairs.append({
                            "pair": f"{sym1}-{sym2}",
                            "correlation": corr,
                            "risk_level": "high" if corr > 0.8 else "moderate"
                        })
                    max_correlation = max(max_correlation, corr)
            
            return {
                "high_correlation": len(high_corr_pairs) > 0,
                "max_correlation": max_correlation,
                "correlated_pairs": high_corr_pairs
            }
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return {"high_correlation": False, "max_correlation": 0.0, "correlated_pairs": []}
    
    def calculate_position_size_adjustment(self, symbol: str, base_size: float, current_positions: List[Dict], 
                                          current_var: float) -> Tuple[float, str]:
        """Calculate position size adjustment based on portfolio risk"""
        try:
            # Check if adding this position would exceed portfolio VaR
            estimated_position_risk = base_size * 0.02  # Estimate 2% risk per position
            projected_var = current_var + estimated_position_risk
            
            if projected_var <= self.max_portfolio_risk:
                return base_size, "approved"
            
            # Calculate maximum allowable size
            remaining_risk_budget = self.max_portfolio_risk - current_var
            if remaining_risk_budget <= 0:
                return 0, "portfolio_risk_limit_exceeded"
            
            # Scale down position size
            risk_ratio = remaining_risk_budget / estimated_position_risk
            adjusted_size = base_size * risk_ratio
            
            logger.warning(f"🚨 Position size reduced for {symbol}: {base_size} → {adjusted_size:.0f} "
                          f"(Portfolio VaR: {current_var:.2%} → {projected_var:.2%})")
            
            return max(0, adjusted_size), "size_reduced_portfolio_risk"
            
        except Exception as e:
            logger.error(f"Error calculating position size adjustment: {e}")
            return base_size * 0.5, "error_fallback"
    
    def get_portfolio_risk_summary(self, positions: List[Dict], market_data: Dict) -> Dict:
        """Get comprehensive portfolio risk summary"""
        try:
            portfolio_var = self.calculate_portfolio_var(positions, market_data)
            correlation_risk = self.check_correlation_risk(positions)
            
            # Calculate sector concentrations
            sector_concentrations = {}
            total_value = sum(abs(float(pos.get('market_value', 0))) for pos in positions)
            
            sector_map = {
                'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'NVDA': 'tech', 'META': 'tech',
                'TSLA': 'auto', 'AMZN': 'consumer', 'NFLX': 'media',
                'SPY': 'market', 'QQQ': 'tech_etf'
            }
            
            for pos in positions:
                sector = sector_map.get(pos['symbol'], 'other')
                value = abs(float(pos.get('market_value', 0)))
                sector_concentrations[sector] = sector_concentrations.get(sector, 0) + value
            
            # Convert to percentages
            for sector in sector_concentrations:
                if total_value > 0:
                    sector_concentrations[sector] = (sector_concentrations[sector] / total_value) * 100
            
            # Risk level assessment
            risk_level = "low"
            if portfolio_var > self.max_portfolio_risk * 0.8:
                risk_level = "high"
            elif portfolio_var > self.max_portfolio_risk * 0.6:
                risk_level = "medium"
            
            return {
                "portfolio_var": portfolio_var,
                "var_limit": self.max_portfolio_risk,
                "risk_level": risk_level,
                "correlation_risk": correlation_risk,
                "sector_concentrations": sector_concentrations,
                "total_positions": len(positions),
                "utilization_pct": (portfolio_var / self.max_portfolio_risk) * 100,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio risk summary: {e}")
            return {
                "portfolio_var": 0.0,
                "var_limit": self.max_portfolio_risk,
                "risk_level": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global portfolio risk manager instance
portfolio_risk_manager = None

def initialize_portfolio_risk_manager(max_portfolio_risk=0.06):
    """Initialize the global portfolio risk manager"""
    global portfolio_risk_manager
    portfolio_risk_manager = PortfolioRiskManager(max_portfolio_risk=max_portfolio_risk)
    logger.info(f"🛡️ Portfolio risk manager initialized (Max VaR: {max_portfolio_risk:.1%})")

if __name__ == "__main__":
    print("🛡️ Portfolio Risk Management System")
    print("Prevents risk accumulation beyond portfolio limits")
