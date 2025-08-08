#!/usr/bin/env python3
"""
📊 PERFORMANCE ATTRIBUTION ANALYSIS MODULE
Factor-based performance attribution, strategy P&L breakdown, and risk-adjusted metrics
Priority 2A implementation for institutional-grade performance analysis
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class PerformanceFactors(Enum):
    """Performance attribution factors"""
    MARKET_BETA = "market_beta"
    SECTOR_ALLOCATION = "sector_allocation" 
    SECURITY_SELECTION = "security_selection"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_TIMING = "volatility_timing"
    TRANSACTION_COSTS = "transaction_costs"
    CURRENCY = "currency"  # For future international expansion

@dataclass
class StrategyPerformance:
    """Performance metrics for individual strategies"""
    strategy_name: str
    total_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    avg_trade_pnl: float
    risk_adjusted_return: float
    volatility: float
    information_ratio: float
    calmar_ratio: float

@dataclass
class FactorAttribution:
    """Factor-based performance attribution"""
    factor: PerformanceFactors
    contribution_pnl: float
    contribution_pct: float
    factor_exposure: float
    factor_return: float
    active_return: float
    tracking_error: float

@dataclass 
class PerformanceBreakdown:
    """Comprehensive performance breakdown"""
    timestamp: datetime
    total_pnl: float
    strategy_performance: List[StrategyPerformance]
    factor_attribution: List[FactorAttribution]
    risk_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    cost_analysis: Dict[str, float]

class PerformanceAttributionAnalyzer:
    """
    Advanced performance attribution analysis system providing:
    - Factor-based performance decomposition
    - Strategy-level P&L breakdown
    - Risk-adjusted performance metrics (Information Ratio, Calmar Ratio)
    - Benchmark comparison analysis
    - Transaction cost impact analysis
    """
    
    def __init__(self, api_client: tradeapi.REST, benchmark_symbol: str = "SPY"):
        self.api = api_client
        self.benchmark_symbol = benchmark_symbol
        self.trade_history: List[Dict] = []
        self.performance_cache = {}
        self.benchmark_cache = {}
        self.factor_cache = {}
        
        # Strategy mappings
        self.strategy_mappings = {
            'momentum': ['BUY signal with momentum', 'momentum_strategy'],
            'mean_reversion': ['SELL signal with reversion', 'mean_reversion_strategy'],
            'multi_factor': ['Enhanced signal', 'ml_prediction'],
            'manual': ['Manual trade', 'Manual']
        }
        
        logger.info("📊 Performance Attribution Analyzer initialized")
    
    def add_trade_record(self, trade_data: Dict):
        """Add trade record for attribution analysis"""
        try:
            # Enhance trade record with attribution data
            enhanced_trade = {
                **trade_data,
                'attribution_timestamp': datetime.now(),
                'strategy_classified': self._classify_trade_strategy(trade_data),
                'market_conditions': self._get_market_conditions_at_trade(trade_data),
                'factor_exposures': self._calculate_trade_factor_exposures(trade_data)
            }
            
            self.trade_history.append(enhanced_trade)
            
            # Limit history size for performance
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-800:]  # Keep most recent 800
                
            logger.debug(f"📝 Trade record added for attribution: {trade_data.get('symbol', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error adding trade record: {e}")
    
    def _classify_trade_strategy(self, trade_data: Dict) -> str:
        """Classify trade into strategy category"""
        try:
            reason = trade_data.get('reason', '').lower()
            
            for strategy, keywords in self.strategy_mappings.items():
                if any(keyword.lower() in reason for keyword in keywords):
                    return strategy
            
            # Default classification based on other signals
            if 'auto' in reason or 'signal' in reason:
                return 'multi_factor'
            else:
                return 'manual'
                
        except Exception as e:
            logger.warning(f"Error classifying trade strategy: {e}")
            return 'unknown'
    
    def _get_market_conditions_at_trade(self, trade_data: Dict) -> Dict:
        """Get market conditions at time of trade"""
        try:
            # Get VIX proxy and market trend
            symbol = trade_data.get('symbol', 'SPY')
            timestamp = trade_data.get('timestamp')
            
            # Simple market condition assessment
            conditions = {
                'vix_proxy': 20.0,  # Default
                'market_trend': 'neutral',
                'volatility_regime': 'normal',
                'sector': self._get_symbol_sector(symbol)
            }
            
            # Could be enhanced with real-time market data
            return conditions
            
        except Exception as e:
            logger.warning(f"Error getting market conditions: {e}")
            return {'vix_proxy': 20.0, 'market_trend': 'neutral', 'volatility_regime': 'normal'}
    
    def _calculate_trade_factor_exposures(self, trade_data: Dict) -> Dict[str, float]:
        """Calculate factor exposures for a trade"""
        try:
            symbol = trade_data.get('symbol', '')
            side = trade_data.get('side', 'buy').lower()
            quantity = float(trade_data.get('quantity', 0))
            
            # Calculate exposures to different factors
            exposures = {}
            
            # Market beta exposure (simplified)
            if symbol in ['SPY', 'QQQ', 'IWM']:
                exposures[PerformanceFactors.MARKET_BETA.value] = 1.0 if side == 'buy' else -1.0
            else:
                # Individual stock beta approximation
                sector_betas = {
                    'tech': 1.2, 'auto': 1.4, 'consumer': 0.9, 
                    'media': 1.1, 'market': 1.0, 'other': 1.0
                }
                sector = self._get_symbol_sector(symbol)
                beta = sector_betas.get(sector, 1.0)
                exposures[PerformanceFactors.MARKET_BETA.value] = beta if side == 'buy' else -beta
            
            # Sector allocation exposure
            sector = self._get_symbol_sector(symbol)
            exposures[PerformanceFactors.SECTOR_ALLOCATION.value] = 1.0 if side == 'buy' else -1.0
            
            # Momentum exposure (based on trade reasoning)
            reason = trade_data.get('reason', '').lower()
            if 'momentum' in reason or 'trend' in reason:
                exposures[PerformanceFactors.MOMENTUM.value] = 1.0 if side == 'buy' else -1.0
            elif 'reversion' in reason or 'oversold' in reason:
                exposures[PerformanceFactors.MEAN_REVERSION.value] = 1.0 if side == 'buy' else -1.0
            
            return exposures
            
        except Exception as e:
            logger.warning(f"Error calculating factor exposures: {e}")
            return {}
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'NVDA': 'tech', 'META': 'tech',
            'TSLA': 'auto', 'AMZN': 'consumer', 'NFLX': 'media',
            'SPY': 'market', 'QQQ': 'tech_etf', 'IWM': 'market'
        }
        return sector_map.get(symbol, 'other')
    
    def calculate_strategy_performance(self, lookback_days: int = 30) -> List[StrategyPerformance]:
        """Calculate performance metrics by strategy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date]
            
            if not recent_trades:
                return []
            
            # Group trades by strategy
            strategy_groups = {}
            for trade in recent_trades:
                strategy = trade.get('strategy_classified', 'unknown')
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(trade)
            
            strategy_performances = []
            
            for strategy_name, trades in strategy_groups.items():
                perf = self._calculate_single_strategy_performance(strategy_name, trades)
                if perf:
                    strategy_performances.append(perf)
            
            logger.info(f"📊 Calculated performance for {len(strategy_performances)} strategies")
            return strategy_performances
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return []
    
    def _calculate_single_strategy_performance(self, strategy_name: str, trades: List[Dict]) -> Optional[StrategyPerformance]:
        """Calculate performance metrics for a single strategy"""
        try:
            if not trades:
                return None
            
            # Extract P&L data
            pnls = []
            for trade in trades:
                # Calculate P&L from trade data
                if 'net_trade_value' in trade and trade['net_trade_value']:
                    entry_value = abs(float(trade.get('trade_value', 0)))
                    net_value = float(trade['net_trade_value'])
                    side = trade.get('side', 'buy').lower()
                    
                    # Simple P&L calculation (this would be enhanced with real exit data)
                    if side == 'buy':
                        pnl = net_value - entry_value  # Simplified
                    else:
                        pnl = entry_value - net_value  # Simplified
                    
                    pnls.append(pnl)
            
            if not pnls:
                return None
            
            # Calculate basic metrics
            total_pnl = sum(pnls)
            num_trades = len(pnls)
            avg_trade_pnl = total_pnl / num_trades
            
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p <= 0]
            
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Risk metrics (simplified)
            if len(pnls) > 1:
                returns_array = np.array(pnls)
                volatility = np.std(returns_array)
                sharpe_ratio = avg_trade_pnl / volatility if volatility > 0 else 0
                
                # Max drawdown (simplified)
                cumulative = np.cumsum(returns_array)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = running_max - cumulative
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
                
                # Information ratio (excess return / tracking error)
                benchmark_return = 0.001  # Simplified benchmark
                excess_returns = returns_array - benchmark_return
                tracking_error = np.std(excess_returns) if len(excess_returns) > 1 else 1
                information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
                
                # Calmar ratio (return / max drawdown)
                calmar_ratio = avg_trade_pnl / max_drawdown if max_drawdown > 0 else 0
                
            else:
                volatility = abs(avg_trade_pnl)
                sharpe_ratio = 1.0 if avg_trade_pnl > 0 else -1.0
                max_drawdown = abs(min(pnls)) if pnls else 0
                information_ratio = 0.5
                calmar_ratio = 0.5
            
            # Risk-adjusted return
            risk_adjusted_return = total_pnl / (volatility * num_trades) if volatility > 0 else total_pnl
            
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_pnl=total_pnl,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                num_trades=num_trades,
                avg_trade_pnl=avg_trade_pnl,
                risk_adjusted_return=risk_adjusted_return,
                volatility=volatility,
                information_ratio=information_ratio,
                calmar_ratio=calmar_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance for {strategy_name}: {e}")
            return None
    
    def calculate_factor_attribution(self, lookback_days: int = 30) -> List[FactorAttribution]:
        """Calculate factor-based performance attribution"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date]
            
            if not recent_trades:
                return []
            
            attributions = []
            
            # Analyze each major factor
            for factor in PerformanceFactors:
                factor_attribution = self._calculate_single_factor_attribution(factor, recent_trades)
                if factor_attribution:
                    attributions.append(factor_attribution)
            
            logger.info(f"📊 Calculated attribution for {len(attributions)} factors")
            return attributions
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return []
    
    def _calculate_single_factor_attribution(self, factor: PerformanceFactors, trades: List[Dict]) -> Optional[FactorAttribution]:
        """Calculate attribution for a single factor"""
        try:
            factor_trades = []
            factor_exposures = []
            
            for trade in trades:
                exposures = trade.get('factor_exposures', {})
                if factor.value in exposures:
                    factor_trades.append(trade)
                    factor_exposures.append(exposures[factor.value])
            
            if not factor_trades:
                return None
            
            # Calculate factor contribution
            total_pnl = 0
            total_portfolio_value = 0
            
            for trade in factor_trades:
                trade_value = float(trade.get('trade_value', 0))
                net_value = float(trade.get('net_trade_value', trade_value))
                pnl = net_value - trade_value  # Simplified
                
                total_pnl += pnl
                total_portfolio_value += trade_value
            
            avg_exposure = np.mean(factor_exposures) if factor_exposures else 0
            
            # Factor return (simplified)
            factor_return = total_pnl / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Active return vs benchmark (simplified)
            benchmark_return = 0.001  # Daily benchmark return assumption
            active_return = factor_return - benchmark_return
            
            # Tracking error (simplified)
            tracking_error = abs(active_return) * 0.5  # Simplified calculation
            
            # Contribution percentage
            total_strategy_pnl = sum(float(t.get('net_trade_value', t.get('trade_value', 0))) - 
                                   float(t.get('trade_value', 0)) for t in trades)
            contribution_pct = (total_pnl / total_strategy_pnl * 100) if total_strategy_pnl != 0 else 0
            
            return FactorAttribution(
                factor=factor,
                contribution_pnl=total_pnl,
                contribution_pct=contribution_pct,
                factor_exposure=avg_exposure,
                factor_return=factor_return,
                active_return=active_return,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution for {factor.value}: {e}")
            return None
    
    def calculate_benchmark_comparison(self, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate performance vs benchmark"""
        try:
            # Get benchmark data
            benchmark_return = self._get_benchmark_return(lookback_days)
            
            # Get strategy returns
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date]
            
            if not recent_trades:
                return {}
            
            # Calculate strategy return
            total_strategy_pnl = 0
            total_invested = 0
            
            for trade in recent_trades:
                trade_value = float(trade.get('trade_value', 0))
                net_value = float(trade.get('net_trade_value', trade_value))
                total_strategy_pnl += (net_value - trade_value)
                total_invested += trade_value
            
            strategy_return = total_strategy_pnl / total_invested if total_invested > 0 else 0
            
            # Calculate comparison metrics
            comparison = {
                'strategy_return': strategy_return,
                'benchmark_return': benchmark_return,
                'excess_return': strategy_return - benchmark_return,
                'tracking_error': abs(strategy_return - benchmark_return) * 2,  # Simplified
                'information_ratio': (strategy_return - benchmark_return) / abs(strategy_return - benchmark_return + 0.001),
                'beta': 1.2 if strategy_return > benchmark_return else 0.8,  # Simplified
                'alpha': strategy_return - (benchmark_return * 1.0),  # Simplified alpha
                'up_capture': min(2.0, strategy_return / benchmark_return) if benchmark_return > 0 else 1.0,
                'down_capture': min(2.0, abs(strategy_return) / abs(benchmark_return)) if benchmark_return < 0 else 1.0
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}
    
    def _get_benchmark_return(self, lookback_days: int) -> float:
        """Get benchmark return for comparison"""
        try:
            # Simple benchmark return calculation
            # In production, this would fetch real SPY data
            
            # Simulate reasonable benchmark return
            daily_return = 0.0004  # ~10% annualized
            return daily_return * lookback_days
            
        except Exception as e:
            logger.warning(f"Error getting benchmark return: {e}")
            return 0.001  # Default small positive return
    
    def generate_comprehensive_attribution(self, lookback_days: int = 30) -> PerformanceBreakdown:
        """Generate comprehensive performance attribution analysis"""
        try:
            # Calculate all components
            strategy_performance = self.calculate_strategy_performance(lookback_days)
            factor_attribution = self.calculate_factor_attribution(lookback_days)
            benchmark_comparison = self.calculate_benchmark_comparison(lookback_days)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(lookback_days)
            
            # Calculate cost analysis
            cost_analysis = self._calculate_cost_analysis(lookback_days)
            
            # Total P&L
            total_pnl = sum(s.total_pnl for s in strategy_performance)
            
            breakdown = PerformanceBreakdown(
                timestamp=datetime.now(),
                total_pnl=total_pnl,
                strategy_performance=strategy_performance,
                factor_attribution=factor_attribution,
                risk_metrics=risk_metrics,
                benchmark_comparison=benchmark_comparison,
                cost_analysis=cost_analysis
            )
            
            logger.info(f"📊 Generated comprehensive attribution: Total P&L=${total_pnl:.2f}")
            return breakdown
            
        except Exception as e:
            logger.error(f"Error generating comprehensive attribution: {e}")
            # Return empty breakdown
            return PerformanceBreakdown(
                timestamp=datetime.now(),
                total_pnl=0.0,
                strategy_performance=[],
                factor_attribution=[],
                risk_metrics={},
                benchmark_comparison={},
                cost_analysis={}
            )
    
    def _calculate_risk_metrics(self, lookback_days: int) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date]
            
            if not recent_trades:
                return {}
            
            # Extract returns
            returns = []
            for trade in recent_trades:
                trade_value = float(trade.get('trade_value', 0))
                net_value = float(trade.get('net_trade_value', trade_value))
                if trade_value > 0:
                    ret = (net_value - trade_value) / trade_value
                    returns.append(ret)
            
            if len(returns) < 2:
                return {}
            
            returns_array = np.array(returns)
            
            risk_metrics = {
                'volatility': float(np.std(returns_array)),
                'skewness': float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 3)),
                'kurtosis': float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 4)),
                'var_95': float(np.percentile(returns_array, 5)),
                'var_99': float(np.percentile(returns_array, 1)),
                'expected_shortfall': float(np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)])),
                'hit_ratio': float(len([r for r in returns if r > 0]) / len(returns)),
                'profit_loss_ratio': float(np.mean([r for r in returns if r > 0]) / abs(np.mean([r for r in returns if r <= 0]))) if any(r <= 0 for r in returns) else 1.0
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_cost_analysis(self, lookback_days: int) -> Dict[str, float]:
        """Calculate transaction cost impact analysis"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01')) > cutoff_date]
            
            if not recent_trades:
                return {}
            
            total_costs = 0
            total_trade_value = 0
            cost_breakdown = {'commission': 0, 'spread': 0, 'market_impact': 0, 'other': 0}
            
            for trade in recent_trades:
                transaction_costs = trade.get('transaction_costs', {})
                trade_value = float(trade.get('trade_value', 0))
                
                total_trade_value += trade_value
                
                if isinstance(transaction_costs, dict):
                    trade_total_cost = transaction_costs.get('total_cost', 0)
                    total_costs += trade_total_cost
                    
                    # Breakdown by cost type
                    cost_breakdown['commission'] += transaction_costs.get('commission', 0)
                    cost_breakdown['spread'] += transaction_costs.get('spread_cost', 0)
                    cost_breakdown['market_impact'] += transaction_costs.get('market_impact', 0)
            
            cost_analysis = {
                'total_transaction_costs': total_costs,
                'cost_as_pct_of_volume': (total_costs / total_trade_value * 100) if total_trade_value > 0 else 0,
                'avg_cost_per_trade': total_costs / len(recent_trades) if recent_trades else 0,
                'cost_breakdown_pct': {
                    k: (v / total_costs * 100) if total_costs > 0 else 0 
                    for k, v in cost_breakdown.items()
                },
                'cost_drag_on_returns': (total_costs / total_trade_value * 100) if total_trade_value > 0 else 0
            }
            
            return cost_analysis
            
        except Exception as e:
            logger.warning(f"Error calculating cost analysis: {e}")
            return {}

# Global instance
performance_attribution_analyzer = None

def initialize_performance_attribution_analyzer(api_client: tradeapi.REST):
    """Initialize the performance attribution analyzer"""
    global performance_attribution_analyzer
    performance_attribution_analyzer = PerformanceAttributionAnalyzer(api_client)
    logger.info("✅ Performance Attribution Analyzer initialized")

def add_trade_for_attribution(trade_data: Dict):
    """Add trade record for attribution analysis"""
    if performance_attribution_analyzer:
        performance_attribution_analyzer.add_trade_record(trade_data)

def get_performance_attribution(lookback_days: int = 30) -> Dict:
    """Get comprehensive performance attribution"""
    if not performance_attribution_analyzer:
        return {"error": "Attribution analyzer not initialized"}
    
    try:
        breakdown = performance_attribution_analyzer.generate_comprehensive_attribution(lookback_days)
        
        # Convert to serializable dictionary
        return {
            "timestamp": breakdown.timestamp.isoformat(),
            "total_pnl": breakdown.total_pnl,
            "strategy_performance": [asdict(s) for s in breakdown.strategy_performance],
            "factor_attribution": [
                {
                    "factor": f.factor.value,
                    "contribution_pnl": f.contribution_pnl,
                    "contribution_pct": f.contribution_pct,
                    "factor_exposure": f.factor_exposure,
                    "factor_return": f.factor_return,
                    "active_return": f.active_return,
                    "tracking_error": f.tracking_error
                } for f in breakdown.factor_attribution
            ],
            "risk_metrics": breakdown.risk_metrics,
            "benchmark_comparison": breakdown.benchmark_comparison,
            "cost_analysis": breakdown.cost_analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting performance attribution: {e}")
        return {"error": str(e)}

def get_strategy_performance_summary() -> Dict:
    """Get summary of strategy performance"""
    if not performance_attribution_analyzer:
        return {"error": "Attribution analyzer not initialized"}
    
    try:
        strategies = performance_attribution_analyzer.calculate_strategy_performance(30)
        return {
            "strategies": [asdict(s) for s in strategies],
            "top_performer": max(strategies, key=lambda x: x.sharpe_ratio).strategy_name if strategies else None,
            "total_strategies": len(strategies)
        }
    except Exception as e:
        logger.error(f"Error getting strategy summary: {e}")
        return {"error": str(e)}
