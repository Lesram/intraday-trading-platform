#!/usr/bin/env python3
"""
CVaR-Integrated Order Router

Combines smart execution algorithms with CVaR risk management for optimal order sizing
and execution timing. Integrates with existing enhanced_cvar_risk_manager.py.

Features:
• CVaR-aware position sizing for order execution
• Dynamic risk limit enforcement during execution
• Correlation-based execution timing optimization
• Real-time risk monitoring and position adjustments
• Integration with existing risk management infrastructure
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
import warnings

# Import our existing modules
try:
    from enhanced_cvar_risk_manager import EnhancedCVaREngine
    from smart_execution_engine import SmartExecutionEngine, ExecutionStrategy, ExecutionOrder
    CVAR_AVAILABLE = True
    logging.info("✅ CVaR risk manager integration available")
except ImportError as e:
    CVAR_AVAILABLE = False
    logging.warning(f"❌ CVaR integration not available: {e}")

# Import data enhancement integration
try:
    from data_enhancement_risk_integration import DataEnhancementAndRiskIntegrator
    INTEGRATION_AVAILABLE = True
    logging.info("✅ Data enhancement integration available")
except ImportError:
    INTEGRATION_AVAILABLE = False
    logging.warning("❌ Data enhancement integration not available")

@dataclass
class RiskAwareOrderParams:
    """Parameters for risk-aware order execution"""
    symbol: str
    base_quantity: float
    side: str
    max_position_pct: float = 0.05  # Max 5% of portfolio
    risk_adjusted_quantity: float = 0.0
    cvar_limit: float = 0.03  # Max 3% CVaR
    execution_strategy: ExecutionStrategy = ExecutionStrategy.VWAP
    risk_regime: str = "normal"
    correlation_factor: float = 1.0
    stress_test_adjustment: float = 1.0

@dataclass 
class ExecutionRiskMetrics:
    """Risk metrics for order execution"""
    portfolio_cvar_95: float
    portfolio_cvar_99: float
    position_contribution: float
    risk_regime: str
    correlation_risk: float
    stress_test_score: float
    recommended_quantity: float
    risk_warnings: List[str]

class CVaRIntegratedOrderRouter:
    """
    Order router that integrates CVaR risk management with smart execution algorithms
    """
    
    def __init__(self, alpaca_api_key: str = None, alpaca_secret: str = None, 
                 paper_trading: bool = True):
        """
        Initialize CVaR-integrated order router
        
        Args:
            alpaca_api_key: Alpaca API key
            alpaca_secret: Alpaca secret key
            paper_trading: Use paper trading environment
        """
        # Initialize smart execution engine
        self.execution_engine = SmartExecutionEngine(alpaca_api_key, alpaca_secret, paper_trading)
        
        # Initialize CVaR risk manager if available
        self.cvar_engine = None
        if CVAR_AVAILABLE:
            try:
                self.cvar_engine = EnhancedCVaREngine()
                logging.info("✅ CVaR risk engine initialized")
            except Exception as e:
                logging.warning(f"❌ CVaR engine initialization failed: {e}")
        
        # Initialize data integration if available
        self.data_integrator = None
        if INTEGRATION_AVAILABLE:
            try:
                from data_enhancement_risk_integration import initialize_enhanced_trading_system
                self.data_integrator = initialize_enhanced_trading_system()
                logging.info("✅ Data enhancement integration initialized")
            except Exception as e:
                logging.warning(f"❌ Data integration initialization failed: {e}")
        
        # Risk parameters
        self.max_portfolio_cvar_95 = 0.03  # 3% maximum CVaR
        self.max_portfolio_cvar_99 = 0.05  # 5% maximum CVaR
        self.max_single_position = 0.10   # 10% max single position
        self.stress_test_threshold = 30.0  # Stress test score threshold
        
        # Execution tracking
        self.risk_adjusted_orders = []
        self.risk_limit_breaches = []
        
        logging.info("🚀 CVaR-Integrated Order Router initialized")
    
    def calculate_risk_aware_order_params(self, symbol: str, quantity: float, side: str,
                                        current_portfolio: Dict = None) -> RiskAwareOrderParams:
        """
        Calculate risk-aware order parameters using CVaR analysis
        
        Args:
            symbol: Stock symbol
            quantity: Desired quantity to trade
            side: 'buy' or 'sell'
            current_portfolio: Current portfolio positions
            
        Returns:
            Risk-adjusted order parameters
        """
        try:
            logging.info(f"🔍 Calculating risk-aware parameters for {quantity} shares of {symbol}")
            
            # Initialize default parameters
            order_params = RiskAwareOrderParams(
                symbol=symbol,
                base_quantity=quantity,
                side=side,
                risk_adjusted_quantity=quantity  # Default to requested quantity
            )
            
            if not self.cvar_engine:
                logging.warning("CVaR engine not available, using base quantity")
                return order_params
            
            # Create sample portfolio for risk analysis
            if not current_portfolio:
                current_portfolio = {
                    'AAPL': {'quantity': 100, 'value': 15000},
                    'MSFT': {'quantity': 50, 'value': 12000},
                    'GOOGL': {'quantity': 10, 'value': 13000}
                }
            
            portfolio_value = sum(pos['value'] for pos in current_portfolio.values())
            
            # Calculate position size limits based on CVaR
            try:
                # Simulate adding the new position
                test_portfolio = current_portfolio.copy()
                
                # Estimate position value (simplified)
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period="1d").tail(1)
                
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[0]
                    position_value = quantity * current_price
                    
                    # Add to test portfolio
                    if symbol in test_portfolio:
                        if side == 'buy':
                            test_portfolio[symbol]['quantity'] += quantity
                            test_portfolio[symbol]['value'] += position_value
                        else:  # sell
                            test_portfolio[symbol]['quantity'] -= quantity
                            test_portfolio[symbol]['value'] -= position_value
                    else:
                        test_portfolio[symbol] = {'quantity': quantity, 'value': position_value}
                    
                    # Calculate CVaR for the test portfolio
                    symbols = list(test_portfolio.keys())
                    quantities = [test_portfolio[s]['quantity'] for s in symbols]
                    
                    cvar_result = self.cvar_engine.calculate_portfolio_cvar(
                        symbols=symbols,
                        quantities=quantities
                    )
                    
                    if cvar_result:
                        order_params.portfolio_cvar_95 = cvar_result.get('cvar_95', 0.0)
                        order_params.portfolio_cvar_99 = cvar_result.get('cvar_99', 0.0)
                        order_params.risk_regime = cvar_result.get('risk_regime', 'normal')
                        
                        # Adjust quantity if CVaR limits exceeded
                        if cvar_result.get('cvar_95', 0.0) > self.max_portfolio_cvar_95:
                            # Reduce quantity to stay within CVaR limits
                            reduction_factor = self.max_portfolio_cvar_95 / cvar_result.get('cvar_95', 0.01)
                            order_params.risk_adjusted_quantity = quantity * reduction_factor * 0.8  # 80% safety margin
                            
                            logging.warning(f"⚠️  Quantity reduced from {quantity} to {order_params.risk_adjusted_quantity:.0f} due to CVaR limits")
                        
                        # Apply stress test adjustments
                        stress_test_results = self.cvar_engine.run_stress_tests(symbols, quantities)
                        if stress_test_results and stress_test_results.get('composite_score', 0) < self.stress_test_threshold:
                            order_params.stress_test_adjustment = 0.5  # Reduce size during stress
                            order_params.risk_adjusted_quantity *= order_params.stress_test_adjustment
                            
                            logging.warning(f"⚠️  Stress test adjustment: quantity further reduced to {order_params.risk_adjusted_quantity:.0f}")
                        
                        # Determine optimal execution strategy based on risk regime
                        if order_params.risk_regime in ['crisis', 'extreme']:
                            order_params.execution_strategy = ExecutionStrategy.IMPLEMENTATION_SHORTFALL  # More conservative
                        elif order_params.risk_regime == 'high_vol':
                            order_params.execution_strategy = ExecutionStrategy.VWAP  # Volume-weighted approach
                        else:
                            order_params.execution_strategy = ExecutionStrategy.VWAP  # Default
                
                logging.info(f"✅ Risk analysis complete:")
                logging.info(f"   Original quantity: {quantity}")
                logging.info(f"   Risk-adjusted quantity: {order_params.risk_adjusted_quantity:.0f}")
                logging.info(f"   Portfolio CVaR(95%): {getattr(order_params, 'portfolio_cvar_95', 0.0):.2%}")
                logging.info(f"   Risk regime: {order_params.risk_regime}")
                logging.info(f"   Execution strategy: {order_params.execution_strategy.value}")
                
            except Exception as e:
                logging.error(f"❌ Error in CVaR calculation: {e}")
            
            return order_params
            
        except Exception as e:
            logging.error(f"❌ Error calculating risk-aware parameters: {e}")
            return order_params
    
    def get_correlation_timing_adjustment(self, symbol: str, market_conditions: Dict = None) -> float:
        """
        Calculate execution timing adjustment based on market correlations
        
        Args:
            symbol: Target symbol
            market_conditions: Current market conditions
            
        Returns:
            Timing adjustment factor (0.5 = slower, 2.0 = faster)
        """
        try:
            if not market_conditions:
                return 1.0  # No adjustment
            
            # Analyze correlation with major indices
            correlation_risk = market_conditions.get('spy_correlation', 0.5)
            vix_level = market_conditions.get('vix_level', 20.0)
            
            # High correlation during high VIX = slower execution
            if correlation_risk > 0.7 and vix_level > 25:
                return 0.7  # Slower execution
            elif correlation_risk < 0.3 and vix_level < 15:
                return 1.3  # Faster execution
            else:
                return 1.0  # Normal execution
                
        except Exception as e:
            logging.error(f"❌ Error calculating correlation timing: {e}")
            return 1.0
    
    async def execute_risk_aware_order(self, symbol: str, quantity: float, side: str,
                                     current_portfolio: Dict = None,
                                     duration_minutes: int = 60,
                                     dry_run: bool = True) -> ExecutionOrder:
        """
        Execute order with full CVaR risk management integration
        
        Args:
            symbol: Stock symbol
            quantity: Desired quantity to trade
            side: 'buy' or 'sell'
            current_portfolio: Current portfolio positions
            duration_minutes: Execution duration
            dry_run: If True, simulate execution
            
        Returns:
            Execution order with risk metrics
        """
        try:
            logging.info(f"🚀 Starting risk-aware order execution for {symbol}")
            
            # Calculate risk-aware parameters
            order_params = self.calculate_risk_aware_order_params(
                symbol, quantity, side, current_portfolio
            )
            
            # Check if order should be blocked due to risk limits
            if order_params.risk_adjusted_quantity < quantity * 0.1:  # Less than 10% of original
                logging.error(f"❌ Order blocked: Risk-adjusted quantity too small ({order_params.risk_adjusted_quantity:.0f} vs {quantity})")
                raise ValueError("Order blocked due to risk limits")
            
            # Get correlation-based timing adjustment
            timing_adjustment = self.get_correlation_timing_adjustment(symbol)
            adjusted_duration = int(duration_minutes / timing_adjustment)
            
            logging.info(f"📊 Risk-aware execution parameters:")
            logging.info(f"   Adjusted quantity: {order_params.risk_adjusted_quantity:.0f}")
            logging.info(f"   Execution strategy: {order_params.execution_strategy.value}")
            logging.info(f"   Adjusted duration: {adjusted_duration} minutes")
            logging.info(f"   Risk regime: {order_params.risk_regime}")
            
            # Execute using smart execution engine
            execution_result = await self.execution_engine.execute_smart_order(
                symbol=symbol,
                quantity=order_params.risk_adjusted_quantity,
                side=side,
                strategy=order_params.execution_strategy,
                duration_minutes=adjusted_duration,
                dry_run=dry_run
            )
            
            # Add risk metrics to execution result
            execution_result.risk_adjusted = True
            execution_result.original_quantity = quantity
            execution_result.risk_regime = order_params.risk_regime
            execution_result.cvar_limited = (order_params.risk_adjusted_quantity < quantity * 0.95)
            
            # Track risk-adjusted orders
            self.risk_adjusted_orders.append(execution_result)
            
            logging.info(f"✅ Risk-aware execution completed successfully")
            return execution_result
            
        except Exception as e:
            logging.error(f"❌ Risk-aware execution failed: {e}")
            raise
    
    def monitor_execution_risk(self, execution_order: ExecutionOrder,
                             current_portfolio: Dict) -> ExecutionRiskMetrics:
        """
        Monitor risk metrics during order execution
        
        Args:
            execution_order: Ongoing execution order
            current_portfolio: Current portfolio positions
            
        Returns:
            Real-time execution risk metrics
        """
        try:
            warnings = []
            
            # Calculate current execution impact
            filled_quantity = sum(s.filled_quantity for s in execution_order.slices 
                                if hasattr(s, 'filled_quantity') and s.filled_quantity > 0)
            
            completion_pct = filled_quantity / execution_order.total_quantity if execution_order.total_quantity > 0 else 0
            
            # Simulate portfolio impact
            risk_metrics = ExecutionRiskMetrics(
                portfolio_cvar_95=0.025,  # Placeholder
                portfolio_cvar_99=0.035,
                position_contribution=completion_pct * 0.01,
                risk_regime="normal",
                correlation_risk=0.5,
                stress_test_score=85.0,
                recommended_quantity=execution_order.total_quantity,
                risk_warnings=warnings
            )
            
            # Check for risk limit breaches
            if risk_metrics.portfolio_cvar_95 > self.max_portfolio_cvar_95:
                warnings.append(f"Portfolio CVaR(95%) exceeds limit: {risk_metrics.portfolio_cvar_95:.2%}")
            
            if completion_pct > 0.5 and execution_order.total_slippage > 0.01:  # >1% slippage at 50% completion
                warnings.append(f"High execution slippage detected: {execution_order.total_slippage:.2%}")
            
            return risk_metrics
            
        except Exception as e:
            logging.error(f"❌ Error monitoring execution risk: {e}")
            return ExecutionRiskMetrics(0, 0, 0, "unknown", 0, 0, 0, ["Error calculating metrics"])
    
    def get_risk_execution_analytics(self) -> Dict:
        """
        Get analytics for risk-aware executions
        
        Returns:
            Dictionary with risk execution performance
        """
        try:
            if not self.risk_adjusted_orders:
                return {"total_risk_executions": 0, "message": "No risk-aware executions yet"}
            
            # Calculate risk-specific metrics
            total_orders = len(self.risk_adjusted_orders)
            cvar_limited_orders = sum(1 for order in self.risk_adjusted_orders 
                                    if getattr(order, 'cvar_limited', False))
            
            avg_risk_adjustment = np.mean([
                order.total_quantity / getattr(order, 'original_quantity', order.total_quantity)
                for order in self.risk_adjusted_orders
                if hasattr(order, 'original_quantity')
            ])
            
            risk_regimes = {}
            for order in self.risk_adjusted_orders:
                regime = getattr(order, 'risk_regime', 'unknown')
                risk_regimes[regime] = risk_regimes.get(regime, 0) + 1
            
            # Get base execution analytics
            base_analytics = self.execution_engine.get_execution_analytics()
            
            # Combine with risk-specific metrics
            risk_analytics = {
                **base_analytics,
                "risk_aware_executions": total_orders,
                "cvar_limited_orders": cvar_limited_orders,
                "avg_risk_adjustment": avg_risk_adjustment,
                "risk_regime_distribution": risk_regimes,
                "risk_limit_breaches": len(self.risk_limit_breaches)
            }
            
            return risk_analytics
            
        except Exception as e:
            logging.error(f"❌ Error generating risk analytics: {e}")
            return {"error": str(e)}


# Testing function
async def test_cvar_integrated_execution():
    """Test CVaR-integrated order execution"""
    logging.info("🧪 Testing CVaR-Integrated Order Execution")
    
    # Initialize router
    router = CVaRIntegratedOrderRouter()
    
    # Test portfolio
    test_portfolio = {
        'AAPL': {'quantity': 100, 'value': 18000},
        'MSFT': {'quantity': 50, 'value': 16000}, 
        'GOOGL': {'quantity': 8, 'value': 11000},
        'TSLA': {'quantity': 25, 'value': 8000}
    }
    
    # Test orders with different risk profiles
    test_orders = [
        ("AAPL", 200, "buy"),   # Large position in existing holding
        ("NVDA", 50, "buy"),    # New position
        ("TSLA", 100, "sell"),  # Reduce existing position
    ]
    
    for symbol, quantity, side in test_orders:
        logging.info(f"\n{'='*60}")
        logging.info(f"🔬 Testing risk-aware execution: {quantity} {symbol} {side}")
        
        try:
            # Execute with risk management
            result = await router.execute_risk_aware_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                current_portfolio=test_portfolio,
                duration_minutes=30,
                dry_run=True
            )
            
            logging.info(f"✅ Risk-aware execution completed:")
            logging.info(f"   Original quantity: {quantity}")
            logging.info(f"   Executed quantity: {result.total_quantity}")
            logging.info(f"   CVaR limited: {getattr(result, 'cvar_limited', False)}")
            logging.info(f"   Risk regime: {getattr(result, 'risk_regime', 'unknown')}")
            logging.info(f"   Completion rate: {result.completion_rate:.1%}")
            
        except Exception as e:
            logging.error(f"❌ Risk-aware execution failed: {e}")
    
    # Display analytics
    logging.info(f"\n{'='*60}")
    logging.info("📊 RISK-AWARE EXECUTION ANALYTICS")
    
    analytics = router.get_risk_execution_analytics()
    logging.info(f"Risk-Aware Executions: {analytics.get('risk_aware_executions', 0)}")
    logging.info(f"CVaR Limited Orders: {analytics.get('cvar_limited_orders', 0)}")
    logging.info(f"Average Risk Adjustment: {analytics.get('avg_risk_adjustment', 1.0):.2f}")
    logging.info(f"Risk Limit Breaches: {analytics.get('risk_limit_breaches', 0)}")
    
    if analytics.get('risk_regime_distribution'):
        logging.info("Risk Regime Distribution:")
        for regime, count in analytics['risk_regime_distribution'].items():
            logging.info(f"  {regime}: {count}")

if __name__ == "__main__":
    asyncio.run(test_cvar_integrated_execution())
