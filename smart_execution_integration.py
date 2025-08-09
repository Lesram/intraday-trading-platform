#!/usr/bin/env python3
"""
Smart Order Execution Integration Module

Integrates smart order execution capabilities with the existing trading system,
providing TWAP, VWAP, and Implementation Shortfall algorithms with CVaR risk management.

Features:
• Seamless integration with existing runfile.py trading system
• Smart execution strategy selection based on market conditions
• CVaR-aware order sizing and risk management
• Real-time execution monitoring and analytics
• Transaction cost analysis and optimization
"""

import asyncio
import logging

# Import our smart execution modules
try:
    from cvar_integrated_order_router import CVaRIntegratedOrderRouter
    from smart_execution_engine import ExecutionOrder, ExecutionStrategy, SmartExecutionEngine
    SMART_EXECUTION_AVAILABLE = True
    logging.info("✅ Smart execution modules available")
except ImportError as e:
    SMART_EXECUTION_AVAILABLE = False
    logging.warning(f"❌ Smart execution not available: {e}")

# Import existing system modules
try:
    from enhanced_cvar_risk_manager import EnhancedCVaREngine
    CVAR_AVAILABLE = True
except ImportError:
    CVAR_AVAILABLE = False

class SmartOrderExecutionIntegrator:
    """
    Integration layer for smart order execution with existing trading system
    """

    def __init__(self, alpaca_api_key: str = None, alpaca_secret: str = None,
                 paper_trading: bool = True):
        """
        Initialize Smart Order Execution Integration
        
        Args:
            alpaca_api_key: Alpaca API key
            alpaca_secret: Alpaca secret key
            paper_trading: Use paper trading environment
        """
        self.initialized = False

        if not SMART_EXECUTION_AVAILABLE:
            logging.error("❌ Smart execution modules not available")
            return

        # Initialize smart execution components
        try:
            self.execution_engine = SmartExecutionEngine(alpaca_api_key, alpaca_secret, paper_trading)
            self.cvar_router = CVaRIntegratedOrderRouter(alpaca_api_key, alpaca_secret, paper_trading)

            # Execution configuration
            self.default_strategy = ExecutionStrategy.VWAP
            self.default_duration = 60  # 60 minutes
            self.risk_aware_execution = True

            # Performance tracking
            self.execution_history = []
            self.total_executions = 0
            self.total_cost_savings = 0.0

            self.initialized = True
            logging.info("🚀 Smart Order Execution Integration initialized successfully")

        except Exception as e:
            logging.error(f"❌ Failed to initialize smart execution integration: {e}")

    def select_optimal_strategy(self, symbol: str, quantity: float,
                               market_conditions: dict = None) -> ExecutionStrategy:
        """
        Select optimal execution strategy based on order characteristics and market conditions
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            market_conditions: Current market conditions
            
        Returns:
            Optimal execution strategy
        """
        try:
            # Default strategy selection logic
            if not market_conditions:
                return ExecutionStrategy.VWAP  # Default to VWAP

            volatility = market_conditions.get('volatility', 0.2)
            volume_ratio = market_conditions.get('volume_ratio', 1.0)  # vs avg volume
            spread = market_conditions.get('spread', 0.001)  # bid-ask spread

            # Strategy selection rules:
            # 1. High volatility + large order → Implementation Shortfall (minimize timing risk)
            # 2. Normal volatility + volume available → VWAP (follow volume patterns)
            # 3. Low volume periods → TWAP (spread execution over time)
            # 4. Small orders → Market execution (no slicing needed)

            if volatility > 0.3 and quantity > 1000:
                return ExecutionStrategy.IMPLEMENTATION_SHORTFALL
            elif volume_ratio < 0.5:  # Low volume period
                return ExecutionStrategy.TWAP
            elif spread > 0.005:  # Wide spread
                return ExecutionStrategy.IMPLEMENTATION_SHORTFALL
            else:
                return ExecutionStrategy.VWAP  # Default for most cases

        except Exception as e:
            logging.warning(f"Error selecting strategy, using default: {e}")
            return ExecutionStrategy.VWAP

    def calculate_optimal_duration(self, symbol: str, quantity: float,
                                  strategy: ExecutionStrategy) -> int:
        """
        Calculate optimal execution duration based on order size and strategy
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            strategy: Execution strategy
            
        Returns:
            Optimal duration in minutes
        """
        try:
            # Base duration calculation based on order size
            if quantity <= 100:
                base_duration = 15  # Small orders: 15 minutes
            elif quantity <= 500:
                base_duration = 30  # Medium orders: 30 minutes
            elif quantity <= 1000:
                base_duration = 60  # Large orders: 1 hour
            else:
                base_duration = 120  # Very large orders: 2 hours

            # Strategy-specific adjustments
            if strategy == ExecutionStrategy.TWAP:
                return base_duration  # TWAP uses base duration
            elif strategy == ExecutionStrategy.VWAP:
                return min(base_duration, 90)  # VWAP limited to market hours patterns
            elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
                return int(base_duration * 0.7)  # IS favors faster execution
            else:
                return base_duration

        except Exception as e:
            logging.warning(f"Error calculating duration, using default: {e}")
            return 60

    async def execute_smart_order_integrated(self, symbol: str, quantity: float, side: str,
                                           strategy: ExecutionStrategy = None,
                                           duration_minutes: int = None,
                                           use_risk_management: bool = True,
                                           current_portfolio: dict = None,
                                           dry_run: bool = True) -> dict:
        """
        Execute smart order with full integration capabilities
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            strategy: Execution strategy (auto-selected if None)
            duration_minutes: Execution duration (auto-calculated if None)
            use_risk_management: Enable CVaR risk management
            current_portfolio: Current portfolio positions
            dry_run: Simulate execution if True
            
        Returns:
            Dictionary with execution results and analytics
        """
        try:
            if not self.initialized:
                raise ValueError("Smart execution integration not initialized")

            logging.info("🚀 Starting integrated smart order execution:")
            logging.info(f"   Symbol: {symbol}, Quantity: {quantity}, Side: {side}")
            logging.info(f"   Risk Management: {'Enabled' if use_risk_management else 'Disabled'}")

            # Auto-select strategy if not provided
            if strategy is None:
                strategy = self.select_optimal_strategy(symbol, quantity)
                logging.info(f"   Auto-selected strategy: {strategy.value}")

            # Auto-calculate duration if not provided
            if duration_minutes is None:
                duration_minutes = self.calculate_optimal_duration(symbol, quantity, strategy)
                logging.info(f"   Auto-calculated duration: {duration_minutes} minutes")

            # Execute with or without risk management
            if use_risk_management and self.cvar_router:
                execution_result = await self.cvar_router.execute_risk_aware_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    current_portfolio=current_portfolio,
                    duration_minutes=duration_minutes,
                    dry_run=dry_run
                )
            else:
                execution_result = await self.execution_engine.execute_smart_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    strategy=strategy,
                    duration_minutes=duration_minutes,
                    dry_run=dry_run
                )

            # Calculate cost savings vs market order
            market_order_slippage = 0.005  # Assume 50 bps slippage for market order
            smart_execution_slippage = execution_result.total_slippage
            cost_savings = (market_order_slippage - smart_execution_slippage) * quantity

            # Prepare integrated results
            integrated_results = {
                'execution_order': execution_result,
                'strategy_used': strategy.value if strategy else execution_result.strategy.value,
                'duration_used': duration_minutes,
                'cost_savings_estimate': cost_savings,
                'risk_managed': use_risk_management,
                'completion_rate': execution_result.completion_rate,
                'total_slippage': execution_result.total_slippage,
                'market_impact': execution_result.total_market_impact,
                'slices_executed': len([s for s in execution_result.slices if hasattr(s, 'status') and
                                      s.status.value == 'FILLED']),
                'total_slices': len(execution_result.slices),
                'execution_time_minutes': (execution_result.end_time - execution_result.start_time).total_seconds() / 60
            }

            # Update performance tracking
            self.execution_history.append(integrated_results)
            self.total_executions += 1
            self.total_cost_savings += cost_savings

            logging.info("✅ Integrated smart execution completed:")
            logging.info(f"   Strategy: {integrated_results['strategy_used']}")
            logging.info(f"   Completion Rate: {integrated_results['completion_rate']:.1%}")
            logging.info(f"   Total Slippage: {integrated_results['total_slippage']:.2%}")
            logging.info(f"   Estimated Cost Savings: ${cost_savings:.2f}")
            logging.info(f"   Execution Time: {integrated_results['execution_time_minutes']:.1f} minutes")

            return integrated_results

        except Exception as e:
            logging.error(f"❌ Integrated smart execution failed: {e}")
            raise

    def get_integration_analytics(self) -> dict:
        """
        Get comprehensive analytics for integrated smart execution
        
        Returns:
            Dictionary with integration performance metrics
        """
        try:
            if not self.execution_history:
                return {"total_executions": 0, "message": "No executions completed yet"}

            # Calculate aggregate metrics
            total_executions = len(self.execution_history)
            avg_completion_rate = sum(r['completion_rate'] for r in self.execution_history) / total_executions
            avg_slippage = sum(r['total_slippage'] for r in self.execution_history) / total_executions
            total_cost_savings = sum(r['cost_savings_estimate'] for r in self.execution_history)

            # Strategy performance breakdown
            strategy_stats = {}
            for result in self.execution_history:
                strategy = result['strategy_used']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'count': 0,
                        'total_slippage': 0.0,
                        'total_completion': 0.0,
                        'total_savings': 0.0
                    }

                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['total_slippage'] += result['total_slippage']
                strategy_stats[strategy]['total_completion'] += result['completion_rate']
                strategy_stats[strategy]['total_savings'] += result['cost_savings_estimate']

            # Calculate strategy averages
            strategy_performance = {}
            for strategy, stats in strategy_stats.items():
                strategy_performance[strategy] = {
                    'executions': stats['count'],
                    'avg_slippage': stats['total_slippage'] / stats['count'],
                    'avg_completion_rate': stats['total_completion'] / stats['count'],
                    'total_cost_savings': stats['total_savings']
                }

            # Risk management metrics
            risk_managed_count = sum(1 for r in self.execution_history if r['risk_managed'])

            analytics = {
                'total_executions': total_executions,
                'avg_completion_rate': avg_completion_rate,
                'avg_slippage': avg_slippage,
                'total_cost_savings': total_cost_savings,
                'avg_cost_savings_per_execution': total_cost_savings / total_executions,
                'risk_managed_executions': risk_managed_count,
                'risk_management_rate': risk_managed_count / total_executions,
                'strategy_performance': strategy_performance,
                'last_execution': self.execution_history[-1]['execution_order'].end_time.isoformat()
            }

            return analytics

        except Exception as e:
            logging.error(f"❌ Error generating integration analytics: {e}")
            return {"error": str(e)}

    def is_available(self) -> bool:
        """Check if smart execution integration is available"""
        return self.initialized and SMART_EXECUTION_AVAILABLE


# Integration functions for existing trading system
def initialize_smart_execution_system(alpaca_api_key: str = None, alpaca_secret: str = None) -> SmartOrderExecutionIntegrator | None:
    """
    Initialize smart execution system for integration with existing trading system
    
    Args:
        alpaca_api_key: Alpaca API key
        alpaca_secret: Alpaca secret key
        
    Returns:
        SmartOrderExecutionIntegrator instance or None if unavailable
    """
    try:
        integrator = SmartOrderExecutionIntegrator(alpaca_api_key, alpaca_secret, paper_trading=True)

        if integrator.is_available():
            logging.info("✅ Smart Order Execution System initialized successfully")
            return integrator
        else:
            logging.warning("❌ Smart Order Execution System initialization failed")
            return None

    except Exception as e:
        logging.error(f"❌ Error initializing smart execution system: {e}")
        return None

async def execute_enhanced_order(symbol: str, quantity: float, side: str,
                               integrator: SmartOrderExecutionIntegrator = None,
                               **kwargs) -> dict:
    """
    Execute order with smart execution enhancements
    
    Args:
        symbol: Stock symbol
        quantity: Order quantity
        side: 'buy' or 'sell'
        integrator: Smart execution integrator instance
        **kwargs: Additional execution parameters
        
    Returns:
        Dictionary with execution results
    """
    if not integrator or not integrator.is_available():
        # Fallback to basic execution
        logging.warning("Smart execution not available, using basic execution")
        return {
            'success': False,
            'message': 'Smart execution not available',
            'fallback_execution': True
        }

    try:
        result = await integrator.execute_smart_order_integrated(
            symbol=symbol,
            quantity=quantity,
            side=side,
            **kwargs
        )

        return {
            'success': True,
            'execution_results': result,
            'smart_execution_used': True
        }

    except Exception as e:
        logging.error(f"❌ Enhanced order execution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'smart_execution_attempted': True
        }


# Testing function
async def test_smart_execution_integration():
    """Test the complete smart execution integration"""
    logging.info("🧪 Testing Smart Order Execution Integration")

    # Initialize integration
    integrator = initialize_smart_execution_system()

    if not integrator:
        logging.error("❌ Integration initialization failed")
        return

    # Test portfolio
    test_portfolio = {
        'AAPL': {'quantity': 200, 'value': 37000},
        'MSFT': {'quantity': 100, 'value': 42000},
        'GOOGL': {'quantity': 20, 'value': 29000}
    }

    # Test various order scenarios
    test_scenarios = [
        # (symbol, quantity, side, description)
        ("AAPL", 300, "buy", "Large order - should use IS strategy"),
        ("TSLA", 150, "buy", "Medium order - should use VWAP strategy"),
        ("MSFT", 50, "sell", "Small order - fast execution"),
        ("NVDA", 75, "buy", "High volatility stock")
    ]

    for symbol, quantity, side, description in test_scenarios:
        logging.info(f"\n{'='*60}")
        logging.info(f"🔬 Testing scenario: {description}")
        logging.info(f"   Order: {quantity} {symbol} {side}")

        try:
            result = await execute_enhanced_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                integrator=integrator,
                current_portfolio=test_portfolio,
                dry_run=True
            )

            if result['success']:
                exec_results = result['execution_results']
                logging.info("✅ Scenario completed successfully:")
                logging.info(f"   Strategy: {exec_results['strategy_used']}")
                logging.info(f"   Duration: {exec_results['duration_used']} minutes")
                logging.info(f"   Completion: {exec_results['completion_rate']:.1%}")
                logging.info(f"   Slippage: {exec_results['total_slippage']:.2%}")
                logging.info(f"   Cost Savings: ${exec_results['cost_savings_estimate']:.2f}")
            else:
                logging.error(f"❌ Scenario failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logging.error(f"❌ Scenario test failed: {e}")

    # Display comprehensive analytics
    logging.info(f"\n{'='*60}")
    logging.info("📊 SMART EXECUTION INTEGRATION ANALYTICS")

    analytics = integrator.get_integration_analytics()

    logging.info(f"Total Smart Executions: {analytics.get('total_executions', 0)}")
    logging.info(f"Average Completion Rate: {analytics.get('avg_completion_rate', 0):.1%}")
    logging.info(f"Average Slippage: {analytics.get('avg_slippage', 0):.2%}")
    logging.info(f"Total Cost Savings: ${analytics.get('total_cost_savings', 0):.2f}")
    logging.info(f"Risk Management Rate: {analytics.get('risk_management_rate', 0):.1%}")

    if analytics.get('strategy_performance'):
        logging.info("\nStrategy Performance Breakdown:")
        for strategy, perf in analytics['strategy_performance'].items():
            logging.info(f"  {strategy}:")
            logging.info(f"    Executions: {perf['executions']}")
            logging.info(f"    Avg Slippage: {perf['avg_slippage']:.2%}")
            logging.info(f"    Avg Completion: {perf['avg_completion_rate']:.1%}")
            logging.info(f"    Total Savings: ${perf['total_cost_savings']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_smart_execution_integration())
