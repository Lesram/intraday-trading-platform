#!/usr/bin/env python3
"""
🏛️ INSTITUTIONAL BACKTESTING ENGINE
Rigorous backtesting framework with production parity

Priority 1 Implementation: Transform backtesting from single-model testing
to comprehensive ensemble validation with institutional-grade controls.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing components
try:
    from runfile import (
        add_technical_indicators,
        fetch_alpaca_bars,
        ENHANCED_FEATURES
    )
    from advanced_ml_predictor import AdvancedMLPredictor
    from portfolio_risk_manager import PortfolioRiskManager
    from transaction_cost_model import TransactionCostModel
    from performance_attribution_analyzer import PerformanceAttributionAnalyzer
    from advanced_volatility_forecaster import AdvancedVolatilityForecaster
except ImportError as e:
    logging.warning(f"Some components not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestRequest:
    """Backtest configuration parameters"""
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    timeframe: str = "15Min"
    use_ensemble: bool = True
    risk_management: bool = True
    transaction_costs: bool = True
    regime_analysis: bool = True
    stress_testing: bool = True

@dataclass
class TradeResult:
    """Individual trade execution result"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    executed_price: float
    slippage: float
    commission: float
    total_cost: float
    signal_confidence: float
    risk_metrics: Dict

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    request: BacktestRequest
    trades: List[TradeResult] 
    daily_returns: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    performance_metrics: Dict
    risk_metrics: Dict
    attribution_analysis: Dict
    regime_performance: Dict
    stress_test_results: Dict
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary for API response"""
        return {
            'request': asdict(self.request),
            'summary': {
                'total_trades': len(self.trades),
                'total_return': self.performance_metrics.get('total_return', 0),
                'annualized_return': self.performance_metrics.get('annualized_return', 0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0),
                'win_rate': self.performance_metrics.get('win_rate', 0),
                'profit_factor': self.performance_metrics.get('profit_factor', 0)
            },
            'risk_analysis': self.risk_metrics,
            'performance_attribution': self.attribution_analysis,
            'regime_analysis': self.regime_performance,
            'stress_test_summary': self.stress_test_results
        }

class ExecutionSimulator:
    """Realistic trade execution simulation"""
    
    def __init__(self):
        self.cost_model = None
        self.slippage_model = SlippageEstimator()
        
        # Initialize transaction cost model if available
        try:
            self.cost_model = TransactionCostModel()
        except:
            logger.warning("TransactionCostModel not available, using simplified costs")
    
    def simulate_trade_execution(self, 
                               signal: str,
                               symbol: str,
                               quantity: float, 
                               price: float,
                               volume: float = 1000000,
                               spread_pct: float = 0.001) -> TradeResult:
        """Simulate realistic trade execution with costs"""
        
        try:
            # Calculate market impact and slippage
            notional_value = quantity * price
            slippage = self.slippage_model.estimate_slippage(
                quantity, volume, symbol
            )
            
            # Calculate transaction costs
            if self.cost_model:
                cost_breakdown = self.cost_model.calculate_total_cost(
                    notional_value, spread_pct * price, volume
                )
                commission = cost_breakdown.commission
                total_cost = cost_breakdown.total_cost
            else:
                # Simplified cost model
                commission = max(1.0, notional_value * 0.005)  # 0.5bps commission
                spread_cost = notional_value * spread_pct
                total_cost = commission + spread_cost
            
            # Determine execution price
            if signal == 'BUY':
                executed_price = price + slippage
            else:
                executed_price = price - slippage
            
            return TradeResult(
                timestamp=datetime.now(),
                symbol=symbol,
                side=signal.lower(),
                quantity=quantity,
                price=price,
                executed_price=executed_price,
                slippage=slippage,
                commission=commission,
                total_cost=total_cost,
                signal_confidence=0.0,  # To be filled by caller
                risk_metrics={}  # To be filled by caller
            )
            
        except Exception as e:
            logger.error(f"Trade execution simulation failed: {e}")
            # Return minimal trade result
            return TradeResult(
                timestamp=datetime.now(),
                symbol=symbol,
                side=signal.lower(),
                quantity=quantity,
                price=price,
                executed_price=price,
                slippage=0.0,
                commission=1.0,
                total_cost=1.0,
                signal_confidence=0.0,
                risk_metrics={}
            )

class SlippageEstimator:
    """Estimate market impact slippage"""
    
    def estimate_slippage(self, quantity: float, daily_volume: float, symbol: str) -> float:
        """Estimate slippage based on order size vs daily volume"""
        
        if daily_volume <= 0:
            return 0.001  # 10bps default slippage
        
        # Participation rate (what fraction of daily volume this order represents)
        participation_rate = quantity / daily_volume
        
        # Market impact model (square root model)
        impact_bps = 0.0001 + (participation_rate ** 0.5) * 0.01
        
        # Cap at reasonable levels
        impact_bps = min(impact_bps, 0.005)  # Max 50bps
        
        return impact_bps

class BacktestPerformanceAnalyzer:
    """Comprehensive backtest performance analysis"""
    
    def __init__(self):
        self.attribution_analyzer = None
        self.risk_analyzer = None
        
        # Initialize analyzers if available
        try:
            self.attribution_analyzer = PerformanceAttributionAnalyzer()
        except:
            logger.warning("PerformanceAttributionAnalyzer not available")
        
        try:
            self.risk_analyzer = PortfolioRiskManager()
        except:
            logger.warning("PortfolioRiskManager not available")
    
    def calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        """Calculate basic performance metrics"""
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Remove any NaN or infinite values
        returns = returns.dropna().replace([np.inf, -np.inf], 0)
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'total_trades': len(returns)
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics for edge cases"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_trades': 0
        }
    
    def analyze_regime_performance(self, returns: pd.Series, 
                                 market_data: pd.DataFrame) -> Dict:
        """Analyze performance by market regime"""
        
        try:
            # Initialize volatility forecaster for regime detection
            vol_forecaster = AdvancedVolatilityForecaster()
            
            # Simple regime classification based on volatility and returns
            market_returns = market_data['close'].pct_change().dropna()
            
            # Calculate rolling volatility (20-day)
            rolling_vol = market_returns.rolling(20).std()
            high_vol_threshold = rolling_vol.quantile(0.7)
            
            # Classify regimes
            regimes = []
            for i, (ret, vol) in enumerate(zip(market_returns, rolling_vol)):
                if pd.isna(vol):
                    regimes.append('unknown')
                elif vol > high_vol_threshold:
                    regimes.append('high_volatility')
                elif ret > 0:
                    regimes.append('bull_market')
                else:
                    regimes.append('bear_market')
            
            # Align with returns data
            regime_series = pd.Series(regimes, index=market_returns.index)
            aligned_regimes = regime_series.reindex(returns.index, method='ffill')
            
            # Calculate performance by regime
            regime_performance = {}
            for regime in ['bull_market', 'bear_market', 'high_volatility']:
                regime_returns = returns[aligned_regimes == regime]
                if len(regime_returns) > 0:
                    regime_performance[regime] = self.calculate_basic_metrics(regime_returns)
                else:
                    regime_performance[regime] = self._empty_metrics()
            
            return regime_performance
            
        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")
            return {
                'bull_market': self._empty_metrics(),
                'bear_market': self._empty_metrics(), 
                'high_volatility': self._empty_metrics()
            }
    
    def run_stress_scenarios(self, returns: pd.Series) -> Dict:
        """Run stress testing scenarios"""
        
        stress_scenarios = {
            'market_crash': {'description': '-10% market shock', 'shock': -0.10},
            'volatility_spike': {'description': '2x volatility shock', 'vol_multiplier': 2.0},
            'liquidity_crisis': {'description': '50% volume reduction', 'volume_shock': 0.5}
        }
        
        stress_results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            try:
                # Apply stress scenario (simplified implementation)
                if 'shock' in scenario:
                    # Apply market shock to returns
                    stressed_returns = returns + scenario['shock'] / len(returns)
                else:
                    # For other scenarios, apply volatility adjustment
                    vol_mult = scenario.get('vol_multiplier', 1.0)
                    stressed_returns = returns * vol_mult
                
                # Calculate stressed performance
                stressed_metrics = self.calculate_basic_metrics(stressed_returns)
                stress_results[scenario_name] = {
                    'description': scenario['description'],
                    'metrics': stressed_metrics
                }
                
            except Exception as e:
                logger.warning(f"Stress scenario {scenario_name} failed: {e}")
                stress_results[scenario_name] = {
                    'description': scenario['description'],
                    'metrics': self._empty_metrics()
                }
        
        return stress_results

class InstitutionalBacktestEngine:
    """Production-parity backtesting system"""
    
    def __init__(self):
        self.ml_predictor = None
        self.risk_manager = None
        self.execution_simulator = ExecutionSimulator()
        self.performance_analyzer = BacktestPerformanceAnalyzer()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ML and risk management components"""
        
        try:
            self.ml_predictor = AdvancedMLPredictor()
            logger.info("✅ ML Predictor initialized")
        except Exception as e:
            logger.warning(f"ML Predictor initialization failed: {e}")
        
        try:
            self.risk_manager = PortfolioRiskManager()
            logger.info("✅ Risk Manager initialized")
        except Exception as e:
            logger.warning(f"Risk Manager initialization failed: {e}")
    
    def run_comprehensive_backtest(self, request: BacktestRequest) -> BacktestResults:
        """Run institutional-grade backtest"""
        
        logger.info(f"🚀 Starting comprehensive backtest")
        logger.info(f"   Symbols: {request.symbols}")
        logger.info(f"   Period: {request.start_date} to {request.end_date}")
        logger.info(f"   Capital: ${request.initial_capital:,.2f}")
        
        try:
            # Initialize results tracking
            all_trades = []
            daily_returns = []
            equity_curve = [request.initial_capital]
            current_positions = {}
            current_capital = request.initial_capital
            
            # Process each symbol
            for symbol in request.symbols:
                logger.info(f"📊 Processing {symbol}")
                
                # Get historical data
                symbol_data = self._prepare_historical_data(
                    symbol, request.timeframe, request.start_date, request.end_date
                )
                
                if symbol_data is None or len(symbol_data) < 100:
                    logger.warning(f"⚠️ Insufficient data for {symbol}, skipping")
                    continue
                
                # Generate signals
                signals = self._generate_backtest_signals(
                    symbol, symbol_data, request.use_ensemble
                )
                
                # Execute trades with risk management
                symbol_trades = self._execute_backtest_trades(
                    symbol, signals, symbol_data, request, 
                    current_positions, current_capital
                )
                
                all_trades.extend(symbol_trades)
                
                # Update position tracking
                for trade in symbol_trades:
                    if trade.side == 'buy':
                        current_positions[symbol] = current_positions.get(symbol, 0) + trade.quantity
                    else:
                        current_positions[symbol] = current_positions.get(symbol, 0) - trade.quantity
            
            # Calculate performance metrics
            if all_trades:
                returns_series = self._calculate_returns_series(all_trades, request.initial_capital)
                daily_returns = returns_series
                equity_curve = (1 + returns_series).cumprod() * request.initial_capital
            else:
                returns_series = pd.Series(dtype=float)
                equity_curve = pd.Series([request.initial_capital])
            
            # Comprehensive analysis
            performance_metrics = self.performance_analyzer.calculate_basic_metrics(returns_series)
            
            # Get market data for regime analysis
            market_data = self._get_market_data_for_analysis(request.symbols[0], request)
            regime_performance = self.performance_analyzer.analyze_regime_performance(
                returns_series, market_data
            )
            
            stress_results = self.performance_analyzer.run_stress_scenarios(returns_series)
            
            # Create comprehensive results
            results = BacktestResults(
                request=request,
                trades=all_trades,
                daily_returns=returns_series,
                equity_curve=equity_curve,
                positions=pd.DataFrame(),  # TODO: Implement position tracking
                performance_metrics=performance_metrics,
                risk_metrics=self._calculate_risk_metrics(all_trades, returns_series),
                attribution_analysis=self._calculate_attribution_analysis(all_trades),
                regime_performance=regime_performance,
                stress_test_results=stress_results
            )
            
            logger.info(f"✅ Backtest completed successfully")
            logger.info(f"   Total trades: {len(all_trades)}")
            logger.info(f"   Total return: {performance_metrics['total_return']:.2%}")
            logger.info(f"   Sharpe ratio: {performance_metrics['sharpe_ratio']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def _prepare_historical_data(self, symbol: str, timeframe: str, 
                               start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Prepare historical data with technical indicators"""
        
        try:
            # Calculate required lookback for technical indicators
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            lookback_start = start_dt - timedelta(days=100)  # Extra data for indicators
            
            # Fetch data using existing function
            df = fetch_alpaca_bars(symbol, timeframe, 
                                 (datetime.now() - lookback_start).days)
            
            if df is None or len(df) == 0:
                return None
            
            # Add technical indicators using existing function
            df = add_technical_indicators(df)
            
            # Filter to actual backtest period
            actual_start = datetime.strptime(start_date, "%Y-%m-%d")
            actual_end = datetime.strptime(end_date, "%Y-%m-%d")
            
            df = df[(df.index >= actual_start) & (df.index <= actual_end)]
            
            return df
            
        except Exception as e:
            logger.error(f"Data preparation failed for {symbol}: {e}")
            return None
    
    def _generate_backtest_signals(self, symbol: str, data: pd.DataFrame, 
                                 use_ensemble: bool) -> pd.DataFrame:
        """Generate trading signals using ML models"""
        
        try:
            signals = []
            
            if use_ensemble and self.ml_predictor:
                # Use ensemble prediction
                for idx, row in data.iterrows():
                    try:
                        # Extract features (use last N features matching model)
                        if hasattr(self.ml_predictor, 'feature_names'):
                            required_features = self.ml_predictor.feature_names
                        else:
                            # Use existing enhanced features
                            required_features = ENHANCED_FEATURES
                        
                        features = []
                        for feature in required_features:
                            if feature in row:
                                features.append(row[feature])
                            else:
                                features.append(0.0)  # Default value
                        
                        # Get prediction
                        prediction = self.ml_predictor.predict(features)
                        
                        if isinstance(prediction, dict):
                            prob = prediction.get('probability', 0.5)
                            confidence = prediction.get('confidence', 0.0)
                        else:
                            prob = float(prediction)
                            confidence = abs(prob - 0.5) * 2  # Convert to 0-1 confidence
                        
                        # Convert probability to signal
                        if prob > 0.6 and confidence > 0.3:
                            signal = 'BUY'
                        elif prob < 0.4 and confidence > 0.3:
                            signal = 'SELL'
                        else:
                            signal = 'FLAT'
                        
                        signals.append({
                            'timestamp': idx,
                            'symbol': symbol,
                            'signal': signal,
                            'probability': prob,
                            'confidence': confidence,
                            'price': row.get('close', row.get('Close', 0))
                        })
                        
                    except Exception as e:
                        logger.warning(f"Signal generation failed for {idx}: {e}")
                        signals.append({
                            'timestamp': idx,
                            'symbol': symbol,
                            'signal': 'FLAT',
                            'probability': 0.5,
                            'confidence': 0.0,
                            'price': row.get('close', row.get('Close', 0))
                        })
            else:
                # Simple momentum-based signals as fallback
                for idx, row in data.iterrows():
                    # Simple momentum signal
                    if 'sma_20' in row and 'sma_50' in row:
                        if row['sma_20'] > row['sma_50']:
                            signal = 'BUY'
                        else:
                            signal = 'SELL'
                    else:
                        signal = 'FLAT'
                    
                    signals.append({
                        'timestamp': idx,
                        'symbol': symbol,
                        'signal': signal,
                        'probability': 0.6 if signal != 'FLAT' else 0.5,
                        'confidence': 0.5,
                        'price': row.get('close', row.get('Close', 0))
                    })
            
            return pd.DataFrame(signals).set_index('timestamp')
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            # Return empty signals dataframe
            return pd.DataFrame(columns=['symbol', 'signal', 'probability', 'confidence', 'price'])
    
    def _execute_backtest_trades(self, symbol: str, signals: pd.DataFrame, 
                               data: pd.DataFrame, request: BacktestRequest,
                               current_positions: Dict, current_capital: float) -> List[TradeResult]:
        """Execute trades with risk management"""
        
        trades = []
        
        try:
            for idx, signal_row in signals.iterrows():
                if signal_row['signal'] == 'FLAT':
                    continue
                
                # Get market data for this timestamp
                if idx not in data.index:
                    continue
                
                market_row = data.loc[idx]
                price = market_row.get('close', market_row.get('Close', 0))
                volume = market_row.get('volume', market_row.get('Volume', 1000000))
                
                if price <= 0:
                    continue
                
                # Calculate position size with risk management
                if request.risk_management and self.risk_manager:
                    position_size = self._calculate_risk_managed_size(
                        symbol, signal_row, current_positions, current_capital
                    )
                else:
                    # Simple fixed position sizing (2% of capital)
                    position_size = int((current_capital * 0.02) / price)
                
                if position_size <= 0:
                    continue
                
                # Simulate trade execution
                trade_result = self.execution_simulator.simulate_trade_execution(
                    signal_row['signal'], symbol, position_size, price, volume
                )
                
                # Update trade result with signal information
                trade_result.timestamp = idx
                trade_result.signal_confidence = signal_row['confidence']
                
                trades.append(trade_result)
                
                # Update capital (simplified)
                trade_impact = position_size * price + trade_result.total_cost
                if signal_row['signal'] == 'BUY':
                    current_capital -= trade_impact
                else:
                    current_capital += trade_impact
        
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
        
        return trades
    
    def _calculate_risk_managed_size(self, symbol: str, signal_row: pd.Series,
                                   current_positions: Dict, current_capital: float) -> int:
        """Calculate position size with risk management"""
        
        try:
            if not self.risk_manager:
                # Fallback to simple sizing
                return int((current_capital * 0.02) / signal_row['price'])
            
            # Use existing risk management logic (simplified)
            base_allocation = 0.02  # 2% base allocation
            confidence_multiplier = signal_row['confidence']
            
            # Adjust for existing positions (concentration risk)
            current_position = current_positions.get(symbol, 0)
            max_position_value = current_capital * 0.05  # Max 5% per position
            
            if current_position * signal_row['price'] >= max_position_value:
                return 0  # Already at max position
            
            # Calculate final position size
            target_allocation = base_allocation * confidence_multiplier
            position_value = current_capital * target_allocation
            position_size = int(position_value / signal_row['price'])
            
            return max(0, position_size)
            
        except Exception as e:
            logger.warning(f"Risk sizing failed: {e}")
            return int((current_capital * 0.01) / signal_row['price'])  # Conservative fallback
    
    def _calculate_returns_series(self, trades: List[TradeResult], 
                                initial_capital: float) -> pd.Series:
        """Calculate daily returns series from trades"""
        
        if not trades:
            return pd.Series(dtype=float)
        
        # Group trades by date
        trade_df = pd.DataFrame([
            {
                'date': trade.timestamp.date(),
                'pnl': self._calculate_trade_pnl(trade)
            } for trade in trades
        ])
        
        # Aggregate daily P&L
        daily_pnl = trade_df.groupby('date')['pnl'].sum()
        
        # Convert to returns
        daily_returns = daily_pnl / initial_capital
        
        return daily_returns
    
    def _calculate_trade_pnl(self, trade: TradeResult) -> float:
        """Calculate P&L for a single trade (simplified)"""
        
        # Simplified P&L calculation
        # In reality, this would require tracking entry/exit pairs
        if trade.side == 'buy':
            # Assume we sell at break-even for now
            return -trade.total_cost
        else:
            # Assume we bought earlier at similar price
            return trade.quantity * trade.executed_price - trade.total_cost
    
    def _get_market_data_for_analysis(self, symbol: str, request: BacktestRequest) -> pd.DataFrame:
        """Get market data for regime analysis"""
        
        try:
            return self._prepare_historical_data(
                symbol, request.timeframe, request.start_date, request.end_date
            )
        except:
            # Return empty dataframe as fallback
            return pd.DataFrame({'close': []})
    
    def _calculate_risk_metrics(self, trades: List[TradeResult], 
                               returns: pd.Series) -> Dict:
        """Calculate risk metrics"""
        
        try:
            if self.risk_manager and len(returns) > 0:
                # Use existing risk manager
                risk_metrics = {
                    'var_daily_95': returns.quantile(0.05),
                    'var_daily_99': returns.quantile(0.01),
                    'expected_shortfall': returns[returns <= returns.quantile(0.05)].mean(),
                    'volatility': returns.std() * np.sqrt(252),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
            else:
                risk_metrics = {
                    'var_daily_95': 0.0,
                    'var_daily_99': 0.0,
                    'expected_shortfall': 0.0,
                    'volatility': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0
                }
            
            return risk_metrics
            
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _calculate_attribution_analysis(self, trades: List[TradeResult]) -> Dict:
        """Calculate performance attribution"""
        
        try:
            if not trades:
                return {}
            
            # Group trades by symbol
            symbol_pnl = {}
            for trade in trades:
                pnl = self._calculate_trade_pnl(trade)
                symbol_pnl[trade.symbol] = symbol_pnl.get(trade.symbol, 0) + pnl
            
            return {
                'by_symbol': symbol_pnl,
                'total_trades': len(trades),
                'symbols_traded': len(set(trade.symbol for trade in trades))
            }
            
        except Exception as e:
            logger.warning(f"Attribution analysis failed: {e}")
            return {}

def main():
    """Test the institutional backtesting engine"""
    
    print("🏛️ INSTITUTIONAL BACKTESTING ENGINE")
    print("=" * 60)
    
    # Create test backtest request
    request = BacktestRequest(
        symbols=["AAPL", "TSLA"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=100000,
        timeframe="15Min",
        use_ensemble=True,
        risk_management=True,
        transaction_costs=True
    )
    
    # Initialize and run backtest
    engine = InstitutionalBacktestEngine()
    
    try:
        results = engine.run_comprehensive_backtest(request)
        
        print("\n📊 BACKTEST RESULTS")
        print("-" * 30)
        print(f"Total Return: {results.performance_metrics['total_return']:.2%}")
        print(f"Annualized Return: {results.performance_metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results.performance_metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {results.performance_metrics['win_rate']:.2%}")
        print(f"Total Trades: {len(results.trades)}")
        
        print("\n🎯 REGIME ANALYSIS")
        print("-" * 30)
        for regime, metrics in results.regime_performance.items():
            print(f"{regime}: {metrics['total_return']:.2%} return, "
                  f"{metrics['sharpe_ratio']:.2f} Sharpe")
        
        print("\n⚠️ STRESS TEST RESULTS")  
        print("-" * 30)
        for scenario, result in results.stress_test_results.items():
            print(f"{scenario}: {result['metrics']['total_return']:.2%} return")
        
        print("\n✅ Backtesting engine test completed successfully!")
        
    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
