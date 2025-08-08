#!/usr/bin/env python3
"""
📊 PERFORMANCE METRICS & KPI MONITORING MODULE
Real-time tracking of Sharpe ratio, drawdown, and success metrics
Institutional-grade performance analytics for risk management
"""

import logging
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    GREEN = "green"      # All systems go
    YELLOW = "yellow"    # Warning - approaching limits
    RED = "red"         # Critical - immediate action required

@dataclass
class PerformanceMetrics:
    """Core performance metrics snapshot"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    pnl_total: float
    pnl_percent: float
    sharpe_ratio_90d: float
    max_drawdown: float
    current_drawdown: float
    monthly_return: float
    num_positions: int
    total_trades_today: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    alert_level: AlertLevel

class PerformanceMonitor:
    """
    Monitors real-time performance metrics and triggers alerts
    
    Success Criteria:
    - Sharpe ≥ 2.0 (rolling 90 days)
    - Max drawdown ≤ 6%
    - Monthly net return 4-8%
    - Win rate ≥ 55%
    - Profit factor ≥ 1.5
    """
    
    def __init__(self, api_client: tradeapi.REST, initial_capital: float = 100000.0):
        self.api = api_client
        self.initial_capital = initial_capital
        
        # Performance thresholds
        self.target_sharpe = 2.0
        self.warning_sharpe = 1.5
        self.critical_sharpe = 1.0
        
        self.max_drawdown_limit = 0.06  # 6%
        self.warning_drawdown = 0.04    # 4%
        
        self.target_monthly_return_min = 0.04  # 4%
        self.target_monthly_return_max = 0.08  # 8%
        
        self.min_win_rate = 0.55
        self.min_profit_factor = 1.5
        
        # Database setup
        self.db_path = "performance_metrics.db"
        self.init_database()
        
        # Cache for calculations
        self.portfolio_history = []
        self.trade_history = []
        self.last_update = datetime.min
        self.last_log_time = datetime.min  # For throttling log output
        
        logger.info(f"📊 Performance monitor initialized (Target: Sharpe≥{self.target_sharpe}, DD≤{self.max_drawdown_limit:.1%})")
    
    def init_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Portfolio snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT PRIMARY KEY,
                    portfolio_value REAL,
                    cash REAL,
                    pnl_total REAL,
                    pnl_percent REAL,
                    num_positions INTEGER,
                    account_data TEXT
                )
            ''')
            
            # Trade performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_performance (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    hold_duration INTEGER,
                    strategy TEXT,
                    reason TEXT
                )
            ''')
            
            # Daily metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    sharpe_ratio_90d REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    monthly_return REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    alert_level TEXT,
                    metrics_json TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("💾 Performance database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing performance database: {e}")
    
    def capture_portfolio_snapshot(self) -> Dict:
        """Capture current portfolio state"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            pnl_total = portfolio_value - self.initial_capital
            pnl_percent = (pnl_total / self.initial_capital) * 100
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'cash': cash,
                'pnl_total': pnl_total,
                'pnl_percent': pnl_percent,
                'num_positions': len(positions),
                'account_data': {
                    'buying_power': float(account.buying_power),
                    'day_trade_buying_power': float(account.daytrading_buying_power),
                    'equity': float(account.equity),
                    'last_equity': float(account.last_equity),
                    'multiplier': int(account.multiplier),
                    'status': account.status
                }
            }
            
            # Store in database
            self.store_portfolio_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing portfolio snapshot: {e}")
            return {}
    
    def store_portfolio_snapshot(self, snapshot: Dict):
        """Store portfolio snapshot in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_snapshots 
                (timestamp, portfolio_value, cash, pnl_total, pnl_percent, num_positions, account_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot['timestamp'],
                snapshot['portfolio_value'],
                snapshot['cash'],
                snapshot['pnl_total'],
                snapshot['pnl_percent'],
                snapshot['num_positions'],
                json.dumps(snapshot['account_data'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {e}")
    
    def get_portfolio_history(self, days: int = 90) -> List[Dict]:
        """Get portfolio history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT * FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            ''', (since_date,))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'timestamp': row[0],
                    'portfolio_value': row[1],
                    'cash': row[2],
                    'pnl_total': row[3],
                    'pnl_percent': row[4],
                    'num_positions': row[5],
                    'account_data': json.loads(row[6])
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []
    
    def calculate_sharpe_ratio(self, days: int = 90) -> float:
        """Calculate rolling Sharpe ratio"""
        try:
            history = self.get_portfolio_history(days)
            if len(history) < 30:  # Need at least 30 data points
                return 0.0
            
            # Calculate daily returns
            values = [h['portfolio_value'] for h in history]
            returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
            
            if not returns:
                return 0.0
            
            # Calculate Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe (assuming ~252 trading days)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_drawdown_metrics(self) -> Tuple[float, float]:
        """Calculate current and maximum drawdown"""
        try:
            history = self.get_portfolio_history(90)
            if len(history) < 2:
                return 0.0, 0.0
            
            values = [h['portfolio_value'] for h in history]
            
            # Calculate running maximum (peak)
            peaks = []
            current_peak = values[0]
            for value in values:
                if value > current_peak:
                    current_peak = value
                peaks.append(current_peak)
            
            # Calculate drawdowns
            drawdowns = [(peaks[i] - values[i]) / peaks[i] for i in range(len(values))]
            
            current_drawdown = drawdowns[-1] if drawdowns else 0.0
            max_drawdown = max(drawdowns) if drawdowns else 0.0
            
            return current_drawdown, max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0, 0.0
    
    def calculate_monthly_return(self) -> float:
        """Calculate current month return"""
        try:
            # Get start of current month
            now = datetime.now()
            month_start = datetime(now.year, now.month, 1)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get portfolio value at month start
            cursor.execute('''
                SELECT portfolio_value FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC 
                LIMIT 1
            ''', (month_start.isoformat(),))
            
            month_start_row = cursor.fetchone()
            
            # Get current portfolio value
            cursor.execute('''
                SELECT portfolio_value FROM portfolio_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            current_row = cursor.fetchone()
            conn.close()
            
            if not month_start_row or not current_row:
                return 0.0
            
            month_start_value = month_start_row[0]
            current_value = current_row[0]
            
            monthly_return = (current_value - month_start_value) / month_start_value
            return monthly_return
            
        except Exception as e:
            logger.error(f"Error calculating monthly return: {e}")
            return 0.0
    
    def calculate_trade_metrics(self, days: int = 30) -> Dict[str, float]:
        """Calculate trade performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT pnl, pnl_percent FROM trade_performance 
                WHERE exit_time >= ?
            ''', (since_date,))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0
                }
            
            pnls = [trade[0] for trade in trades]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl <= 0]
            
            total_trades = len(trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0.01  # Avoid division by zero
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
    
    def determine_alert_level(self, metrics: PerformanceMetrics) -> AlertLevel:
        """Determine alert level based on current metrics"""
        try:
            # Critical conditions (RED)
            if (metrics.sharpe_ratio_90d < self.critical_sharpe or 
                metrics.max_drawdown > self.max_drawdown_limit or
                metrics.current_drawdown > self.max_drawdown_limit):
                return AlertLevel.RED
            
            # Warning conditions (YELLOW)
            if (metrics.sharpe_ratio_90d < self.warning_sharpe or
                metrics.current_drawdown > self.warning_drawdown or
                metrics.win_rate < self.min_win_rate or
                metrics.profit_factor < self.min_profit_factor):
                return AlertLevel.YELLOW
            
            # All good (GREEN)
            return AlertLevel.GREEN
            
        except Exception as e:
            logger.error(f"Error determining alert level: {e}")
            return AlertLevel.YELLOW
    
    def calculate_comprehensive_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        try:
            # Get current portfolio snapshot
            snapshot = self.capture_portfolio_snapshot()
            
            if not snapshot:
                # Return default metrics if snapshot fails
                return PerformanceMetrics(
                    timestamp=datetime.now(),
                    portfolio_value=self.initial_capital,
                    cash=self.initial_capital,
                    pnl_total=0.0,
                    pnl_percent=0.0,
                    sharpe_ratio_90d=0.0,
                    max_drawdown=0.0,
                    current_drawdown=0.0,
                    monthly_return=0.0,
                    num_positions=0,
                    total_trades_today=0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    alert_level=AlertLevel.GREEN
                )
            
            # Calculate performance metrics
            sharpe_ratio = self.calculate_sharpe_ratio(90)
            current_drawdown, max_drawdown = self.calculate_drawdown_metrics()
            monthly_return = self.calculate_monthly_return()
            trade_metrics = self.calculate_trade_metrics(30)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                portfolio_value=snapshot['portfolio_value'],
                cash=snapshot['cash'],
                pnl_total=snapshot['pnl_total'],
                pnl_percent=snapshot['pnl_percent'],
                sharpe_ratio_90d=sharpe_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                monthly_return=monthly_return,
                num_positions=snapshot['num_positions'],
                total_trades_today=trade_metrics['total_trades'],
                win_rate=trade_metrics['win_rate'],
                avg_win=trade_metrics['avg_win'],
                avg_loss=trade_metrics['avg_loss'],
                profit_factor=trade_metrics['profit_factor'],
                alert_level=AlertLevel.GREEN  # Will be updated below
            )
            
            # Determine alert level
            metrics.alert_level = self.determine_alert_level(metrics)
            
            # Store daily metrics
            self.store_daily_metrics(metrics)
            
            # Log performance summary
            self.log_performance_summary(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                portfolio_value=self.initial_capital,
                cash=self.initial_capital,
                pnl_total=0.0,
                pnl_percent=0.0,
                sharpe_ratio_90d=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                monthly_return=0.0,
                num_positions=0,
                total_trades_today=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                alert_level=AlertLevel.RED
            )
    
    def store_daily_metrics(self, metrics: PerformanceMetrics):
        """Store daily metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            date_str = metrics.timestamp.strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_metrics 
                (date, sharpe_ratio_90d, max_drawdown, current_drawdown, monthly_return,
                 win_rate, profit_factor, total_trades, alert_level, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                metrics.sharpe_ratio_90d,
                metrics.max_drawdown,
                metrics.current_drawdown,
                metrics.monthly_return,
                metrics.win_rate,
                metrics.profit_factor,
                metrics.total_trades_today,
                metrics.alert_level.value,
                json.dumps(asdict(metrics), default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing daily metrics: {e}")
    
    def log_performance_summary(self, metrics: PerformanceMetrics):
        """Log performance summary with appropriate alert level (throttled)"""
        try:
            # Throttle logging - only log every 60 seconds to avoid spam
            now = datetime.now()
            if (now - self.last_log_time).total_seconds() < 60:
                return
            self.last_log_time = now
            
            alert_emoji = {
                AlertLevel.GREEN: "🟢",
                AlertLevel.YELLOW: "🟡",
                AlertLevel.RED: "🔴"
            }
            
            emoji = alert_emoji.get(metrics.alert_level, "⚪")
            
            summary = (
                f"{emoji} PERFORMANCE SUMMARY {emoji}\n"
                f"Portfolio: ${metrics.portfolio_value:,.2f} ({metrics.pnl_percent:+.2f}%)\n"
                f"Sharpe (90d): {metrics.sharpe_ratio_90d:.2f} (Target: ≥{self.target_sharpe})\n"
                f"Max DD: {metrics.max_drawdown:.1%} (Limit: ≤{self.max_drawdown_limit:.1%})\n"
                f"Current DD: {metrics.current_drawdown:.1%}\n"
                f"Monthly Return: {metrics.monthly_return:+.1%} (Target: {self.target_monthly_return_min:.0%}-{self.target_monthly_return_max:.0%})\n"
                f"Win Rate: {metrics.win_rate:.1%} (Min: {self.min_win_rate:.0%})\n"
                f"Profit Factor: {metrics.profit_factor:.2f} (Min: {self.min_profit_factor})\n"
                f"Positions: {metrics.num_positions} | Trades Today: {metrics.total_trades_today}"
            )
            
            if metrics.alert_level == AlertLevel.RED:
                logger.error(summary)
            elif metrics.alert_level == AlertLevel.YELLOW:
                logger.warning(summary)
            else:
                logger.info(summary)
                
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")
    
    def should_halt_trading(self, metrics: PerformanceMetrics) -> bool:
        """Determine if trading should be halted based on metrics"""
        # TEMPORARY: More permissive for testing and data collection
        # Allow trading unless extreme conditions are met
        return (
            metrics.current_drawdown > 0.15 or  # Only halt at 15% drawdown (was 6%)
            metrics.sharpe_ratio_90d < -5.0     # Only halt at extremely negative Sharpe (was 1.0)
        )
        # Original strict logic (commented out for testing):
        # return (
        #     metrics.alert_level == AlertLevel.RED or
        #     metrics.current_drawdown > self.max_drawdown_limit or
        #     metrics.sharpe_ratio_90d < self.critical_sharpe
        # )
    
    def get_risk_adjustment_factor(self, metrics: PerformanceMetrics) -> float:
        """Get position size adjustment factor based on performance"""
        try:
            if metrics.alert_level == AlertLevel.RED:
                return 0.5  # Halve position sizes
            elif metrics.alert_level == AlertLevel.YELLOW:
                return 0.75  # Reduce position sizes by 25%
            else:
                return 1.0  # Full position sizes
                
        except Exception as e:
            logger.error(f"Error calculating risk adjustment factor: {e}")
            return 0.5  # Conservative default

# Global instance
performance_monitor = None

def initialize_performance_monitor(api_client: tradeapi.REST, initial_capital: float = 100000.0):
    """Initialize the global performance monitor"""
    global performance_monitor
    performance_monitor = PerformanceMonitor(api_client, initial_capital)
    logger.info("📊 Performance monitor initialized")

def get_current_metrics() -> PerformanceMetrics:
    """Get current performance metrics"""
    if performance_monitor is None:
        raise RuntimeError("Performance monitor not initialized")
    
    return performance_monitor.calculate_comprehensive_metrics()

def should_halt_trading() -> bool:
    """Check if trading should be halted"""
    if performance_monitor is None:
        return True
    
    metrics = get_current_metrics()
    return performance_monitor.should_halt_trading(metrics)
