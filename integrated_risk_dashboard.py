#!/usr/bin/env python3
"""
📊 INTEGRATED RISK DASHBOARD
Phase 1 Optimization: Combined risk management and stop-loss monitoring dashboard
Part of Audit Item 4: Trading Strategy Reevaluation
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import asyncio

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our unified systems
from unified_risk_manager import get_risk_manager, PositionRisk, RiskLevel
from strategy_stop_losses import get_stop_loss_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RiskDashboardData:
    """Combined risk dashboard data"""
    timestamp: str
    overall_status: str
    risk_level: str
    portfolio_summary: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    stop_loss_status: Dict[str, Any]
    strategy_performance: Dict[str, Any]
    alerts: List[str]
    recommendations: List[str]

class IntegratedRiskDashboard:
    """Comprehensive risk management and stop-loss dashboard"""
    
    def __init__(self):
        self.risk_manager = get_risk_manager()
        self.stop_loss_manager = get_stop_loss_manager()
        self.dashboard_history = []
        
        logger.info("📊 Integrated Risk Dashboard initialized")
    
    async def generate_dashboard_data(self, positions: List[PositionRisk], 
                                    account_value: float, market_data: Dict[str, float]) -> RiskDashboardData:
        """Generate comprehensive dashboard data"""
        
        # Get risk metrics from unified risk manager
        risk_metrics = self.risk_manager.assess_portfolio_risk(positions, account_value)
        risk_dashboard = self.risk_manager.get_risk_dashboard()
        
        # Get stop-loss status
        stop_loss_dashboard = self.stop_loss_manager.get_stop_loss_dashboard()
        
        # Check for triggered stops
        triggered_stops = await self.stop_loss_manager.check_stop_triggers(market_data)
        
        # Calculate overall status
        overall_status = self._determine_overall_status(risk_metrics, len(triggered_stops))
        
        # Generate portfolio summary
        portfolio_summary = self._generate_portfolio_summary(positions, account_value, market_data)
        
        # Get strategy performance
        strategy_performance = self._analyze_strategy_performance(positions)
        
        # Combine all alerts
        all_alerts = risk_dashboard.get('alerts', [])
        if triggered_stops:
            all_alerts.extend([f"🛑 Stop-loss triggered: {stop.symbol}" for stop in triggered_stops])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_metrics, stop_loss_dashboard, portfolio_summary)
        
        # Create dashboard data
        dashboard_data = RiskDashboardData(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            risk_level=risk_metrics.risk_level.value,
            portfolio_summary=portfolio_summary,
            risk_metrics={
                "portfolio_heat": risk_dashboard['portfolio_heat'],
                "concentration_risk": risk_dashboard['concentration_risk'],
                "daily_var": risk_dashboard['daily_var'],
                "portfolio_volatility": risk_dashboard['portfolio_volatility'],
                "leverage_ratio": risk_dashboard['leverage_ratio']
            },
            stop_loss_status={
                "active_stops": stop_loss_dashboard['active_stops'],
                "triggered_today": stop_loss_dashboard['triggered_today'],
                "strategy_stats": stop_loss_dashboard['strategy_stats']
            },
            strategy_performance=strategy_performance,
            alerts=all_alerts,
            recommendations=recommendations
        )
        
        # Store in history
        self.dashboard_history.append(dashboard_data)
        
        # Keep only last 100 entries
        if len(self.dashboard_history) > 100:
            self.dashboard_history = self.dashboard_history[-100:]
        
        return dashboard_data
    
    def _determine_overall_status(self, risk_metrics, triggered_stops_count: int) -> str:
        """Determine overall portfolio status"""
        
        if triggered_stops_count > 0:
            return "ALERT - Stops Triggered"
        elif risk_metrics.risk_level == RiskLevel.CRITICAL:
            return "CRITICAL - High Risk"
        elif risk_metrics.risk_level == RiskLevel.HIGH:
            return "WARNING - Elevated Risk"
        elif risk_metrics.portfolio_heat > 0.2:
            return "CAUTION - High Heat"
        else:
            return "HEALTHY - Normal Operations"
    
    def _generate_portfolio_summary(self, positions: List[PositionRisk], 
                                  account_value: float, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate portfolio summary statistics"""
        
        if not positions:
            return {
                "total_positions": 0,
                "total_value": f"${account_value:,.0f}",
                "largest_position": "None",
                "cash_percentage": "100.0%"
            }
        
        total_position_value = sum(pos.position_value for pos in positions)
        largest_position = max(positions, key=lambda p: p.position_value)
        
        # Current P&L estimation
        total_unrealized_pnl = 0
        for pos in positions:
            if pos.symbol in market_data:
                current_price = market_data[pos.symbol]
                shares = pos.position_value / (current_price * 0.99)  # Rough share estimate
                unrealized_pnl = shares * (current_price - (pos.position_value / shares))
                total_unrealized_pnl += unrealized_pnl
        
        return {
            "total_positions": len(positions),
            "total_invested": f"${total_position_value:,.0f}",
            "total_value": f"${account_value:,.0f}",
            "cash_percentage": f"{((account_value - total_position_value) / account_value * 100):.1f}%",
            "largest_position": f"{largest_position.symbol} ({largest_position.portfolio_percentage:.1%})",
            "unrealized_pnl": f"${total_unrealized_pnl:,.0f}"
        }
    
    def _analyze_strategy_performance(self, positions: List[PositionRisk]) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        
        # This would integrate with actual strategy tracking
        # For now, we'll provide placeholder data based on our evaluation
        return {
            "automated_signal_trading": {
                "positions": len([p for p in positions if p.risk_score > 0.6]),
                "performance_grade": "F",
                "recent_return": "-2.1%",
                "status": "Needs Improvement"
            },
            "momentum_strategy": {
                "positions": len([p for p in positions if p.beta > 1.5]),
                "performance_grade": "F", 
                "recent_return": "-1.8%",
                "status": "Needs Improvement"
            },
            "mean_reversion_strategy": {
                "positions": len([p for p in positions if p.beta < 0.8]),
                "performance_grade": "F",
                "recent_return": "+0.5%",
                "status": "Needs Improvement"
            },
            "portfolio_rebalancing": {
                "positions": len([p for p in positions if 0.8 <= p.beta <= 1.2]),
                "performance_grade": "C",
                "recent_return": "+1.2%",
                "status": "Acceptable"
            }
        }
    
    def _generate_recommendations(self, risk_metrics, stop_loss_data: Dict, 
                                portfolio_summary: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_metrics.portfolio_heat > 0.2:
            recommendations.append("🔥 Reduce portfolio heat by closing high-risk positions")
        
        if risk_metrics.concentration_risk > 0.5:
            recommendations.append("⚖️ Diversify portfolio - concentration risk is high")
        
        if risk_metrics.volatility > 0.3:
            recommendations.append("📉 Consider reducing position sizes due to high volatility")
        
        # Stop-loss recommendations
        if stop_loss_data['active_stops'] < 3:
            recommendations.append("🎯 Ensure all positions have appropriate stop-losses")
        
        if stop_loss_data['triggered_today'] > 2:
            recommendations.append("🛑 Review trading strategy - multiple stops triggered today")
        
        # Strategy-specific recommendations
        recommendations.append("🔧 Implement unified risk management improvements (Phase 1 priority)")
        recommendations.append("📊 Add dynamic confidence thresholds for automated trading")
        recommendations.append("⏰ Consider shorter holding periods for momentum strategies")
        
        return recommendations
    
    def generate_html_dashboard(self, dashboard_data: RiskDashboardData) -> str:
        """Generate HTML dashboard for web display"""
        
        # Color coding based on risk level
        risk_colors = {
            "low": "#28a745",      # Green
            "medium": "#ffc107",   # Yellow
            "high": "#fd7e14",     # Orange
            "critical": "#dc3545"  # Red
        }
        
        risk_color = risk_colors.get(dashboard_data.risk_level, "#6c757d")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integrated Risk Management Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .status-card {{ background: {risk_color}; color: white; padding: 15px; 
                               border-radius: 8px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                                gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .alerts {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; 
                          border-radius: 8px; margin-bottom: 20px; }}
                .recommendations {{ background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; 
                                   border-radius: 8px; }}
                .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                                 gap: 15px; margin-top: 15px; }}
                .strategy-card {{ background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🛡️ Integrated Risk Management Dashboard</h1>
                    <p>Last Updated: {dashboard_data.timestamp}</p>
                </div>
                
                <div class="status-card">
                    <h2>{dashboard_data.overall_status}</h2>
                    <p>Risk Level: {dashboard_data.risk_level.upper()}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data.risk_metrics['portfolio_heat']}</div>
                        <div class="metric-label">Portfolio Heat</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data.stop_loss_status['active_stops']}</div>
                        <div class="metric-label">Active Stop-Losses</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data.risk_metrics['daily_var']}</div>
                        <div class="metric-label">Daily VaR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data.portfolio_summary['total_positions']}</div>
                        <div class="metric-label">Total Positions</div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>📊 Portfolio Summary</h3>
                    <p><strong>Total Value:</strong> {dashboard_data.portfolio_summary['total_value']}</p>
                    <p><strong>Cash:</strong> {dashboard_data.portfolio_summary['cash_percentage']}</p>
                    <p><strong>Largest Position:</strong> {dashboard_data.portfolio_summary['largest_position']}</p>
                    <p><strong>Unrealized P&L:</strong> {dashboard_data.portfolio_summary['unrealized_pnl']}</p>
                </div>
                
                <div class="metric-card">
                    <h3>🎯 Strategy Performance</h3>
                    <div class="strategy-grid">
        """
        
        # Add strategy cards
        for strategy, data in dashboard_data.strategy_performance.items():
            strategy_name = strategy.replace("_", " ").title()
            html += f"""
                        <div class="strategy-card">
                            <h4>{strategy_name}</h4>
                            <p><strong>Grade:</strong> {data['performance_grade']}</p>
                            <p><strong>Recent Return:</strong> {data['recent_return']}</p>
                            <p><strong>Positions:</strong> {data['positions']}</p>
                            <p><strong>Status:</strong> {data['status']}</p>
                        </div>
            """
        
        html += """
                    </div>
                </div>
        """
        
        # Add alerts if any
        if dashboard_data.alerts:
            html += f"""
                <div class="alerts">
                    <h3>⚠️ Active Alerts</h3>
                    <ul>
            """
            for alert in dashboard_data.alerts:
                html += f"<li>{alert}</li>"
            html += "</ul></div>"
        
        # Add recommendations
        html += f"""
                <div class="recommendations">
                    <h3>💡 Recommendations</h3>
                    <ul>
        """
        for rec in dashboard_data.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_dashboard_snapshot(self, dashboard_data: RiskDashboardData, filename: Optional[str] = None):
        """Save dashboard data snapshot to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_dashboard_snapshot_{timestamp}.json"
        
        # Convert dataclass to dict for JSON serialization
        data_dict = asdict(dashboard_data)
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        logger.info(f"📸 Dashboard snapshot saved: {filename}")
        return filename
    
    def get_dashboard_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze dashboard trends over specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_data = [
            data for data in self.dashboard_history 
            if datetime.fromisoformat(data.timestamp) > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        first_data = recent_data[0]
        latest_data = recent_data[-1]
        
        trends = {
            "period": f"Last {hours_back} hours",
            "data_points": len(recent_data),
            "risk_level_changes": len(set(d.risk_level for d in recent_data)),
            "stops_triggered": sum(d.stop_loss_status['triggered_today'] for d in recent_data),
            "alert_frequency": sum(len(d.alerts) for d in recent_data) / len(recent_data)
        }
        
        return trends

# Global dashboard instance
integrated_dashboard = IntegratedRiskDashboard()

def get_dashboard() -> IntegratedRiskDashboard:
    """Get the global dashboard instance"""
    return integrated_dashboard

if __name__ == "__main__":
    # Test the integrated dashboard
    logger.info("🧪 Testing Integrated Risk Dashboard...")
    
    async def test_dashboard():
        # Create sample data
        test_positions = [
            PositionRisk("AAPL", 15000, 0.15, 0.25, 1.2, 145.0, 450, 0.6),
            PositionRisk("MSFT", 12000, 0.12, 0.22, 1.1, 380.0, 360, 0.5),
            PositionRisk("NVDA", 8000, 0.08, 0.35, 1.8, 850.0, 280, 0.7),
            PositionRisk("SPY", 25000, 0.25, 0.15, 1.0, 450.0, 750, 0.3)
        ]
        
        market_data = {"AAPL": 148.0, "MSFT": 385.0, "NVDA": 880.0, "SPY": 455.0}
        account_value = 100000
        
        # Generate dashboard
        dashboard_data = await integrated_dashboard.generate_dashboard_data(
            test_positions, account_value, market_data
        )
        
        print("🛡️ INTEGRATED RISK DASHBOARD TEST:")
        print(f"Overall Status: {dashboard_data.overall_status}")
        print(f"Risk Level: {dashboard_data.risk_level}")
        print(f"Portfolio Heat: {dashboard_data.risk_metrics['portfolio_heat']}")
        print(f"Active Stops: {dashboard_data.stop_loss_status['active_stops']}")
        print(f"Total Alerts: {len(dashboard_data.alerts)}")
        print(f"Recommendations: {len(dashboard_data.recommendations)}")
        
        # Save snapshot
        snapshot_file = integrated_dashboard.save_dashboard_snapshot(dashboard_data)
        print(f"Snapshot saved: {snapshot_file}")
        
        # Generate HTML dashboard
        html_content = integrated_dashboard.generate_html_dashboard(dashboard_data)
        with open("risk_dashboard.html", "w", encoding='utf-8') as f:
            f.write(html_content)
        print("HTML dashboard saved: risk_dashboard.html")
        
        print("\n✅ Integrated Risk Dashboard test completed!")
    
    # Run the test
    asyncio.run(test_dashboard())
