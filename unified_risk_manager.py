#!/usr/bin/env python3
"""
🛡️ UNIFIED RISK MANAGEMENT SYSTEM
Phase 1 Optimization: Comprehensive risk management for all trading strategies
Part of Audit Item 4: Trading Strategy Reevaluation
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Portfolio risk metrics container"""
    portfolio_heat: float  # Current portfolio heat (0-100%)
    var_1d: float         # 1-day Value at Risk
    max_drawdown: float   # Current maximum drawdown
    sharpe_ratio: float   # Recent Sharpe ratio
    concentration_risk: float  # Concentration in top 3 positions
    leverage_ratio: float      # Current leverage usage
    volatility: float         # Portfolio volatility
    risk_level: RiskLevel     # Overall risk classification

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_value: float
    portfolio_percentage: float
    volatility: float
    beta: float
    stop_loss_price: float | None
    max_position_loss: float
    risk_score: float

class UnifiedRiskManager:
    """Comprehensive risk management system for all trading strategies"""

    def __init__(self):
        # Risk limits and thresholds
        self.max_portfolio_heat = 0.25        # Maximum 25% portfolio heat
        self.max_position_size = 0.10         # Maximum 10% per position
        self.max_sector_exposure = 0.40       # Maximum 40% per sector
        self.max_daily_loss = 0.03            # Maximum 3% daily portfolio loss
        self.max_drawdown_limit = 0.15        # Stop trading at 15% drawdown

        # Volatility-based parameters
        self.base_stop_loss = 0.03            # Base 3% stop-loss
        self.volatility_multiplier = 1.5      # Stop-loss volatility adjustment
        self.min_stop_loss = 0.015           # Minimum 1.5% stop-loss
        self.max_stop_loss = 0.08            # Maximum 8% stop-loss

        # Risk monitoring
        self.current_metrics = None
        self.risk_alerts = []
        self.position_limits = {}

        logger.info("🛡️ Unified Risk Management System initialized")

    def calculate_portfolio_heat(self, positions: list[PositionRisk], account_value: float) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        total_risk = 0.0

        for position in positions:
            # Risk is maximum potential loss per position
            position_risk = position.max_position_loss / account_value
            total_risk += position_risk

        return min(total_risk, 1.0)  # Cap at 100%

    def calculate_position_size(self, symbol: str, confidence: float, kelly_fraction: float,
                              current_price: float, account_value: float, volatility: float) -> int:
        """Calculate optimal position size with unified risk management"""

        # Base position size using Kelly Criterion (but capped)
        kelly_position_size = kelly_fraction * 0.5  # Use half-Kelly for safety

        # Confidence-based adjustment
        confidence_adjustment = min(confidence / 0.75, 1.5)  # Cap at 150% for high confidence

        # Volatility adjustment (reduce size for high volatility)
        volatility_adjustment = 1.0 / (1.0 + volatility * 2)

        # Combined position size (as fraction of account)
        target_position_fraction = kelly_position_size * confidence_adjustment * volatility_adjustment

        # Apply risk limits
        target_position_fraction = min(target_position_fraction, self.max_position_size)

        # Convert to dollar amount and shares
        target_dollar_amount = account_value * target_position_fraction
        target_shares = int(target_dollar_amount / current_price)

        # Final risk check
        portfolio_heat_impact = self.estimate_position_heat(target_shares, current_price, volatility, account_value)

        if self.current_metrics and (self.current_metrics.portfolio_heat + portfolio_heat_impact) > self.max_portfolio_heat:
            # Reduce position size to stay within heat limits
            available_heat = self.max_portfolio_heat - self.current_metrics.portfolio_heat
            max_allowed_shares = int((available_heat * account_value) / (current_price * self.base_stop_loss))
            target_shares = min(target_shares, max_allowed_shares)

        logger.info(f"📊 Position size for {symbol}: {target_shares} shares "
                   f"(${target_shares * current_price:,.0f}, {target_position_fraction:.1%} of portfolio)")

        return max(target_shares, 0)

    def estimate_position_heat(self, shares: int, price: float, volatility: float, account_value: float) -> float:
        """Estimate the portfolio heat contribution of a position"""
        position_value = shares * price

        # Estimate potential loss using volatility-adjusted stop-loss
        volatility_stop = max(self.min_stop_loss, min(self.max_stop_loss,
                                                     self.base_stop_loss + volatility * self.volatility_multiplier))

        max_loss = position_value * volatility_stop
        heat_contribution = max_loss / account_value

        return heat_contribution

    def calculate_stop_loss(self, symbol: str, entry_price: float, volatility: float,
                           confidence: float, strategy_type: str) -> float:
        """Calculate dynamic stop-loss based on multiple factors"""

        # Base stop-loss adjusted for volatility
        volatility_stop = self.base_stop_loss + (volatility * self.volatility_multiplier)

        # Confidence adjustment (higher confidence = slightly wider stops)
        confidence_adjustment = 1.0 + (confidence - 0.5) * 0.2  # ±10% adjustment

        # Strategy-specific adjustments
        strategy_multipliers = {
            "automated_signal_trading": 1.0,    # Standard stops
            "momentum_strategy": 0.8,           # Tighter stops for momentum
            "mean_reversion_strategy": 1.2,     # Wider stops for mean reversion
            "portfolio_rebalancing": 1.5        # Very wide stops for long-term positions
        }

        strategy_multiplier = strategy_multipliers.get(strategy_type, 1.0)

        # Calculate final stop-loss percentage
        stop_loss_pct = volatility_stop * confidence_adjustment * strategy_multiplier

        # Apply limits
        stop_loss_pct = max(self.min_stop_loss, min(self.max_stop_loss, stop_loss_pct))

        # Convert to price
        stop_loss_price = entry_price * (1.0 - stop_loss_pct)

        logger.info(f"🛡️ Stop-loss for {symbol}: ${stop_loss_price:.2f} "
                   f"({stop_loss_pct:.1%} from entry ${entry_price:.2f})")

        return stop_loss_price

    def assess_portfolio_risk(self, positions: list[PositionRisk], account_value: float) -> RiskMetrics:
        """Comprehensive portfolio risk assessment"""

        if not positions:
            return RiskMetrics(
                portfolio_heat=0.0, var_1d=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                concentration_risk=0.0, leverage_ratio=0.0, volatility=0.0,
                risk_level=RiskLevel.LOW
            )

        # Calculate portfolio heat
        portfolio_heat = self.calculate_portfolio_heat(positions, account_value)

        # Calculate concentration risk (top 3 positions)
        position_weights = [pos.portfolio_percentage for pos in positions]
        position_weights.sort(reverse=True)
        concentration_risk = sum(position_weights[:3])

        # Portfolio volatility (weighted average)
        portfolio_volatility = sum(pos.volatility * pos.portfolio_percentage for pos in positions)

        # Simple VaR estimation (1-day, 95% confidence)
        var_1d = account_value * portfolio_volatility * 1.65  # 95% VaR approximation

        # Determine overall risk level
        risk_level = self._classify_risk_level(portfolio_heat, concentration_risk, portfolio_volatility)

        metrics = RiskMetrics(
            portfolio_heat=portfolio_heat,
            var_1d=var_1d,
            max_drawdown=0.0,  # Would need historical data
            sharpe_ratio=0.0,   # Would need historical data
            concentration_risk=concentration_risk,
            leverage_ratio=sum(pos.portfolio_percentage for pos in positions),
            volatility=portfolio_volatility,
            risk_level=risk_level
        )

        self.current_metrics = metrics
        return metrics

    def _classify_risk_level(self, portfolio_heat: float, concentration: float, volatility: float) -> RiskLevel:
        """Classify overall portfolio risk level"""

        # Risk scoring based on multiple factors
        heat_score = portfolio_heat / self.max_portfolio_heat
        concentration_score = concentration / 0.6  # 60% concentration threshold
        volatility_score = volatility / 0.25  # 25% volatility threshold

        overall_score = (heat_score + concentration_score + volatility_score) / 3

        if overall_score >= 0.9:
            return RiskLevel.CRITICAL
        elif overall_score >= 0.7:
            return RiskLevel.HIGH
        elif overall_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def check_trade_approval(self, symbol: str, trade_type: str, quantity: int,
                           price: float, strategy: str) -> tuple[bool, str]:
        """Check if a trade should be approved based on risk management rules"""

        trade_value = quantity * price

        # Check if we have current risk metrics
        if not self.current_metrics:
            logger.warning("⚠️ No current risk metrics available, allowing trade with basic checks")

        # Basic size limits
        if quantity <= 0:
            return False, "Invalid quantity"

        # Portfolio heat check
        if self.current_metrics:
            estimated_heat_increase = self.estimate_position_heat(quantity, price, 0.02, 100000)  # Rough estimate
            if self.current_metrics.portfolio_heat + estimated_heat_increase > self.max_portfolio_heat:
                return False, f"Trade would exceed portfolio heat limit ({self.max_portfolio_heat:.1%})"

        # Strategy-specific checks
        if strategy == "automated_signal_trading" and trade_value < 1000:
            return False, "Minimum trade size for automated signals: $1,000"

        logger.info(f"✅ Trade approved: {trade_type} {quantity} {symbol} @ ${price:.2f}")
        return True, "Trade approved"

    def generate_risk_alerts(self, metrics: RiskMetrics) -> list[str]:
        """Generate risk alerts based on current portfolio metrics"""

        alerts = []

        # Portfolio heat alerts
        if metrics.portfolio_heat > self.max_portfolio_heat * 0.8:
            alerts.append(f"🔥 HIGH PORTFOLIO HEAT: {metrics.portfolio_heat:.1%} (limit: {self.max_portfolio_heat:.1%})")

        # Concentration alerts
        if metrics.concentration_risk > 0.6:
            alerts.append(f"⚠️ HIGH CONCENTRATION: Top 3 positions = {metrics.concentration_risk:.1%}")

        # Volatility alerts
        if metrics.volatility > 0.3:
            alerts.append(f"📈 HIGH VOLATILITY: Portfolio vol = {metrics.volatility:.1%}")

        # VaR alerts
        if metrics.var_1d > 50000:  # $50k daily VaR threshold
            alerts.append(f"💰 HIGH VAR: 1-day VaR = ${metrics.var_1d:,.0f}")

        # Risk level alerts
        if metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            alerts.append(f"🚨 {metrics.risk_level.value.upper()} RISK LEVEL DETECTED")

        self.risk_alerts = alerts
        return alerts

    def get_risk_dashboard(self) -> dict[str, Any]:
        """Get comprehensive risk dashboard data"""

        if not self.current_metrics:
            return {"status": "No risk data available"}

        alerts = self.generate_risk_alerts(self.current_metrics)

        return {
            "timestamp": datetime.now().isoformat(),
            "risk_level": self.current_metrics.risk_level.value,
            "portfolio_heat": f"{self.current_metrics.portfolio_heat:.1%}",
            "heat_limit": f"{self.max_portfolio_heat:.1%}",
            "concentration_risk": f"{self.current_metrics.concentration_risk:.1%}",
            "portfolio_volatility": f"{self.current_metrics.volatility:.1%}",
            "daily_var": f"${self.current_metrics.var_1d:,.0f}",
            "leverage_ratio": f"{self.current_metrics.leverage_ratio:.1%}",
            "active_alerts": len(alerts),
            "alerts": alerts,
            "risk_limits": {
                "max_portfolio_heat": f"{self.max_portfolio_heat:.1%}",
                "max_position_size": f"{self.max_position_size:.1%}",
                "max_daily_loss": f"{self.max_daily_loss:.1%}",
                "max_drawdown_limit": f"{self.max_drawdown_limit:.1%}"
            }
        }

# Global risk manager instance
risk_manager = UnifiedRiskManager()

def get_risk_manager() -> UnifiedRiskManager:
    """Get the global risk manager instance"""
    return risk_manager

if __name__ == "__main__":
    # Test the risk management system
    logger.info("🧪 Testing Unified Risk Management System...")

    # Create sample positions
    test_positions = [
        PositionRisk("AAPL", 15000, 0.15, 0.25, 1.2, 145.0, 450, 0.6),
        PositionRisk("MSFT", 12000, 0.12, 0.22, 1.1, 380.0, 360, 0.5),
        PositionRisk("NVDA", 8000, 0.08, 0.35, 1.8, 850.0, 280, 0.7)
    ]

    # Test risk assessment
    metrics = risk_manager.assess_portfolio_risk(test_positions, 100000)
    dashboard = risk_manager.get_risk_dashboard()

    print("🛡️ UNIFIED RISK MANAGEMENT TEST RESULTS:")
    print(f"Portfolio Heat: {dashboard['portfolio_heat']}")
    print(f"Risk Level: {dashboard['risk_level']}")
    print(f"Concentration Risk: {dashboard['concentration_risk']}")
    print(f"Daily VaR: {dashboard['daily_var']}")
    print(f"Active Alerts: {dashboard['active_alerts']}")

    if dashboard['alerts']:
        print("Active Alerts:")
        for alert in dashboard['alerts']:
            print(f"  {alert}")

    # Test position sizing
    shares = risk_manager.calculate_position_size("TSLA", 0.8, 0.15, 900.0, 100000, 0.3)
    print(f"\nOptimal position size for TSLA: {shares} shares")

    # Test stop-loss calculation
    stop_price = risk_manager.calculate_stop_loss("TSLA", 900.0, 0.3, 0.8, "momentum_strategy")
    print(f"Stop-loss for TSLA: ${stop_price:.2f}")

    print("\n✅ Unified Risk Management System test completed!")
