#!/usr/bin/env python3
"""
🛡️ RISK SERVICE
Basic risk management service for trading operations
"""

from typing import Any

from infra.logging import get_structured_logger

logger = get_structured_logger(__name__)


class RiskService:
    """
    Risk management service for trading operations
    Provides position sizing, drawdown limits, and risk checks
    """

    def __init__(self):
        self.max_position_size = 1000  # Maximum shares per position
        self.max_portfolio_heat = 0.1  # Maximum 10% portfolio risk
        self.max_daily_loss = 0.05  # Maximum 5% daily loss

        logger.info("🛡️ Risk Service initialized")

    def check_position_risk(
        self,
        symbol: str,
        quantity: int,
        price: float,
        current_portfolio_value: float = 100000.0,
    ) -> dict[str, Any]:
        """
        Check if a position meets risk requirements

        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Share price
            current_portfolio_value: Current portfolio value

        Returns:
            Risk assessment result
        """
        position_value = quantity * price
        portfolio_heat = position_value / current_portfolio_value

        # Risk checks
        risks = []

        if quantity > self.max_position_size:
            risks.append(
                f"Position size {quantity} exceeds maximum {self.max_position_size}"
            )

        if portfolio_heat > self.max_portfolio_heat:
            risks.append(
                f"Portfolio heat {portfolio_heat:.2%} exceeds maximum {self.max_portfolio_heat:.2%}"
            )

        approved = len(risks) == 0

        result = {
            "approved": approved,
            "risks": risks,
            "position_value": position_value,
            "portfolio_heat": portfolio_heat,
            "max_quantity": min(
                self.max_position_size,
                int(current_portfolio_value * self.max_portfolio_heat / price),
            ),
        }

        if approved:
            logger.info(
                f"✅ Risk check passed for {symbol}: {quantity} shares @ ${price}"
            )
        else:
            logger.warning(f"⚠️ Risk check failed for {symbol}: {', '.join(risks)}")

        return result

    def check_portfolio_risk(self, current_pnl: float, portfolio_value: float) -> bool:
        """
        Check if portfolio risk is within limits

        Args:
            current_pnl: Current P&L for the day
            portfolio_value: Total portfolio value

        Returns:
            True if within risk limits
        """
        daily_loss_pct = abs(current_pnl) / portfolio_value if current_pnl < 0 else 0

        if daily_loss_pct > self.max_daily_loss:
            logger.warning(
                f"🚨 Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss:.2%}"
            )
            return False

        return True

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_per_share: float,
        portfolio_value: float = 100000.0,
    ) -> int:
        """
        Calculate optimal position size based on risk

        Args:
            symbol: Trading symbol
            price: Share price
            risk_per_share: Risk amount per share (e.g., stop loss distance)
            portfolio_value: Total portfolio value

        Returns:
            Recommended position size in shares
        """
        if risk_per_share <= 0:
            logger.warning(f"⚠️ Invalid risk per share for {symbol}: {risk_per_share}")
            return 0

        # Risk 1% of portfolio per trade
        risk_amount = portfolio_value * 0.01
        position_size = int(risk_amount / risk_per_share)

        # Apply maximum limits
        position_size = min(position_size, self.max_position_size)

        # Ensure we can afford the position
        max_affordable = int(portfolio_value * self.max_portfolio_heat / price)
        position_size = min(position_size, max_affordable)

        logger.info(
            f"📊 Position size for {symbol}: {position_size} shares (risk: ${risk_amount:.2f})"
        )

        return position_size
