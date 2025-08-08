#!/usr/bin/env python3
"""
🛡️ RISK SERVICE
Risk management service wrapping UnifiedRiskManager with structured decisions
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from unified_risk_manager import get_risk_manager, RiskMetrics
from adapters.alpaca_client import AlpacaAccount, AlpacaPosition
from infra.settings import settings
from infra.logging import get_structured_logger, log_risk_event

logger = get_structured_logger("services.risk")


class RiskDecisionType(Enum):
    """Risk decision types"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


@dataclass
class RiskDecision:
    """Structured risk decision"""
    decision: RiskDecisionType
    original_qty: float
    approved_qty: float
    reasons: List[str]
    risk_metrics: Dict[str, Any]
    portfolio_heat_before: float
    portfolio_heat_after: float
    confidence_adjustment: float
    timestamp: datetime


class RiskBlocked(Exception):
    """Exception raised when risk limits block a trade"""
    
    def __init__(self, decision: RiskDecision):
        self.decision = decision
        super().__init__(f"Trade blocked: {', '.join(decision.reasons)}")


class RiskService:
    """Risk management service with structured decisions"""
    
    def __init__(self):
        self.risk_manager = get_risk_manager()
        self.daily_losses = 0.0
        self.daily_loss_reset_date = datetime.utcnow().date()
        self.max_drawdown_reached = False
        
        logger.info("🛡️ Risk service initialized")
    
    async def check_trade_risk(self, symbol: str, side: str, qty: float, 
                              price: float, positions: List[AlpacaPosition], 
                              account: AlpacaAccount) -> RiskDecision:
        """Check trade risk and return structured decision"""
        
        reasons = []
        original_qty = qty
        approved_qty = qty
        confidence_adjustment = 1.0
        
        # Reset daily losses if new day
        current_date = datetime.utcnow().date()
        if current_date != self.daily_loss_reset_date:
            self.daily_losses = 0.0
            self.daily_loss_reset_date = current_date
            logger.info("🔄 Daily loss counter reset")
        
        # Calculate current portfolio metrics
        portfolio_value = account.portfolio_value
        current_positions_list = [
            {
                'symbol': pos.symbol,
                'market_value': pos.market_value,
                'qty': pos.qty,
                'unrealized_pl': pos.unrealized_pl
            }
            for pos in positions
        ]
        
        # Get current risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk(current_positions_list, portfolio_value)
        portfolio_heat_before = risk_metrics.portfolio_heat if hasattr(risk_metrics, 'portfolio_heat') else 0.0
        
        # 1. Daily Loss Guard
        daily_loss_limit = settings.daily_loss_limit * portfolio_value
        if self.daily_losses > daily_loss_limit:
            reasons.append(f"Daily loss limit exceeded: ${self.daily_losses:.2f} > ${daily_loss_limit:.2f}")
            approved_qty = 0.0
        
        # 2. Max Drawdown Guard 
        max_drawdown_limit = settings.max_drawdown_limit
        if hasattr(risk_metrics, 'max_drawdown') and risk_metrics.max_drawdown > max_drawdown_limit:
            reasons.append(f"Max drawdown limit exceeded: {risk_metrics.max_drawdown:.1%} > {max_drawdown_limit:.1%}")
            approved_qty = 0.0
        
        # 3. Portfolio Heat Limit
        trade_value = qty * price
        estimated_loss = trade_value * 0.08  # Assume 8% stop loss
        heat_contribution = estimated_loss / portfolio_value
        portfolio_heat_after = portfolio_heat_before + heat_contribution
        
        if portfolio_heat_after > settings.max_portfolio_heat:
            # Try to reduce position size to fit within heat limit
            max_heat_available = settings.max_portfolio_heat - portfolio_heat_before
            if max_heat_available > 0.01:  # At least 1% heat available
                max_trade_value = (max_heat_available * portfolio_value) / 0.08
                max_qty = int(max_trade_value / price)
                if max_qty > 0:
                    approved_qty = max_qty
                    reasons.append(f"Position size reduced for portfolio heat: {qty} → {max_qty}")
                    confidence_adjustment = 0.8  # Reduce confidence for smaller position
                else:
                    approved_qty = 0.0
                    reasons.append(f"Portfolio heat limit exceeded: {portfolio_heat_after:.1%} > {settings.max_portfolio_heat:.1%}")
            else:
                approved_qty = 0.0
                reasons.append(f"Portfolio heat limit exceeded: {portfolio_heat_after:.1%} > {settings.max_portfolio_heat:.1%}")
        
        # 4. Position Size Limit
        position_value = approved_qty * price
        position_percentage = position_value / portfolio_value
        if position_percentage > settings.max_position_size:
            max_position_value = settings.max_position_size * portfolio_value
            max_qty = int(max_position_value / price)
            if max_qty < approved_qty:
                approved_qty = max_qty
                reasons.append(f"Position size limit: {position_percentage:.1%} > {settings.max_position_size:.1%}")
                confidence_adjustment = min(confidence_adjustment, 0.9)
        
        # 5. Account Buying Power Check
        required_buying_power = approved_qty * price
        if side.lower() == "buy" and required_buying_power > account.buying_power:
            max_affordable_qty = int(account.buying_power / price)
            if max_affordable_qty < approved_qty:
                approved_qty = max_affordable_qty
                reasons.append(f"Insufficient buying power: ${required_buying_power:.2f} > ${account.buying_power:.2f}")
        
        # Determine decision type
        if approved_qty == 0:
            decision_type = RiskDecisionType.REJECTED
        elif approved_qty != original_qty:
            decision_type = RiskDecisionType.MODIFIED
        else:
            decision_type = RiskDecisionType.APPROVED
            reasons.append("All risk checks passed")
        
        # Final portfolio heat calculation
        if approved_qty > 0:
            final_trade_value = approved_qty * price
            final_estimated_loss = final_trade_value * 0.08
            final_heat_contribution = final_estimated_loss / portfolio_value
            portfolio_heat_after = portfolio_heat_before + final_heat_contribution
        else:
            portfolio_heat_after = portfolio_heat_before
        
        # Create risk decision
        decision = RiskDecision(
            decision=decision_type,
            original_qty=original_qty,
            approved_qty=approved_qty,
            reasons=reasons,
            risk_metrics={
                "portfolio_value": portfolio_value,
                "buying_power": account.buying_power,
                "daily_losses": self.daily_losses,
                "max_drawdown": getattr(risk_metrics, 'max_drawdown', 0.0),
                "position_percentage": (approved_qty * price) / portfolio_value if approved_qty > 0 else 0.0
            },
            portfolio_heat_before=portfolio_heat_before,
            portfolio_heat_after=portfolio_heat_after,
            confidence_adjustment=confidence_adjustment,
            timestamp=datetime.utcnow()
        )
        
        # Log risk decision
        log_risk_event(
            "trade_risk_check",
            symbol=symbol,
            side=side,
            decision=decision_type.value,
            original_qty=original_qty,
            approved_qty=approved_qty,
            reasons=reasons,
            portfolio_heat_before=portfolio_heat_before,
            portfolio_heat_after=portfolio_heat_after
        )
        
        # Log decision
        if decision_type == RiskDecisionType.APPROVED:
            logger.info(f"✅ Trade approved: {symbol} {side} {approved_qty} shares")
        elif decision_type == RiskDecisionType.MODIFIED:
            logger.info(f"⚠️ Trade modified: {symbol} {side} {original_qty} → {approved_qty} shares")
        else:
            logger.warning(f"🚫 Trade rejected: {symbol} {side} {original_qty} shares")
        
        return decision
    
    def update_daily_loss(self, loss_amount: float) -> None:
        """Update daily loss tracking"""
        self.daily_losses += loss_amount
        
        log_risk_event(
            "daily_loss_update",
            loss_amount=loss_amount,
            total_daily_loss=self.daily_losses,
            daily_loss_limit=settings.daily_loss_limit
        )
        
        logger.info(f"💸 Daily loss updated: ${loss_amount:.2f} (Total: ${self.daily_losses:.2f})")
    
    def check_position_limits(self, symbol: str, new_position_value: float, 
                            total_portfolio_value: float) -> bool:
        """Check if position would exceed limits"""
        
        position_percentage = new_position_value / total_portfolio_value
        
        if position_percentage > settings.max_position_size:
            logger.warning(f"🚫 Position limit exceeded for {symbol}: {position_percentage:.1%} > {settings.max_position_size:.1%}")
            return False
        
        return True
    
    def get_risk_summary(self, positions: List[AlpacaPosition], 
                        account: AlpacaAccount) -> Dict[str, Any]:
        """Get current risk summary"""
        
        current_positions_list = [
            {
                'symbol': pos.symbol,
                'market_value': pos.market_value,
                'qty': pos.qty,
                'unrealized_pl': pos.unrealized_pl
            }
            for pos in positions
        ]
        
        risk_metrics = self.risk_manager.calculate_portfolio_risk(current_positions_list, account.portfolio_value)
        
        return {
            "portfolio_value": account.portfolio_value,
            "buying_power": account.buying_power,
            "cash": account.cash,
            "daily_losses": self.daily_losses,
            "daily_loss_limit": settings.daily_loss_limit * account.portfolio_value,
            "portfolio_heat": getattr(risk_metrics, 'portfolio_heat', 0.0),
            "portfolio_heat_limit": settings.max_portfolio_heat,
            "max_position_size": settings.max_position_size,
            "max_drawdown": getattr(risk_metrics, 'max_drawdown', 0.0),
            "max_drawdown_limit": settings.max_drawdown_limit,
            "risk_level": getattr(risk_metrics, 'risk_level', "unknown"),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global risk service instance
risk_service = RiskService()
