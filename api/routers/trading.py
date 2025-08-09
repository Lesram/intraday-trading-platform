#!/usr/bin/env python3
"""
📈 TRADING ROUTER
Trading execution endpoints with risk integration
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from adapters.alpaca_client import AlpacaClient
from adapters.errors import AlpacaError
from infra.logging import get_structured_logger, log_trade_event
from services.broker_service import broker_service
from services.risk_service import RiskDecisionType, risk_service

router = APIRouter()
logger = get_structured_logger("api.trading")


# Pydantic DTOs
class MarketOrderRequest(BaseModel):
    """Market order request DTO"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Trading symbol")
    side: str = Field(..., pattern="^(buy|sell)$", description="Order side")
    qty: float = Field(..., gt=0, description="Quantity to trade")
    oco_stop: float | None = Field(None, gt=0, description="OCO stop loss price")


class OrderCancelRequest(BaseModel):
    """Order cancellation request DTO"""
    reason: str | None = Field(None, description="Cancellation reason")


class TradingModeRequest(BaseModel):
    """Trading mode configuration DTO"""
    auto_trading: bool = Field(..., description="Enable/disable auto trading")
    risk_management: bool = Field(..., description="Enable/disable risk management")
    max_portfolio_heat: float | None = Field(None, ge=0.01, le=1.0, description="Max portfolio heat")


class OrderResponse(BaseModel):
    """Order response DTO"""
    request_id: str
    status: str
    order_ref: str
    symbol: str
    side: str
    qty: float
    approved_qty: float
    risk_decision: str
    reasons: list[str]
    timestamp: str


class OrderCancelResponse(BaseModel):
    """Order cancellation response DTO"""
    request_id: str
    status: str
    order_ref: str
    cancelled: bool
    timestamp: str


class TradingModeResponse(BaseModel):
    """Trading mode response DTO"""
    request_id: str
    status: str
    auto_trading: bool
    risk_management: bool
    timestamp: str


async def get_alpaca_data():
    """Dependency to get Alpaca account and positions data"""
    async with AlpacaClient() as client:
        account = await client.get_account()
        positions = await client.get_positions()
        return account, positions


@router.post("/orders/market", response_model=OrderResponse)
async def submit_market_order(
    order_request: MarketOrderRequest,
    request: Request,
    alpaca_data = Depends(get_alpaca_data)
) -> OrderResponse:
    """Submit a market order with risk checks"""

    request_id = getattr(request.state, 'request_id', 'unknown')
    account, positions = alpaca_data

    try:
        # Get current market price (simplified - would normally fetch real price)
        # For now, use a mock price based on symbol
        mock_prices = {
            "AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0,
            "TSLA": 200.0, "NVDA": 400.0, "SPY": 450.0
        }
        current_price = mock_prices.get(order_request.symbol.upper(), 100.0)

        logger.info(f"📤 Processing market order: {order_request.side} {order_request.qty} {order_request.symbol}")

        # Step 1: Risk Service Check
        risk_decision = await risk_service.check_trade_risk(
            symbol=order_request.symbol,
            side=order_request.side,
            qty=order_request.qty,
            price=current_price,
            positions=positions,
            account=account
        )

        # Step 2: Handle Risk Decision
        if risk_decision.decision == RiskDecisionType.REJECTED:
            log_trade_event(
                "order_rejected_by_risk",
                order_request.symbol,
                request_id=request_id,
                side=order_request.side,
                qty=order_request.qty,
                reasons=risk_decision.reasons
            )

            return OrderResponse(
                request_id=request_id,
                status="rejected",
                order_ref="",
                symbol=order_request.symbol,
                side=order_request.side,
                qty=order_request.qty,
                approved_qty=0.0,
                risk_decision="REJECTED",
                reasons=risk_decision.reasons,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        # Step 3: Submit Order via Broker Service
        try:
            order = await broker_service.submit_market_order(
                symbol=order_request.symbol,
                side=order_request.side,
                qty=risk_decision.approved_qty,
                oco_stop=order_request.oco_stop
            )

            log_trade_event(
                "order_submitted",
                order_request.symbol,
                request_id=request_id,
                order_id=order.id,
                side=order_request.side,
                original_qty=order_request.qty,
                approved_qty=risk_decision.approved_qty,
                risk_decision=risk_decision.decision.value
            )

            return OrderResponse(
                request_id=request_id,
                status="submitted",
                order_ref=order.id,
                symbol=order_request.symbol,
                side=order_request.side,
                qty=order_request.qty,
                approved_qty=risk_decision.approved_qty,
                risk_decision=risk_decision.decision.value,
                reasons=risk_decision.reasons,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        except AlpacaError as e:
            log_trade_event(
                "order_submission_failed",
                order_request.symbol,
                request_id=request_id,
                error=str(e),
                error_code=e.error_code
            )

            # Map specific errors to appropriate HTTP status codes
            if e.status_code in [401, 403]:
                status_code = 401
            elif e.status_code in [422, 400]:
                # Return HTTP 422 for broker validation errors
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": "rejected",
                        "symbol": order_request.symbol,
                        "side": order_request.side,
                        "qty": order_request.qty,
                        "reasons": [e.message or "Invalid order parameters"],
                        "error_code": e.error_code,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                )
            elif e.status_code >= 500:
                # Map 5xx to service unavailable
                raise HTTPException(
                    status_code=503,
                    detail="Broker service temporarily unavailable. Please try again later."
                )
            else:
                status_code = 400

            raise HTTPException(
                status_code=status_code,
                detail={
                    "request_id": request_id,
                    "status": "rejected",
                    "error": "Order submission failed",
                    "message": e.message,
                    "error_code": e.error_code
                }
            )

    except HTTPException:
        # Re-raise HTTPExceptions so they're handled by FastAPI
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in market order: {str(e)}")
        
        # Check for specific error types
        error_str = str(e).lower()
        
        # Check if it's a connection/timeout error -> map to 503
        if any(keyword in error_str for keyword in ["connectionerror", "timeout", "retryerror", "connection"]):
            raise HTTPException(
                status_code=503,
                detail="Broker service temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "request_id": request_id,
                    "status": "error",
                    "error": "Internal server error",
                    "message": str(e)
                }
            )


@router.post("/orders/{order_id}/cancel", response_model=OrderCancelResponse)
async def cancel_order(
    order_id: str,
    cancel_request: OrderCancelRequest,
    request: Request
) -> OrderCancelResponse:
    """Cancel an order by ID"""

    request_id = getattr(request.state, 'request_id', 'unknown')

    try:
        logger.info(f"🚫 Cancelling order: {order_id}")

        # Get order details before cancellation
        order = broker_service.get_order(order_id)
        if not order:
            raise HTTPException(
                status_code=404,
                detail={
                    "request_id": request_id,
                    "error": "Order not found",
                    "order_id": order_id
                }
            )

        # Attempt cancellation
        success = await broker_service.cancel_order(order_id)

        log_trade_event(
            "order_cancellation_request",
            order.symbol,
            request_id=request_id,
            order_id=order_id,
            success=success,
            reason=cancel_request.reason
        )

        return OrderCancelResponse(
            request_id=request_id,
            status="success" if success else "failed",
            order_ref=order_id,
            cancelled=success,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        logger.error(f"❌ Error cancelling order {order_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "error": "Cancellation failed",
                "message": str(e)
            }
        )


@router.post("/mode", response_model=TradingModeResponse)
async def set_trading_mode(
    mode_request: TradingModeRequest,
    request: Request
) -> TradingModeResponse:
    """Set trading mode configuration"""

    request_id = getattr(request.state, 'request_id', 'unknown')

    try:
        logger.info(f"⚙️ Setting trading mode: auto={mode_request.auto_trading}, risk={mode_request.risk_management}")

        # Here you would update global trading configuration
        # For now, just log the settings

        log_trade_event(
            "trading_mode_change",
            "SYSTEM",
            request_id=request_id,
            auto_trading=mode_request.auto_trading,
            risk_management=mode_request.risk_management,
            max_portfolio_heat=mode_request.max_portfolio_heat
        )

        return TradingModeResponse(
            request_id=request_id,
            status="updated",
            auto_trading=mode_request.auto_trading,
            risk_management=mode_request.risk_management,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        logger.error(f"❌ Error setting trading mode: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "error": "Mode update failed",
                "message": str(e)
            }
        )


@router.get("/orders/{order_id}")
async def get_order_status(order_id: str, request: Request) -> dict[str, Any]:
    """Get order status by ID"""

    request_id = getattr(request.state, 'request_id', 'unknown')

    order = broker_service.get_order(order_id)
    if not order:
        raise HTTPException(
            status_code=404,
            detail={
                "request_id": request_id,
                "error": "Order not found",
                "order_id": order_id
            }
        )

    return {
        "request_id": request_id,
        "order": broker_service.to_dict(order)
    }


@router.get("/orders")
async def get_orders(request: Request, active_only: bool = False) -> dict[str, Any]:
    """Get all orders"""

    request_id = getattr(request.state, 'request_id', 'unknown')

    if active_only:
        orders = broker_service.get_active_orders()
    else:
        orders = list(broker_service.orders.values())

    return {
        "request_id": request_id,
        "orders": [broker_service.to_dict(order) for order in orders],
        "count": len(orders)
    }


@router.get("/risk/summary")
async def get_risk_summary(request: Request, alpaca_data = Depends(get_alpaca_data)) -> dict[str, Any]:
    """Get current risk summary"""

    request_id = getattr(request.state, 'request_id', 'unknown')
    account, positions = alpaca_data

    risk_summary = risk_service.get_risk_summary(positions, account)

    return {
        "request_id": request_id,
        "risk_summary": risk_summary
    }
