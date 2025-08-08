#!/usr/bin/env python3
"""
🏪 BROKER SERVICE  
Order state machine and trading execution service
"""

import asyncio
import uuid
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from adapters.alpaca_client import AlpacaClient, AlpacaOrder
from adapters.errors import AlpacaError
from infra.logging import get_structured_logger, log_trade_event

logger = get_structured_logger("services.broker")


class OrderState(Enum):
    """Order state machine states"""
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class InternalOrder:
    """Internal order representation with state tracking"""
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    state: OrderState
    created_at: datetime
    updated_at: datetime
    
    # Alpaca fields
    alpaca_order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    
    # OCO Stop Loss
    oco_stop_price: Optional[float] = None
    oco_stop_order_id: Optional[str] = None
    
    # Audit trail
    state_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.state_history is None:
            self.state_history = []


class BrokerService:
    """Order management service with state machine"""
    
    def __init__(self):
        self.orders: Dict[str, InternalOrder] = {}
        self.alpaca_client = None
        logger.info("🏪 Broker service initialized")
    
    async def initialize(self):
        """Initialize the broker service with Alpaca client"""
        self.alpaca_client = AlpacaClient()
        await self.alpaca_client.__aenter__()
        logger.info("🔗 Alpaca client connected")
    
    async def shutdown(self):
        """Shutdown the broker service"""
        if self.alpaca_client:
            await self.alpaca_client.__aexit__(None, None, None)
            logger.info("🔌 Alpaca client disconnected")
    
    def _transition_state(self, order: InternalOrder, new_state: OrderState, 
                         reason: str = "", additional_data: Dict[str, Any] = None) -> None:
        """Transition order to new state with audit logging"""
        
        old_state = order.state
        order.state = new_state
        order.updated_at = datetime.utcnow()
        
        # Add to state history
        state_change = {
            "timestamp": order.updated_at.isoformat() + "Z",
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason,
            "additional_data": additional_data or {}
        }
        order.state_history.append(state_change)
        
        # Emit structured audit log
        log_trade_event(
            "order_state_transition",
            order.symbol,
            order_id=order.id,
            client_order_id=order.client_order_id,
            alpaca_order_id=order.alpaca_order_id,
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
            **additional_data or {}
        )
        
        logger.info(f"📊 Order {order.id} state: {old_state.value} → {new_state.value} ({reason})")
    
    async def submit_market_order(self, symbol: str, side: str, qty: float, 
                                 oco_stop: Optional[float] = None) -> InternalOrder:
        """Submit market order with optional OCO stop loss"""
        
        if not self.alpaca_client:
            raise RuntimeError("Broker service not initialized")
        
        # Create internal order
        order_id = str(uuid.uuid4())
        client_order_id = f"ITP-{order_id[:8]}"
        
        order = InternalOrder(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            state=OrderState.NEW,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            oco_stop_price=oco_stop,
            state_history=[]
        )
        
        self.orders[order_id] = order
        
        try:
            # Log order creation
            log_trade_event(
                "order_created",
                symbol,
                order_id=order_id,
                client_order_id=client_order_id,
                side=side,
                qty=qty,
                order_type="market",
                oco_stop_price=oco_stop
            )
            
            # Transition to SUBMITTED
            self._transition_state(order, OrderState.SUBMITTED, "Submitting to Alpaca")
            
            # Submit to Alpaca
            alpaca_order = await self.alpaca_client.submit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="market",
                time_in_force="day",
                client_order_id=client_order_id
            )
            
            # Update with Alpaca order ID
            order.alpaca_order_id = alpaca_order.id
            
            # Transition to ACCEPTED
            self._transition_state(order, OrderState.ACCEPTED, "Accepted by Alpaca", {
                "alpaca_order_id": alpaca_order.id,
                "alpaca_status": alpaca_order.status
            })
            
            # Start monitoring the order
            asyncio.create_task(self._monitor_order(order))
            
            return order
            
        except AlpacaError as e:
            # Order rejected
            self._transition_state(order, OrderState.REJECTED, f"Alpaca error: {e.message}", {
                "error_code": e.error_code,
                "status_code": e.status_code
            })
            raise
        except Exception as e:
            self._transition_state(order, OrderState.REJECTED, f"Unexpected error: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by internal order ID"""
        
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"⚠️ Order not found for cancellation: {order_id}")
            return False
        
        if order.state in [OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED]:
            logger.warning(f"⚠️ Cannot cancel order in state {order.state.value}: {order_id}")
            return False
        
        try:
            # Cancel with Alpaca
            if order.alpaca_order_id:
                success = await self.alpaca_client.cancel_order(order.alpaca_order_id)
                
                if success:
                    self._transition_state(order, OrderState.CANCELED, "Cancelled by user request")
                    
                    # Cancel OCO stop if exists
                    if order.oco_stop_order_id:
                        await self.alpaca_client.cancel_order(order.oco_stop_order_id)
                        logger.info(f"🛑 OCO stop order cancelled: {order.oco_stop_order_id}")
                    
                    return True
                else:
                    return False
            else:
                # Order not yet submitted to Alpaca
                self._transition_state(order, OrderState.CANCELED, "Cancelled before submission")
                return True
                
        except AlpacaError as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e.message}")
            return False
    
    async def _monitor_order(self, order: InternalOrder) -> None:
        """Monitor order status and handle fills"""
        
        if not order.alpaca_order_id:
            return
        
        max_checks = 60  # Monitor for up to 10 minutes (10s intervals)
        check_count = 0
        
        while check_count < max_checks and order.state in [OrderState.ACCEPTED, OrderState.PARTIALLY_FILLED]:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                check_count += 1
                
                # Get latest order status from Alpaca
                alpaca_order = await self.alpaca_client.get_order(order.alpaca_order_id)
                
                # Update fill information
                order.filled_qty = alpaca_order.filled_qty or 0.0
                order.filled_avg_price = alpaca_order.filled_avg_price
                
                # Handle state transitions based on Alpaca status
                if alpaca_order.status == "filled":
                    self._transition_state(order, OrderState.FILLED, "Order filled", {
                        "filled_qty": order.filled_qty,
                        "filled_avg_price": order.filled_avg_price
                    })
                    
                    # Submit OCO stop loss if configured
                    if order.oco_stop_price:
                        await self._submit_oco_stop(order)
                    
                    break
                    
                elif alpaca_order.status == "partially_filled":
                    if order.state != OrderState.PARTIALLY_FILLED:
                        self._transition_state(order, OrderState.PARTIALLY_FILLED, "Partial fill", {
                            "filled_qty": order.filled_qty,
                            "filled_avg_price": order.filled_avg_price
                        })
                
                elif alpaca_order.status in ["canceled", "expired", "rejected"]:
                    new_state = OrderState.CANCELED if alpaca_order.status == "canceled" else OrderState.REJECTED
                    self._transition_state(order, new_state, f"Alpaca status: {alpaca_order.status}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error monitoring order {order.id}: {str(e)}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _submit_oco_stop(self, order: InternalOrder) -> None:
        """Submit OCO stop loss order after main order fills"""
        
        if not order.oco_stop_price or order.filled_qty == 0:
            return
        
        try:
            # Determine stop side (opposite of main order)
            stop_side = "sell" if order.side == "buy" else "buy"
            
            # Submit stop loss order
            stop_order = await self.alpaca_client.submit_order(
                symbol=order.symbol,
                side=stop_side,
                qty=order.filled_qty,  # Use filled quantity
                order_type="stop",
                stop_price=order.oco_stop_price,
                time_in_force="gtc",  # Good till cancelled
                client_order_id=f"{order.client_order_id}-STOP"
            )
            
            order.oco_stop_order_id = stop_order.id
            
            log_trade_event(
                "oco_stop_created",
                order.symbol,
                order_id=order.id,
                stop_order_id=stop_order.id,
                stop_price=order.oco_stop_price,
                qty=order.filled_qty
            )
            
            logger.info(f"🛑 OCO stop loss created: {stop_order.id} @ ${order.oco_stop_price}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create OCO stop for order {order.id}: {str(e)}")
    
    def get_order(self, order_id: str) -> Optional[InternalOrder]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[InternalOrder]:
        """Get all orders for a symbol"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_active_orders(self) -> List[InternalOrder]:
        """Get all active orders"""
        active_states = [OrderState.NEW, OrderState.SUBMITTED, OrderState.ACCEPTED, OrderState.PARTIALLY_FILLED]
        return [order for order in self.orders.values() if order.state in active_states]
    
    def to_dict(self, order: InternalOrder) -> Dict[str, Any]:
        """Convert internal order to dictionary for API responses"""
        return {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "order_type": order.order_type,
            "state": order.state.value,
            "created_at": order.created_at.isoformat() + "Z",
            "updated_at": order.updated_at.isoformat() + "Z",
            "alpaca_order_id": order.alpaca_order_id,
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
            "oco_stop_price": order.oco_stop_price,
            "oco_stop_order_id": order.oco_stop_order_id,
            "state_history": order.state_history
        }


# Global broker service instance
broker_service = BrokerService()
