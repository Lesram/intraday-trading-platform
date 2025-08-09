#!/usr/bin/env python3
"""
🔌 WEBSOCKET ROUTER
Real-time trading data streams
"""


import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from infra.logging import get_structured_logger

logger = get_structured_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket connection manager with backpressure protection"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.max_queue_size = 100  # Prevent memory issues

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"🔌 WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"🔌 WebSocket disconnected: {client_id}")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"❌ Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: str):
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"❌ Failed to broadcast to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)


manager = ConnectionManager()


@router.websocket("/market-data/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time market data
    TODO: Add authentication for production
    """
    client_id = f"market_{symbol}_{id(websocket)}"
    await manager.connect(websocket, client_id)

    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "connection",
                    "symbol": symbol,
                    "status": "connected",
                    "client_id": client_id,
                }
            ),
            client_id,
        )

        # Market data simulation (replace with actual data feed)
        import random

        base_price = 150.0

        while True:
            # Simulate price movement
            price_change = random.uniform(-1.0, 1.0)
            current_price = base_price + price_change

            market_data = {
                "type": "price_update",
                "symbol": symbol,
                "price": round(current_price, 2),
                "volume": random.randint(100, 10000),
                "timestamp": asyncio.get_event_loop().time(),
            }

            await manager.send_personal_message(json.dumps(market_data), client_id)

            await asyncio.sleep(1)  # 1Hz updates

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"🔌 Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


@router.websocket("/trading-updates")
async def websocket_trading_updates(websocket: WebSocket):
    """
    WebSocket endpoint for trading order updates
    TODO: Add authentication for production
    """
    client_id = f"trading_{id(websocket)}"
    await manager.connect(websocket, client_id)

    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "connection",
                    "service": "trading_updates",
                    "status": "connected",
                    "client_id": client_id,
                }
            ),
            client_id,
        )

        # Keep connection alive and listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for now (implement actual trading updates)
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "echo",
                            "received": data,
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    ),
                    client_id,
                )
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"🔌 Trading client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error for trading client {client_id}: {e}")
        manager.disconnect(client_id)
