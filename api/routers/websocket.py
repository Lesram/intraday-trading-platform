#!/usr/bin/env python3
"""
🔌 WEBSOCKET ROUTER
Real-time WebSocket connections for live data
"""


from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live data"""
    await websocket.accept()

    try:
        while True:
            # Keep connection alive with periodic data
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": "2025-08-08T12:00:00Z"
            })

            await websocket.receive_text()  # Wait for client message

    except Exception:
        pass  # Client disconnected
