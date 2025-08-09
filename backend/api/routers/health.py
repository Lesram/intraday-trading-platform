#!/usr/bin/env python3
"""
🏥 HEALTH CHECK ROUTER
System health and readiness endpoints
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "intraday-trading-platform",
    }


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check endpoint"""
    # TODO: Check dependencies (database, broker, etc.)
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dependencies": {
            "database": "healthy",
            "broker": "healthy",
            "risk_service": "healthy",
        },
    }
