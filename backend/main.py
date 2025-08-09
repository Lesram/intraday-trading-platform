#!/usr/bin/env python3
"""
🏗️ FASTAPI APPLICATION FACTORY
Main FastAPI app factory with lifespan context and proper middleware setup
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import routers (will create these in subsequent steps)
from api.routers import health, portfolio, signals, trading, websocket
from infra.logging import RequestIDMiddleware, get_structured_logger, setup_logging

# Import infrastructure
from infra.settings import settings

# Initialize logger
logger = get_structured_logger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("🚀 Starting Intraday Trading Platform")

    try:
        # Initialize components
        logger.info("📡 Initializing trading systems...")

        # Initialize HTTP client manager
        from infra.http_client import http_manager

        await http_manager.initialize()

        # Here we would initialize:
        # - Database connections
        # - ML model loading
        # - Trading engine startup
        # - Cache initialization
        # - Background tasks

        logger.info("✅ Trading platform started successfully")
        yield

    except Exception as e:
        logger.error(f"❌ Failed to start trading platform: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down trading platform...")

        # Close HTTP client
        from infra.http_client import http_manager

        await http_manager.close()

        # Cleanup:
        # - Close database connections
        # - Cancel background tasks
        # - Save state
        # - Close trading sessions

        logger.info("✅ Trading platform shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    # Setup logging first
    setup_logging(settings.log_level, settings.log_format)

    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Advanced Intraday Trading Platform with ML Integration, Risk Management, and Optimization Systems",
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Add middleware (order matters!)

    # 1. Trusted Host Middleware (security)
    if settings.allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 3. Request ID Middleware (for logging)
    app.add_middleware(RequestIDMiddleware)

    # Mount routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
    app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
    app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "status": "operational",
        }

    # Direct health endpoint for tests (alias to /api/health)
    @app.get("/health")
    async def health_check():
        """Direct health check endpoint for tests"""
        from datetime import datetime

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": settings.version,
            "environment": settings.environment,
        }

    logger.info(f"📱 FastAPI app created - Environment: {settings.environment}")

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🌐 Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )
