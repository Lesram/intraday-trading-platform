#!/usr/bin/env python3
"""
🚀 FASTAPI METRICS ENDPOINT
Prometheus metrics exposure endpoint for trading observability
"""

import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse

from infra.logging import get_structured_logger
from infra.metrics import get_trading_metrics

logger = get_structured_logger(__name__)

# Create FastAPI app for metrics
app = FastAPI(
    title="Trading Metrics API",
    description="Prometheus metrics and observability for trading platform",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "trading-metrics",
        "timestamp": asyncio.get_event_loop().time(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """
    Prometheus metrics endpoint

    Returns all trading metrics in Prometheus exposition format
    """
    try:
        metrics = get_trading_metrics()
        prometheus_data = metrics.get_metrics()

        logger.debug(f"📊 Metrics exported: {len(prometheus_data)} bytes")

        return Response(content=prometheus_data, media_type="text/plain; charset=utf-8")

    except Exception as e:
        logger.error(f"❌ Metrics export failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metrics export failed: {str(e)}"
        ) from e


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Human-readable metrics summary for debugging
    """
    try:
        metrics = get_trading_metrics()
        summary = metrics.get_metrics_summary()

        return {
            "status": "success",
            "metrics_summary": summary,
            "endpoint_info": {
                "prometheus_endpoint": "/metrics",
                "format": "Prometheus exposition format",
                "content_type": "text/plain",
            },
        }

    except Exception as e:
        logger.error(f"❌ Metrics summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}") from e


@app.get("/metrics/test")
async def test_metrics():
    """
    Generate test metrics for verification
    """
    try:
        logger.info("🧪 Generating test metrics...")

        metrics = get_trading_metrics()

        # Generate sample metrics
        metrics.record_order_submitted("AAPL", "buy", "momentum_test")
        metrics.record_order_filled("AAPL", "buy")
        metrics.record_risk_block("position_limit", "TSLA", "medium")
        metrics.record_order_submit_latency("AAPL", "buy", 0.025)
        metrics.record_risk_check_latency("AAPL", 0.010)

        # Update portfolio metrics
        metrics.update_portfolio_metrics(
            {
                "portfolio_value": 75000.0,
                "buying_power": 35000.0,
                "position_count": 3,
                "daily_pnl": 850.0,
                "portfolio_heat": 0.12,
            }
        )

        # System health
        metrics.set_broker_connection_status(True)
        metrics.set_risk_service_status(True)
        metrics.update_market_data_lag("test_source", 0.08)

        return {
            "status": "success",
            "message": "Test metrics generated",
            "sample_metrics": metrics.get_metrics_summary(),
            "next_steps": [
                "Visit /metrics for Prometheus format",
                "Visit /metrics/summary for human-readable format",
                "Configure Prometheus to scrape /metrics endpoint",
            ],
        }

    except Exception as e:
        logger.error(f"❌ Test metrics generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}") from e


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Trading Metrics API",
        "version": "1.0.0",
        "description": "Prometheus metrics and observability for trading platform",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Prometheus metrics (text/plain)",
            "/metrics/summary": "Human-readable metrics summary",
            "/metrics/test": "Generate test metrics",
        },
        "integration": {
            "prometheus_config": {
                "job_name": "trading-platform",
                "metrics_path": "/metrics",
                "scrape_interval": "15s",
            },
            "grafana_dashboard": "Import trading metrics dashboard",
            "alerting": "Configure alerts on order_submit_latency, risk_blocks_total",
        },
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize metrics on startup"""
    logger.info("🚀 Trading Metrics API starting up...")

    # Initialize metrics
    metrics = get_trading_metrics()

    # Set initial system status
    metrics.set_broker_connection_status(False)  # Will be updated by actual services
    metrics.set_risk_service_status(False)  # Will be updated by actual services

    logger.info("📊 Metrics API ready for Prometheus scraping")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Trading Metrics API shutting down...")


def run_metrics_server(host: str = "0.0.0.0", port: int = 8081):
    """
    Run the metrics server

    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to bind to (default: 8081)
    """
    logger.info(f"🌐 Starting Trading Metrics API on http://{host}:{port}")
    logger.info(f"📊 Prometheus metrics available at http://{host}:{port}/metrics")

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Metrics API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    parser.add_argument(
        "--test", action="store_true", help="Generate test metrics on startup"
    )

    args = parser.parse_args()

    if args.test:
        # Generate test metrics before starting server
        print("🧪 Generating test metrics...")
        asyncio.run(test_metrics())

    run_metrics_server(host=args.host, port=args.port)
