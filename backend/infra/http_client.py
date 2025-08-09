#!/usr/bin/env python3
"""
🌐 HTTP CLIENT MANAGEMENT
Proper httpx client lifespan management for FastAPI
"""

from contextlib import asynccontextmanager

import httpx

from infra.logging import get_structured_logger

logger = get_structured_logger(__name__)


class HTTPClientManager:
    """
    Global HTTP client manager with proper lifecycle
    Ensures clients are created/closed with FastAPI app lifespan
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._timeout = httpx.Timeout(30.0, connect=10.0)
        self._limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)

    async def initialize(self):
        """Initialize the HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=self._limits,
                headers={"User-Agent": "IntraDay-Trading-Platform/1.0"},
            )
            logger.info("🌐 HTTP client initialized")
        return self._client

    async def close(self):
        """Close the HTTP client and cleanup connections"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("🌐 HTTP client closed")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the current HTTP client (must be initialized first)"""
        if self._client is None:
            raise RuntimeError("HTTP client not initialized. Call initialize() first.")
        return self._client

    def is_initialized(self) -> bool:
        """Check if client is initialized"""
        return self._client is not None


# Global instance
http_manager = HTTPClientManager()


def get_http_client() -> httpx.AsyncClient:
    """Dependency for getting the HTTP client in FastAPI endpoints"""
    return http_manager.client


@asynccontextmanager
async def http_client_lifespan():
    """Context manager for HTTP client lifecycle in tests"""
    await http_manager.initialize()
    try:
        yield http_manager.client
    finally:
        await http_manager.close()
