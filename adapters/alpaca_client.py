#!/usr/bin/env python3
"""
🦙 ALPACA CLIENT ADAPTER
Modern async Alpaca client with httpx, retries, and proper error handling
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from infra.logging import get_structured_logger
from infra.settings import settings

from .errors import *

logger = get_structured_logger("adapters.alpaca")


@dataclass
class AlpacaOrder:
    """Alpaca order response"""
    id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    submitted_at: datetime
    filled_avg_price: float | None = None
    filled_qty: float | None = None


@dataclass
class AlpacaPosition:
    """Alpaca position response"""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str


@dataclass
class AlpacaAccount:
    """Alpaca account response"""
    account_number: str
    status: str
    buying_power: float
    equity: float
    cash: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool
    last_equity: float = 0.0  # Previous day's equity for daily loss calculation


class AlpacaClient:
    """Async Alpaca API client with modern features"""

    def __init__(self):
        self.api_key = settings.alpaca_api_key
        self.secret_key = settings.alpaca_secret_key
        self.base_url = settings.alpaca_base_url

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not configured")

        # HTTP client configuration
        self.timeout = httpx.Timeout(
            connect=5.0,    # Connection timeout
            read=30.0,      # Read timeout
            write=10.0,     # Write timeout
            pool=60.0       # Total timeout
        )

        self.limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100
        )

        # Headers for authentication
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
            "User-Agent": f"IntraDay-Trading-Platform/{settings.version}"
        }

        self.client = None
        logger.info("🦙 Alpaca client initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
            limits=self.limits
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, AlpacaConnectionError, AlpacaRateLimitError))
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make HTTP request with retries and error handling"""

        if not self.client:
            raise AlpacaError("Client not initialized - use async context manager")

        try:
            response = await self.client.request(method, endpoint, **kwargs)

            # Handle different response codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 201:
                return response.json()
            elif response.status_code == 204:
                return {}  # No content
            else:
                # Parse error response
                try:
                    error_data = response.json()
                except:
                    error_data = {"message": response.text, "code": "PARSE_ERROR"}

                # Map to specific exception
                raise map_alpaca_error(response.status_code, error_data)

        except httpx.RequestError as e:
            logger.error(f"Alpaca request error: {e}")
            raise AlpacaConnectionError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise AlpacaDataError(f"Invalid JSON response: {str(e)}")

    async def submit_order(self, symbol: str, side: str, qty: float,
                          order_type: str = "market", time_in_force: str = "day",
                          limit_price: float | None = None,
                          stop_price: float | None = None,
                          client_order_id: str | None = None) -> AlpacaOrder:
        """Submit an order with idempotency"""

        # Generate idempotency key if not provided
        if not client_order_id:
            client_order_id = str(uuid.uuid4())

        order_data = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id
        }

        if limit_price:
            order_data["limit_price"] = str(limit_price)
        if stop_price:
            order_data["stop_price"] = str(stop_price)

        logger.info(f"📤 Submitting {side} order: {qty} shares of {symbol}")

        response_data = await self._make_request("POST", "/v2/orders", json=order_data)

        logger.info(f"✅ Order submitted: {response_data.get('id')} - Status: {response_data.get('status')}")

        return AlpacaOrder(
            id=response_data["id"],
            symbol=response_data["symbol"],
            side=response_data["side"],
            qty=float(response_data["qty"]),
            order_type=response_data["type"],
            status=response_data["status"],
            submitted_at=datetime.fromisoformat(response_data["submitted_at"].replace("Z", "+00:00")),
            filled_avg_price=float(response_data["filled_avg_price"]) if response_data.get("filled_avg_price") else None,
            filled_qty=float(response_data["filled_qty"]) if response_data.get("filled_qty") else None
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID"""
        logger.info(f"🚫 Cancelling order: {order_id}")

        try:
            await self._make_request("DELETE", f"/v2/orders/{order_id}")
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
        except AlpacaError as e:
            if e.status_code == 404:
                logger.warning(f"⚠️ Order not found for cancellation: {order_id}")
                return False
            raise

    async def get_account(self) -> AlpacaAccount:
        """Get account information"""
        response_data = await self._make_request("GET", "/v2/account")

        return AlpacaAccount(
            account_number=response_data["account_number"],
            status=response_data["status"],
            buying_power=float(response_data["buying_power"]),
            equity=float(response_data["equity"]),
            cash=float(response_data["cash"]),
            portfolio_value=float(response_data["portfolio_value"]),
            day_trade_count=int(response_data["daytrade_count"]),
            pattern_day_trader=response_data["pattern_day_trader"],
            last_equity=float(response_data.get("last_equity", response_data["equity"]))
        )

    async def get_positions(self) -> list[AlpacaPosition]:
        """Get all positions"""
        response_data = await self._make_request("GET", "/v2/positions")

        positions = []
        for pos_data in response_data:
            positions.append(AlpacaPosition(
                symbol=pos_data["symbol"],
                qty=float(pos_data["qty"]),
                market_value=float(pos_data["market_value"]),
                avg_entry_price=float(pos_data["avg_entry_price"]),
                unrealized_pl=float(pos_data["unrealized_pl"]),
                unrealized_plpc=float(pos_data["unrealized_plpc"]),
                side=pos_data["side"]
            ))

        return positions

    async def get_orders(self, status: str | None = None,
                        since: datetime | None = None,
                        limit: int = 50) -> list[AlpacaOrder]:
        """Get orders with optional filtering"""

        params = {"limit": str(limit)}
        if status:
            params["status"] = status
        if since:
            params["after"] = since.isoformat()

        response_data = await self._make_request("GET", "/v2/orders", params=params)

        orders = []
        for order_data in response_data:
            orders.append(AlpacaOrder(
                id=order_data["id"],
                symbol=order_data["symbol"],
                side=order_data["side"],
                qty=float(order_data["qty"]),
                order_type=order_data["type"],
                status=order_data["status"],
                submitted_at=datetime.fromisoformat(order_data["submitted_at"].replace("Z", "+00:00")),
                filled_avg_price=float(order_data["filled_avg_price"]) if order_data.get("filled_avg_price") else None,
                filled_qty=float(order_data["filled_qty"]) if order_data.get("filled_qty") else None
            ))

        return orders

    async def get_order(self, order_id: str) -> AlpacaOrder:
        """Get single order by ID"""
        response_data = await self._make_request("GET", f"/v2/orders/{order_id}")

        return AlpacaOrder(
            id=response_data["id"],
            symbol=response_data["symbol"],
            side=response_data["side"],
            qty=float(response_data["qty"]),
            order_type=response_data["type"],
            status=response_data["status"],
            submitted_at=datetime.fromisoformat(response_data["submitted_at"].replace("Z", "+00:00")),
            filled_avg_price=float(response_data["filled_avg_price"]) if response_data.get("filled_avg_price") else None,
            filled_qty=float(response_data["filled_qty"]) if response_data.get("filled_qty") else None
        )
