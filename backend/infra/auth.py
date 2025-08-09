#!/usr/bin/env python3
"""
🔐 AUTHENTICATION MIDDLEWARE
Secure trading endpoints with API key or OAuth2 authentication
"""

import os
import secrets
import time
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from infra.logging import get_structured_logger
from infra.settings import settings

logger = get_structured_logger(__name__)

# Security schemes
security_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthError(HTTPException):
    """Authentication error with structured response"""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_api_keys() -> set:
    """Get valid API keys from environment"""
    api_keys_env = os.getenv("TRADING_API_KEYS", "")
    if not api_keys_env:
        # Generate a default key for development (not for production!)
        if settings.environment == "development":
            default_key = "dev-key-" + secrets.token_urlsafe(16)
            logger.warning(f"🔓 Using default API key for development: {default_key}")
            return {default_key}
        else:
            logger.error("❌ No TRADING_API_KEYS configured for production")
            return set()

    keys = {key.strip() for key in api_keys_env.split(",") if key.strip()}
    logger.info(f"🔑 Loaded {len(keys)} API keys")
    return keys


# Cache API keys on module load
VALID_API_KEYS = get_api_keys()


async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    """Verify API key from header"""
    if not api_key:
        logger.warning("🔓 Missing API key in request")
        raise AuthError("API key required")

    if not VALID_API_KEYS:
        logger.error("❌ No valid API keys configured")
        raise AuthError("Authentication service unavailable")

    if api_key not in VALID_API_KEYS:
        logger.warning(f"🔓 Invalid API key attempted: {api_key[:8]}...")
        raise AuthError("Invalid API key")

    logger.debug(f"✅ Valid API key: {api_key[:8]}...")
    return api_key


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),
) -> str:
    """Verify Bearer token (for future OAuth2 implementation)"""
    if not credentials:
        logger.warning("🔓 Missing Bearer token in request")
        raise AuthError("Bearer token required")

    token = credentials.credentials

    # For now, treat bearer tokens as API keys
    # TODO: Implement proper JWT/OAuth2 validation
    if token not in VALID_API_KEYS:
        logger.warning(f"🔓 Invalid Bearer token attempted: {token[:8]}...")
        raise AuthError("Invalid Bearer token")

    logger.debug(f"✅ Valid Bearer token: {token[:8]}...")
    return token


async def verify_trading_auth(
    api_key: str | None = Depends(api_key_header),
    bearer_token: HTTPAuthorizationCredentials | None = Depends(security_bearer),
) -> str:
    """
    Flexible authentication - accepts either API key or Bearer token
    Used for protecting trading endpoints
    """
    auth_errors = []

    # Try API key first
    if api_key:
        try:
            return await verify_api_key(api_key)
        except AuthError as e:
            auth_errors.append(f"API key: {e.detail}")

    # Try Bearer token
    if bearer_token:
        try:
            return await verify_bearer_token(bearer_token)
        except AuthError as e:
            auth_errors.append(f"Bearer token: {e.detail}")

    # If no auth method provided or all failed
    if not api_key and not bearer_token:
        logger.warning("🔓 No authentication provided")
        raise AuthError(
            "Authentication required. Provide X-API-Key header or Authorization: Bearer token"
        )

    logger.warning(f"🔓 Authentication failed: {'; '.join(auth_errors)}")
    raise AuthError("Authentication failed")


# Dependency aliases for easy use
RequireAuth = Annotated[str, Depends(verify_trading_auth)]
RequireAPIKey = Annotated[str, Depends(verify_api_key)]
RequireBearer = Annotated[str, Depends(verify_bearer_token)]


def generate_api_key() -> str:
    """Generate a new API key for administrative purposes"""
    return "trading-" + secrets.token_urlsafe(32)


# Middleware for request-level auth logging
async def log_auth_attempt(request, call_next):
    """Log authentication attempts (optional middleware)"""
    start_time = time.time()

    # Check if this is a protected endpoint
    protected_paths = ["/api/trading", "/api/portfolio"]
    is_protected = any(request.url.path.startswith(path) for path in protected_paths)

    if is_protected:
        api_key = request.headers.get("X-API-Key", "")
        auth_header = request.headers.get("Authorization", "")
        logger.info(
            f"🔐 Auth attempt on {request.url.path} - API key: {'✅' if api_key else '❌'}, Bearer: {'✅' if auth_header.startswith('Bearer ') else '❌'}"
        )

    response = await call_next(request)

    process_time = time.time() - start_time
    if (
        is_protected
        and hasattr(response, "status_code")
        and response.status_code == 401
    ):
        logger.warning(f"🔓 Auth failed on {request.url.path} ({process_time:.3f}s)")

    return response
