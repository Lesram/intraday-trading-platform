#!/usr/bin/env python3
"""
❌ ADAPTER ERRORS
Custom exception classes for external service adapters
"""

from typing import Optional, Dict, Any


class AdapterError(Exception):
    """Base exception for all adapter errors"""
    
    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.service = service
        self.details = details or {}
        super().__init__(self.message)


class AlpacaError(AdapterError):
    """Alpaca API specific errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message, "alpaca", details)


class AlpacaConnectionError(AlpacaError):
    """Network/connection errors with Alpaca"""
    pass


class AlpacaAuthenticationError(AlpacaError):
    """Authentication errors with Alpaca"""
    pass


class AlpacaRateLimitError(AlpacaError):
    """Rate limiting errors from Alpaca"""
    pass


class AlpacaOrderError(AlpacaError):
    """Order-specific errors from Alpaca"""
    pass


class AlpacaDataError(AlpacaError):
    """Data retrieval errors from Alpaca"""
    pass


class AlpacaInsufficientFundsError(AlpacaError):
    """Insufficient buying power for order"""
    pass


class AlpacaMarketClosedError(AlpacaError):
    """Market is closed for trading"""
    pass


def map_alpaca_error(status_code: int, error_response: Dict[str, Any]) -> AlpacaError:
    """Map HTTP status codes and Alpaca error responses to specific exceptions"""
    
    message = error_response.get('message', 'Unknown Alpaca error')
    code = error_response.get('code', 'UNKNOWN')
    
    if status_code == 401:
        return AlpacaAuthenticationError(message, status_code, code, error_response)
    elif status_code == 403:
        if 'insufficient' in message.lower():
            return AlpacaInsufficientFundsError(message, status_code, code, error_response)
        else:
            return AlpacaAuthenticationError(message, status_code, code, error_response)
    elif status_code == 422:
        if 'market' in message.lower() and 'closed' in message.lower():
            return AlpacaMarketClosedError(message, status_code, code, error_response)
        else:
            return AlpacaOrderError(message, status_code, code, error_response)
    elif status_code == 429:
        return AlpacaRateLimitError(message, status_code, code, error_response)
    elif status_code >= 500:
        return AlpacaConnectionError(message, status_code, code, error_response)
    else:
        return AlpacaError(message, status_code, code, error_response)
