#!/usr/bin/env python3
"""
📝 STRUCTURED LOGGING CONFIGURATION
Request ID tracking and JSON structured logging for FastAPI
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Context variable for request ID tracking
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and track request IDs"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Set in context for logging
        request_id_context.set(request_id)

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_context.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "stack_info",
                    "exc_info",
                    "exc_text",
                    "message",
                ]:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter"""

    def format(self, record: logging.LogRecord) -> str:
        # Get request ID
        request_id = request_id_context.get()
        request_part = f" [{request_id[:8]}]" if request_id else ""

        # Format message
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        base_msg = f"{timestamp} | {record.levelname:8} | {record.name}{request_part} | {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"

        return base_msg


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure application logging"""

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Set formatter
    if log_format.lower() == "json":
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("alpaca_trade_api").setLevel(logging.INFO)

    # Set our application loggers
    logging.getLogger("backend").setLevel(level)
    logging.getLogger("services").setLevel(level)
    logging.getLogger("adapters").setLevel(level)


def get_structured_logger(name: str) -> logging.Logger:
    """Get a logger with structured formatting"""
    return logging.getLogger(name)


def log_trade_event(event_type: str, symbol: str, **kwargs) -> None:
    """Log structured trading events"""
    logger = get_structured_logger("trading")

    log_data = {
        "event_type": event_type,
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }

    logger.info(f"Trade Event: {event_type}", extra=log_data)


def log_risk_event(event_type: str, **kwargs) -> None:
    """Log structured risk management events"""
    logger = get_structured_logger("risk")

    log_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }

    logger.info(f"Risk Event: {event_type}", extra=log_data)


def log_performance_metric(metric_name: str, value: float, **kwargs) -> None:
    """Log performance metrics"""
    logger = get_structured_logger("metrics")

    log_data = {
        "metric_name": metric_name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }

    logger.info(f"Performance Metric: {metric_name}={value}", extra=log_data)
