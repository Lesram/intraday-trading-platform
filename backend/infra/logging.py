#!/usr/bin/env python3
"""
📝 STRUCTURED LOGGING
Simplified logging infrastructure for trading platform
"""

import logging
import sys

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get structured logger for module

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


# Global logger for quick access
logger = get_structured_logger(__name__)

if __name__ == "__main__":
    # Test logging
    test_logger = get_structured_logger("test")
    test_logger.info("📝 Structured logging system ready")
    test_logger.debug("Debug message (may not show)")
    test_logger.warning("⚠️ Warning message")
    test_logger.error("❌ Error message")
    print("✅ Logging system test completed")
