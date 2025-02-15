"""
Logging configuration and utilities for arize-experiment.
"""

import os
import sys
import logging
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging with proper formatting and level.

    Args:
        level: Optional logging level override. If not provided,
              uses LOG_LEVEL environment variable or defaults to INFO.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers and add our stdout handler
    root_logger.handlers = []
    root_logger.addHandler(stdout_handler)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {log_level}")

    # Suppress verbose logging from dependencies
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("arize").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Name for the logger, typically __name__

    Returns:
        Logger instance configured with proper formatting
    """
    logger = logging.getLogger(name)
    logger.debug(f"Created logger: {name}")
    return logger
