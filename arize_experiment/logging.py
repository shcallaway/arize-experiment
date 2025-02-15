"""
Centralized logging configuration for arize-experiment.
"""

import logging
import sys
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging for the entire application.

    Args:
        level: Optional logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If not provided, defaults to INFO.
    """
    # Set default level to INFO if not specified
    log_level = getattr(logging, level.upper()) if level else logging.INFO
    
    # Create formatter with consistent format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Set level for third-party loggers to WARNING to reduce noise
    for logger_name in logging.root.manager.loggerDict:
        if not logger_name.startswith('arize_experiment'):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    This is the preferred way to get a logger in the application.
    It ensures consistent naming and configuration.

    Args:
        name: Name for the logger, typically __name__

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
