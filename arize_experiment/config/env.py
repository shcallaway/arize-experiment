"""
Environment variable management for arize-experiment.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArizeConfig:
    """Configuration for Arize API credentials."""

    api_key: str
    space_id: str


class EnvironmentError(Exception):
    """Raised when required environment variables are missing."""

    pass


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    logger.debug("Environment variables loaded from .env file")


def get_arize_config() -> ArizeConfig:
    """Get Arize API configuration from environment.

    Returns:
        ArizeConfig with API credentials

    Raises:
        EnvironmentError: If required variables are not set
    """
    # Load environment variables first
    load_environment()

    api_key = os.getenv("ARIZE_API_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")

    logger.debug(f"Retrieved API key (length: {len(api_key) if api_key else 0})")
    logger.debug(f"Retrieved space ID (length: {len(space_id) if space_id else 0})")

    if not api_key:
        msg = (
            "ARIZE_API_KEY environment variable is not set.\n"
            "Please set your Arize API key in the .env file:\n"
            "ARIZE_API_KEY=your_api_key_here"
        )
        logger.error(msg)
        raise EnvironmentError(msg)

    if not space_id:
        msg = (
            "ARIZE_SPACE_ID environment variable is not set.\n"
            "Please set your Arize space ID in the .env file:\n"
            "ARIZE_SPACE_ID=your_space_id_here"
        )
        logger.error(msg)
        raise EnvironmentError(msg)

    logger.debug("Successfully loaded Arize configuration")
    return ArizeConfig(api_key=api_key, space_id=space_id)


def get_log_level() -> str:
    """Get configured log level from environment.

    Returns:
        String log level, defaults to "INFO"
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    logger.debug(f"Log level set to: {level}")
    return level
