"""
Enhanced configuration management for arize-experiment.
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from arize_experiment.infrastructure.arize_client import ArizeClientConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there are issues with configuration."""
    pass



class ConfigManager:
    """Manages application configuration.
    
    This class handles:
    1. Loading environment variables
    2. Creating configuration objects
    3. Validating configurations
    4. Providing access to configurations
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self._load_environment()
        self._arize_config: Optional[ArizeClientConfig] = None

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        try:
            load_dotenv()
            logger.debug("Environment variables loaded from .env file")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {str(e)}")

    def get_arize_config(self) -> ArizeClientConfig:
        """Get Arize API configuration.
        
        This method caches the configuration after first creation.
        
        Returns:
            ArizeClientConfig instance
        
        Raises:
            ConfigurationError: If required variables are missing
        """
        if self._arize_config is not None:
            return self._arize_config

        try:
            api_key = self._get_required_env("ARIZE_API_KEY")
            space_id = self._get_required_env("ARIZE_SPACE_ID")
            developer_key = self._get_required_env("ARIZE_DEVELOPER_KEY")

            self._arize_config = ArizeClientConfig(
                api_key=api_key,
                space_id=space_id,
                developer_key=developer_key,
            )

            return self._arize_config

        except Exception as e:
            error_msg = f"Failed to create Arize configuration: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConfigurationError(error_msg) from e

    def _get_required_env(self, name: str) -> str:
        """Get a required environment variable.
        
        Args:
            name: Name of the environment variable
        
        Returns:
            Value of the environment variable
        
        Raises:
            ConfigurationError: If the variable is not set
        """
        value = os.getenv(name)
        if not value:
            error_msg = (
                f"{name} environment variable is not set.\n"
                f"Please set it in your .env file:\n"
                f"{name}=your_value_here"
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        return value


# Global configuration manager instance
config_manager = ConfigManager()
