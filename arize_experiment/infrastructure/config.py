"""
Enhanced configuration management for arize-experiment.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from arize_experiment.infrastructure.arize_client import ArizeClientConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there are issues with configuration."""
    pass


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    dataset: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    evaluators: Optional[list[str]] = None

    def __post_init__(self):
        """Validate the configuration."""
        if not self.name:
            raise ConfigurationError("Experiment name is required")
        if not self.dataset:
            raise ConfigurationError("Dataset name is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns:
            Dict containing the configuration
        """
        config = {
            "name": self.name,
            "dataset": self.dataset,
        }

        if self.description:
            config["description"] = self.description
        if self.tags:
            config["tags"] = self.tags
        if self.evaluators:
            config["evaluators"] = self.evaluators

        return config


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
            default_dataset = os.getenv("DATASET")

            self._arize_config = ArizeClientConfig(
                api_key=api_key,
                space_id=space_id,
                developer_key=developer_key,
                default_dataset=default_dataset
            )

            return self._arize_config

        except Exception as e:
            error_msg = f"Failed to create Arize configuration: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConfigurationError(error_msg) from e

    def create_experiment_config(
        self,
        name: str,
        dataset: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        evaluators: Optional[list[str]] = None,
    ) -> ExperimentConfig:
        """Create a new experiment configuration.
        
        Args:
            name: Name of the experiment
            dataset: Name of the dataset to use
            description: Optional experiment description
            tags: Optional key-value pairs for experiment metadata
            evaluators: Optional list of evaluator names to use
        
        Returns:
            ExperimentConfig instance
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            return ExperimentConfig(
                name=name,
                dataset=dataset,
                description=description,
                tags=tags,
                evaluators=evaluators,
            )
        except Exception as e:
            error_msg = f"Failed to create experiment configuration: {str(e)}"
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
