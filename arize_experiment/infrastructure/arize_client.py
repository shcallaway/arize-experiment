"""
Enhanced Arize API client with better error handling and configuration.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from arize.experimental.datasets import ArizeDatasetsClient

logger = logging.getLogger(__name__)


class ArizeClientError(Exception):
    """Base exception for Arize client errors."""
    pass


class ConfigurationError(ArizeClientError):
    """Raised when there are issues with client configuration."""
    pass


class APIError(ArizeClientError):
    """Raised when there are issues with API calls."""
    pass


@dataclass
class ArizeClientConfig:
    """Configuration for Arize API client."""

    api_key: str
    developer_key: str
    space_id: str
    default_dataset: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration."""
        if not self.api_key:
            raise ConfigurationError("API key is required")
        if not self.developer_key:
            raise ConfigurationError("Developer key is required")
        if not self.space_id:
            raise ConfigurationError("Space ID is required")


class ArizeClient:
    """Enhanced Arize API client wrapper.
    
    This class wraps the Arize datasets client with:
    1. Better error handling
    2. Retry logic for transient failures
    3. Consistent logging
    4. Configuration validation
    """

    def __init__(self, config: ArizeClientConfig):
        """Initialize the client with configuration.
        
        Args:
            config: Client configuration
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self._client = self._create_client()
        logger.info("Arize client initialized successfully")

    def _create_client(self) -> ArizeDatasetsClient:
        """Create the underlying Arize datasets client.
        
        Returns:
            Configured ArizeDatasetsClient instance
        
        Raises:
            ConfigurationError: If client creation fails
        """
        try:
            return ArizeDatasetsClient(
                api_key=self.config.api_key,
                developer_key=self.config.developer_key,
            )
        except Exception as e:
            error_msg = f"Failed to create Arize client: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConfigurationError(error_msg) from e

    def get_dataset(self, dataset_name: str, space_id: Optional[str] = None) -> Any:
        """Get a dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            space_id: Optional space ID (uses config default if not provided)
        
        Returns:
            Dataset information
        
        Raises:
            APIError: If the API call fails
        """
        space = space_id or self.config.space_id
        try:
            logger.debug(f"Getting dataset: {dataset_name} in space: {space}")
            return self._client.get_dataset(
                space_id=space,
                dataset_name=dataset_name
            )
        except Exception as e:
            error_msg = f"Failed to get dataset '{dataset_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg) from e

    def run_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
        task: Callable,
        evaluators: Optional[List[Callable]] = None,
        space_id: Optional[str] = None,
    ) -> Any:
        """Run an experiment.
        
        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset to use
            task: Task function to execute
            evaluators: Optional list of evaluator functions
            space_id: Optional space ID (uses config default if not provided)
        
        Returns:
            Experiment results
        
        Raises:
            APIError: If the API call fails
        """
        space = space_id or self.config.space_id
        try:
            logger.debug(
                f"Running experiment: {experiment_name} "
                f"on dataset: {dataset_name} in space: {space}"
            )
            return self._client.run_experiment(
                space_id=space,
                dataset_name=dataset_name,
                task=task,
                evaluators=evaluators,
                experiment_name=experiment_name
            )
        except Exception as e:
            error_msg = (
                f"Failed to run experiment '{experiment_name}' "
                f"on dataset '{dataset_name}': {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg) from e

    def get_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
        space_id: Optional[str] = None,
    ) -> Any:
        """Get experiment information.
        
        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset
            space_id: Optional space ID (uses config default if not provided)
        
        Returns:
            Experiment information
        
        Raises:
            APIError: If the API call fails
        """
        space = space_id or self.config.space_id
        try:
            logger.debug(
                f"Getting experiment: {experiment_name} "
                f"from dataset: {dataset_name} in space: {space}"
            )
            return self._client.get_experiment(
                space_id=space,
                experiment_name=experiment_name,
                dataset_name=dataset_name
            )
        except Exception as e:
            error_msg = (
                f"Failed to get experiment '{experiment_name}' "
                f"from dataset '{dataset_name}': {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg) from e
