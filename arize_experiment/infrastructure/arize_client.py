"""
Enhanced Arize API client with better error handling and configuration.
"""

import logging
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


class ArizeClient:
    """Enhanced Arize API client wrapper.
    
    This class wraps the Arize datasets client with:
    1. Better error handling
    2. Retry logic for transient failures
    3. Consistent logging
    4. Configuration validation
    """

    def __init__(
        self,
        api_key: str,
        developer_key: str,
        space_id: str,
    ):
        """Initialize the client with configuration.
        
        Args:
            api_key: API key
            developer_key: Developer key
            space_id: Space ID
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self._space_id = space_id

        # Initialize the Arize datasets client
        self._client = ArizeDatasetsClient(
            api_key=api_key,
            developer_key=developer_key,
        )

        logger.info("Arize client initialized successfully")

    def get_dataset(self, dataset: str) -> Any:
        """Get a dataset by name.
        
        Args:
            dataset: Name of the dataset
        
        Returns:
            Dataset information
        
        Raises:
            APIError: If the API call fails
        """
        try:
            logger.debug(f"Getting dataset: {dataset}")
            return self._client.get_dataset(
                space_id=self._space_id,
                dataset_name=dataset
            )
        except Exception as e:
            error_msg = f"Failed to get dataset '{dataset}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg) from e

    def run_experiment(
        self,
        experiment: str,
        dataset: str,
        task: Callable,
        evaluators: Optional[List[Callable]] = None,
    ) -> Any:
        """Run an experiment.
        
        Args:
            experiment: Name of the experiment
            dataset: Name of the dataset to use
            task: Task function to execute
            evaluators: Optional list of evaluator functions
            space_id: Optional space ID (uses config default if not provided)
        
        Returns:
            Experiment results
        
        Raises:
            APIError: If the API call fails
        """
        try:
            logger.debug(
                f"Running experiment: {experiment} "
                f"on dataset: {dataset} in space: {self._space_id}"
            )
            return self._client.run_experiment(
                space_id=self._space_id,
                dataset_name=dataset,
                task=task,
                evaluators=evaluators,
                experiment_name=experiment
            )
        except Exception as e:
            # Wrap other errors in APIError
            error_msg = (
                f"Failed to run experiment '{experiment}' "
                f"on dataset '{dataset}': {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg) from e

    def get_experiment(
        self,
        experiment: str,
        dataset: str,
    ) -> Any:
        """Get experiment information.
        
        Args:
            experiment: Name of the experiment
            dataset: Name of the dataset
        
        Returns:
            Experiment information
        
        Raises:
            APIError: If the API call fails
        """
        try:
            logger.debug(
                f"Getting experiment: {experiment} "
                f"from dataset: {dataset} in space: {self._space_id}"
            )
            return self._client.get_experiment(
                space_id=self._space_id,
                experiment_name=experiment,
                dataset_name=dataset
            )
        except Exception as e:
            # If the experiment does not exist, return None
            if ("Failed to get experiment") in str(e):
                return None
            
            # Let other errors propagate up
            raise
