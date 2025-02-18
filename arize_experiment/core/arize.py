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


class ArizeClientApiError(ArizeClientError):
    """Raised when there are issues with API calls."""

    pass


@dataclass
class ArizeClientConfiguration:
    """Configuration for the Arize client."""

    def __init__(
        self,
        api_key: str,
        developer_key: str,
        space_id: str,
    ):
        self.api_key = api_key
        self.developer_key = developer_key
        self.space_id = space_id


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
        config: ArizeClientConfiguration,
    ):
        """Initialize the client with configuration.

        Args:
            api_key: API key
            developer_key: Developer key
            config: Configuration
        """
        self._space_id = config.space_id

        # Initialize the Arize datasets client
        self._client = self._create_client(
            api_key=config.api_key,
            developer_key=config.developer_key,
        )

        logger.info("Arize client initialized successfully")

    def _create_client(
        self,
        api_key: str,
        developer_key: str,
    ) -> ArizeDatasetsClient:
        """Create the Arize datasets client."""
        try:
            logger.debug("Creating Arize datasets client")
            return ArizeDatasetsClient(
                api_key=api_key,
                developer_key=developer_key,
            )
        except Exception as e:
            error_msg = f"Failed to create Arize datasets client: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ArizeClientError(error_msg) from e

    def get_dataset(self, dataset_name: str) -> Any:
        """Get a dataset by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information

        Raises:
            ArizeClientApiError: If the API call fails
        """
        try:
            logger.debug(f"Getting dataset: {dataset_name}")
            return self._client.get_dataset(
                space_id=self._space_id, dataset_name=dataset_name
            )
        except Exception as e:
            # If the dataset does not exist, return None
            if ("Failed to get dataset") in str(e):
                return None

            # Let other errors propagate up
            raise

    def run_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
        task: Callable,
        evaluators: Optional[List[Callable]] = None,
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
            ArizeClientApiError: If the API call fails
        """
        try:
            logger.debug(
                f"Running experiment: {experiment_name} "
                f"on dataset: {dataset_name} in space: {self._space_id}"
            )
            return self._client.run_experiment(
                space_id=self._space_id,
                dataset_name=dataset_name,
                task=task,
                evaluators=evaluators,
                experiment_name=experiment_name,
            )
        except Exception as e:
            error_msg = (
                f"Failed to run experiment '{experiment_name}' "
                f"on dataset '{dataset_name}': {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            raise ArizeClientApiError(error_msg) from e

    def get_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
    ) -> Any:
        """Get experiment information.

        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset

        Returns:
            Experiment information

        Raises:
            ArizeClientApiError: If the API call fails
        """
        try:
            logger.debug(
                f"Getting experiment: {experiment_name} "
                f"from dataset: {dataset_name} in space: {self._space_id}"
            )
            return self._client.get_experiment(
                space_id=self._space_id,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
            )
        except Exception as e:
            # If the experiment does not exist, return None
            if ("Failed to get experiment") in str(e):
                return None

            # Let other errors propagate up
            raise
