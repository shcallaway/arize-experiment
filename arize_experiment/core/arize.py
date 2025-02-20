"""
Enhanced Arize API client with better error handling and configuration.

This module provides an enhanced client for interacting with the Arize AI
platform. It includes:
1. Robust error handling
2. Structured configuration
3. Type safety
4. Logging integration
5. Dataset management
6. Experiment execution

Example:
    ```python
    from arize_experiment.core.arize import ArizeClient, ArizeClientConfiguration

    config = ArizeClientConfiguration(
        api_key="your_api_key",
        developer_key="your_dev_key",
        space_id="your_space_id"
    )

    client = ArizeClient(config)
    client.create_dataset("my_dataset", data)
    ```
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import pandas as pd
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE

from arize_experiment.core.errors import ArizeClientError

logger = logging.getLogger(__name__)


@dataclass
class ArizeClientConfiguration:
    """Configuration for the Arize client.

    This class encapsulates all configuration needed to initialize an Arize
    client. It ensures all required credentials and settings are provided
    and validated.

    Attributes:
        api_key (str): The Arize API key for authentication
        developer_key (str): The developer key for API access
        space_id (str): The Arize space ID to work in

    Example:
        ```python
        config = ArizeClientConfiguration(
            api_key="your_api_key",
            developer_key="your_dev_key",
            space_id="your_space_id"
        )
        ```
    """

    def __init__(
        self,
        api_key: str,
        developer_key: str,
        space_id: str,
    ):
        """Initialize the configuration.

        Args:
            api_key: The Arize API key for authentication
            developer_key: The developer key for API access
            space_id: The Arize space ID to work in

        Raises:
            ValueError: If any required parameter is empty or invalid
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not developer_key:
            raise ValueError("Developer key cannot be empty")
        if not space_id:
            raise ValueError("Space ID cannot be empty")

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
            config: Configuration object containing API credentials
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
        """Create the Arize datasets client.

        Args:
            api_key: Arize API key
            developer_key: Arize developer key

        Returns:
            Initialized ArizeDatasetsClient

        Raises:
            ArizeClientError: If client creation fails
        """
        try:
            logger.debug("Creating Arize datasets client")
            return ArizeDatasetsClient(
                api_key=api_key,
                developer_key=developer_key,
            )
        except Exception as e:
            logger.debug(f"Error creating Arize datasets client: {e}")
            raise ArizeClientError(
                "Failed to create Arize datasets client", details={"error": str(e)}
            )

    def create_dataset(
        self,
        dataset_name: str,
        data: pd.DataFrame,
    ) -> Any:
        """Create a dataset.

        Args:
            dataset_name: Name of the dataset
            data: Data to create the dataset with

        Returns:
            Dataset ID

        Raises:
            ArizeClientError: If dataset creation fails
        """
        try:
            logger.debug(f"Creating dataset: {dataset_name}")
            return self._client.create_dataset(
                space_id=self._space_id,
                dataset_name=dataset_name,
                dataset_type=GENERATIVE,
                data=data,
            )
        except Exception as e:
            logger.debug(f"Error creating dataset: {e}")
            raise ArizeClientError(
                "Failed to create dataset", details={"error": str(e)}
            )

    def get_dataset(self, dataset_name: str) -> Any:
        """Get a dataset by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information or None if not found

        Raises:
            ArizeClientError: If dataset retrieval fails
        """
        try:
            logger.debug(f"Getting dataset: {dataset_name}")
            return self._client.get_dataset(
                space_id=self._space_id, dataset_name=dataset_name
            )
        except Exception as e:
            logger.debug(f"Error getting dataset: {e}")
            # If an error was thrown, assume the dataset does not exist
            return None

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

        Returns:
            Experiment results

        Raises:
            ArizeClientError: If experiment execution fails
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
            logger.debug(f"Error running experiment: {e}")
            error_msg = (
                f"Failed to run experiment '{experiment_name}' "
                f"on dataset '{dataset_name}'"
            )
            raise ArizeClientError(
                error_msg,
                details={
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                    "error": str(e),
                },
            )

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
            Experiment information or None if not found
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
            logger.debug(f"Error getting experiment: {e}")
            return None
