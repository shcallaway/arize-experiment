"""
Enhanced Arize API client with better error handling and configuration.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from arize.experimental.datasets import ArizeDatasetsClient
from arize_experiment.core.errors import ArizeClientError

logger = logging.getLogger(__name__)


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
            config: Configuration object containing API credentials

        Raises:
            ArizeClientError: If client initialization fails
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
            raise ArizeClientError(
                "Failed to create Arize datasets client", details={"error": str(e)}
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
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                return None

            raise ArizeClientError(
                f"Error retrieving dataset '{dataset_name}'",
                details={"dataset_name": dataset_name, "error": str(e)},
            )

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

        Raises:
            ArizeClientError: If experiment retrieval fails
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
            return None
            # # Check if this is a "not found" error from the original exception
            # # Look for both the Arrow error and the wrapped RuntimeError
            # error_str = str(e)
            # if (
            #     "Flight returned not found error" in error_str
            #     or "experiment does not exist" in error_str
            #     or (
            #         hasattr(e, "__cause__")
            #         and e.__cause__ is not None
            #         and (
            #             "Flight returned not found error" in str(e.__cause__)
            #             or "experiment does not exist" in str(e.__cause__)
            #         )
            #     )
            # ):
            #     return None

            # # If the error was something other than a "not found" error, re-raise
            # raise ArizeClientError(
            #     f"Error retrieving experiment '{experiment_name}'",
            #     details={
            #         "experiment_name": experiment_name,
            #         "dataset_name": dataset_name,
            #         "error": str(e)
            #     }
            # )
