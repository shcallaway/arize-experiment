"""
Base command class for CLI commands.

This module provides the base command class that all CLI commands inherit from.
It provides shared functionality like Arize client initialization and common
validation methods.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, final

from arize_experiment.core.arize import ArizeClient, ArizeClientConfiguration
from arize_experiment.core.errors import ConfigurationError, HandlerError
from arize_experiment.core.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class BaseCommand(ABC):
    """Base class for CLI commands.

    This class provides shared functionality for all CLI commands including:
    1. Arize client initialization
    2. Environment variable handling
    3. Common validation methods
    """

    def __init__(self) -> None:
        """Initialize the command.

        This sets up the schema validator and Arize client needed for
        command execution. Raises appropriate errors if initialization fails.

        Raises:
            HandlerError: If handler initialization fails
            ConfigurationError: If configuration is invalid
        """
        self._schema_validator = SchemaValidator()
        self._arize_client = self._initialize_arize_client()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> None:
        """Execute the command.

        This abstract method must be implemented by all command classes.
        It defines the main execution logic for the command.

        Args:
            *args: Positional arguments for the command
            **kwargs: Keyword arguments for the command

        Raises:
            NotImplementedError: If the child class does not implement this method
        """
        pass

    @final
    def _initialize_arize_client(self) -> ArizeClient:
        """Initialize the Arize client.

        Returns:
            Arize client instance
        """
        # Get Arize values from env
        logger.debug("Creating Arize client configuration")
        try:
            arize_config = ArizeClientConfiguration(
                api_key=self._get_arize_api_key(),
                space_id=self._get_arize_space_id(),
                developer_key=self._get_arize_developer_key(),
            )
        except Exception as e:
            raise HandlerError(
                "Failed to create Arize client configuration", details={"error": str(e)}
            )

        # Initialize Arize client
        logger.debug("Initializing Arize client")
        try:
            arize_client = ArizeClient(
                config=arize_config,
            )
        except Exception as e:
            raise HandlerError(
                "Failed to initialize Arize client", details={"error": str(e)}
            )

        return arize_client

    @final
    def _get_arize_api_key(self) -> str:
        """Get the Arize API key.

        Returns:
            Arize API key
        """
        return self._get_required_env("ARIZE_API_KEY")

    @final
    def _get_arize_space_id(self) -> str:
        """Get the Arize space ID.

        Returns:
            Arize space ID
        """
        return self._get_required_env("ARIZE_SPACE_ID")

    @final
    def _get_arize_developer_key(self) -> str:
        """Get the Arize developer key.

        Returns:
            Arize developer key
        """
        return self._get_required_env("ARIZE_DEVELOPER_KEY")

    @final
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

        # If the variable is not set, raise an error
        if not value:
            error_msg = (
                f"{name} environment variable is not set.\n"
                f"Please set it in your .env file:\n"
                f"{name}=your_value_here"
            )
            raise ConfigurationError(error_msg)

        return value

    @final
    def _verify_dataset_exists(self, dataset_name: str) -> None:
        """Verify that a dataset exists.

        Args:
            dataset_name: Name of the dataset to verify
        """
        if not self._dataset_exists(dataset_name):
            raise ConfigurationError(
                f"Dataset '{dataset_name}' does not exist",
                details={"dataset": dataset_name},
            )

    @final
    def _verify_dataset_does_not_exist(self, dataset_name: str) -> None:
        """Verify that a dataset does not exist.

        Args:
            dataset_name: Name of the dataset to verify
        """
        if self._dataset_exists(dataset_name):
            raise ConfigurationError(
                f"Dataset '{dataset_name}' already exists",
                details={"dataset": dataset_name},
            )

    @final
    def _verify_experiment_does_not_exist(
        self, experiment_name: str, dataset_name: str
    ) -> None:
        """Verify that an experiment does not exist.

        Args:
            experiment_name: Name of the experiment to verify
            dataset_name: Name of the dataset
        """
        if self._experiment_exists(experiment_name, dataset_name):
            raise ConfigurationError(
                f"Experiment '{experiment_name}' already exists",
                details={
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                },
            )

    @final
    def _dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists.

        Args:
            dataset_name: Name of the dataset to check
        """
        logger.info(f"Checking if dataset '{dataset_name}' exists")
        try:
            dataset_exists = self._arize_client.get_dataset(
                dataset_name=dataset_name,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to check if dataset '{dataset_name}' exists",
                details={"error": str(e)},
            )

        return dataset_exists is not None

    @final
    def _experiment_exists(self, experiment_name: str, dataset_name: str) -> bool:
        """Check if an experiment exists.

        Args:
            experiment_name: Name of the experiment to check
            dataset_name: Name of the dataset
        """
        logger.info(f"Checking if experiment '{experiment_name}' exists")
        try:
            experiment_exists = self._arize_client.get_experiment(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
            )
        except Exception as e:
            raise HandlerError(
                f"Failed to check if experiment '{experiment_name}' exists",
                details={"error": str(e)},
            )

        return experiment_exists is not None
