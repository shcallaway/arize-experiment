"""
Command handlers for the CLI interface.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import click

from arize_experiment.core.arize import ArizeClient, ArizeClientConfiguration
from arize_experiment.core.errors import ConfigurationError, HandlerError
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task import Task
from arize_experiment.core.task_registry import TaskRegistry
from arize_experiment.core.validation import SchemaValidator

logger = logging.getLogger(__name__)


class Handler:
    """Handles CLI command execution.

    This class coordinates between the CLI interface and the application's
    core services. It handles:
    1. Parameter validation and processing
    2. Service coordination
    3. Error handling and user feedback
    4. Schema validation
    """

    def __init__(self) -> None:
        """Initialize the command handler."""
        self.schema_validator = SchemaValidator()

    def run(  # noqa: C901
        self,
        experiment_name: str,
        dataset_name: str,
        task_name: str,
        raw_tags: Optional[List[str]] = None,
        evaluator_names: Optional[List[str]] = None,
    ) -> None:
        """Handle the run experiment command.

        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset
            task_name: Name of the task
            raw_tags: Optional list of key=value tag strings
            evaluator_names: Optional list of evaluator names

        Raises:
            HandlerError: If command execution fails
            ConfigurationError: If command configuration fails
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

        # Parse tags
        parsed_tags = self._parse_raw_tags(raw_tags)
        if parsed_tags:
            logger.info(f"Using tags: {parsed_tags}")

        # Make sure dataset exists
        logger.info(f"Checking if dataset '{dataset_name}' exists")
        try:
            dataset_exists = arize_client.get_dataset(
                dataset_name=dataset_name,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to check if dataset '{dataset_name}' exists",
                details={"error": str(e)},
            )

        # If the dataset does not exist, raise an error
        if dataset_exists is None:
            raise ConfigurationError(
                f"Dataset '{dataset_name}' does not exist",
                details={"dataset": dataset_name},
            )

        # Make sure experiment DOES NOT exist
        logger.info(f"Checking if experiment '{experiment_name}' exists")
        try:
            experiment_exists = arize_client.get_experiment(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
            )
        except Exception as e:
            raise HandlerError(
                f"Failed to check if experiment '{experiment_name}' exists",
                details={"error": str(e)},
            )

        # If the experiment already exists, raise an error
        if experiment_exists is not None:
            raise ConfigurationError(
                f"Experiment '{experiment_name}' already exists",
                details={
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                },
            )

        # Create task instance
        logger.info(f"Creating task '{task_name}'")
        try:
            task = self._create_task(task_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create task '{task_name}'",
                details={"error": str(e)},
            )

        # Validate dataset schema against task requirements
        logger.info("Validating dataset schema")
        try:
            errors = self.schema_validator.validate(dataset_name, task, arize_client)
            if errors:
                raise ConfigurationError(
                    "Dataset schema incompatible with task",
                    details={
                        "dataset": dataset_name,
                        "task": task_name,
                        "errors": [
                            {
                                "path": e.path,
                                "message": e.message,
                                "expected": e.expected,
                                "actual": e.actual,
                            }
                            for e in errors
                        ],
                    },
                )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise e
            raise ConfigurationError(
                "Failed to validate dataset schema",
                details={"dataset": dataset_name, "task": task_name, "error": str(e)},
            )

        # Make sure evaluators are provided
        if not evaluator_names:
            raise HandlerError("No evaluators provided")

        # Create evaluator callables
        evaluators = self._create_evaluators(evaluator_names)

        # Run experiment
        logger.info(f"Running experiment: {experiment_name}")
        try:
            result = arize_client.run_experiment(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                task=task,
                evaluators=cast(Optional[List[Callable[..., Any]]], evaluators),
            )
        except Exception as e:
            raise HandlerError(
                f"Failed to run experiment '{experiment_name}'",
                details={
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                    "error": str(e),
                },
            )

        # Print the result of the experiment
        logger.debug(f"Experiment result: {result}")
        if hasattr(result, "success"):
            if result.success:
                click.secho(
                    f"\nSuccessfully ran experiment '{experiment_name}'", fg="green"
                )
            else:
                click.secho(
                    f"\nExperiment '{experiment_name}' failed: {result.error}", fg="red"
                )
        else:
            # Handle raw Arize API result
            click.secho(
                f"\nExperiment '{experiment_name}' completed. Result: {result}",
                fg="green",
            )

    def _get_arize_api_key(self) -> str:
        """Get the Arize API key.

        Returns:
            Arize API key
        """
        return self._get_required_env("ARIZE_API_KEY")

    def _get_arize_space_id(self) -> str:
        """Get the Arize space ID.

        Returns:
            Arize space ID
        """
        return self._get_required_env("ARIZE_SPACE_ID")

    def _get_arize_developer_key(self) -> str:
        """Get the Arize developer key.

        Returns:
            Arize developer key
        """
        return self._get_required_env("ARIZE_DEVELOPER_KEY")

    def _parse_raw_tags(
        self, raw_tags: Optional[List[str]]
    ) -> Optional[Dict[str, str]]:
        """Parse tag strings into a dictionary.

        Args:
            raw_tags: Optional list of key=value strings

        Returns:
            Dictionary of parsed tags or None

        Raises:
            ConfigurationError: If tag format is invalid
        """
        if not raw_tags:
            return None

        tags = {}
        for tag in raw_tags:
            try:
                key, value = tag.split("=", 1)
                tags[key.strip()] = value.strip()
            except ValueError:
                raise ConfigurationError(
                    f"Invalid tag format: {tag}. Use key=value format."
                )

        return tags

    def _create_task(
        self,
        task_name: str,
    ) -> Task:
        """Create task instance from name.

        Args:
            task_name: Name of the task

        Returns:
            Task instance

        Raises:
            HandlerError: If task creation fails
        """
        if not task_name:
            raise HandlerError("No task name provided")

        try:
            # Get the task class from the registry
            task_class = TaskRegistry.get(task_name)
            task = task_class()
        except Exception as e:
            raise HandlerError(
                f"Failed to create task '{task_name}'",
                details={"error": str(e)},
            )

        return task

    def _create_evaluators(self, names: Optional[List[str]]) -> Sequence[BaseEvaluator]:
        """Create evaluator instances from names.

        Args:
            names: List of evaluator names

        Returns:
            List of evaluator instances

        Raises:
            HandlerError: If evaluator creation fails
        """
        if not names:
            return []

        evaluators = []
        for name in names:
            try:
                # Get the evaluator class from the registry
                evaluator_class = EvaluatorRegistry.get(name)
                evaluator = evaluator_class()
                evaluators.append(evaluator)
            except Exception as e:
                raise HandlerError(
                    f"Failed to create evaluator '{name}'",
                    details={"error": str(e)},
                )

        return evaluators

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
