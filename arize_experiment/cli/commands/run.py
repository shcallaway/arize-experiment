"""
Command handler for running experiments.

This module provides the handler for executing experiment runs in the arize-experiment
framework. It handles parameter validation, service coordination, error handling,
and user feedback.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import click

from arize_experiment.core.base_command import BaseCommand
from arize_experiment.core.errors import ConfigurationError, HandlerError
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task import Task
from arize_experiment.core.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class RunCommand(BaseCommand):
    """Handles experiment run command execution."""

    def execute(  # noqa: C901
        self,
        experiment_name: str,
        dataset_name: str,
        task_name: str,
        raw_tags: Optional[List[str]] = None,
        evaluator_names: Optional[List[str]] = None,
    ) -> None:
        """Handle the run experiment command.

        This method:
        1. Validates all input parameters
        2. Verifies dataset existence
        3. Creates and configures the task
        4. Sets up evaluators
        5. Runs the experiment
        6. Provides feedback on the result

        Args:
            experiment_name: Name of the experiment to run
            dataset_name: Name of the dataset to use
            task_name: Name of the task to execute
            raw_tags: Optional list of key=value tag strings
            evaluator_names: Optional list of evaluator names to use

        Raises:
            HandlerError: If command execution fails
            ConfigurationError: If command configuration is invalid
        """
        # Parse tags
        parsed_tags = self._parse_raw_tags(raw_tags)
        if parsed_tags:
            logger.info(f"Using tags: {parsed_tags}")

        self._verify_dataset_exists(dataset_name)

        self._verify_experiment_does_not_exist(experiment_name, dataset_name)

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
            errors = self._schema_validator.validate(
                dataset_name, task, self._arize_client
            )
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
            result = self._arize_client.run_experiment(
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
