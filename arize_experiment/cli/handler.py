"""
Command handlers for the CLI interface.
"""

import logging
import os
from typing import Dict, List, Optional
import click
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.tasks.sentiment_classification import SentimentClassificationTask
from arize_experiment.evaluators.sentiment_classification_accuracy import (
    SentimentClassificationAccuracyEvaluator,
)
from arize_experiment.core.arize import Arize
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class HandlerError(Exception):
    """Raised when command handling fails."""

    pass


class Handler:
    """Handles CLI command execution.

    This class coordinates between the CLI interface and the application's
    core services. It handles:
    1. Parameter validation and processing
    2. Service coordination
    3. Error handling and user feedback
    """

    def __init__(self):
        """Initialize the command handler."""
        self._load_env()

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        try:
            load_dotenv()
            logger.debug("Environment variables loaded from .env file")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {str(e)}")

    def run(
        self,
        name: str,
        dataset: Optional[str] = None,
        tags: Optional[List[str]] = None,
        evaluators: Optional[List[str]] = None,
    ) -> None:
        """Handle the run experiment command.

        Args:
            name: Name of the experiment
            dataset: Optional dataset name
            tags: Optional list of key=value tag strings
            evaluators: Optional list of evaluator names

        Raises:
            HandlerError: If command execution fails
        """
        try:
            # Get Arize values from env
            logger.debug("Loading Arize values from env")
            try:
                arize_api_key = self._get_arize_api_key()
                arize_space_id = self._get_arize_space_id()
                arize_developer_key = self._get_arize_developer_key()
            except Exception as e:
                raise HandlerError(f"Failed to load Arize values from env: {str(e)}")

            # Initialize Arize client
            logger.debug("Initializing Arize client")
            try:
                arize = Arize(
                    api_key=arize_api_key,
                    developer_key=arize_developer_key,
                    space_id=arize_space_id,
                )
            except Exception as e:
                raise HandlerError(f"Failed to initialize Arize client: {str(e)}")

            # Get dataset name
            dataset = self._get_dataset()
            logger.info(f"Using dataset: {dataset}")

            # Parse tags
            parsed_tags = self._parse_tags(tags)
            if parsed_tags:
                logger.info(f"Using tags: {parsed_tags}")

            # Check if experiment already exists
            logger.info(f"Checking if experiment '{name}' already exists")
            try:
                existing = arize.get_experiment(
                    experiment=name,
                    dataset=dataset,
                )
            except Exception as e:
                raise HandlerError(
                    f"Failed to check if experiment '{name}' already exists: {str(e)}"
                )

            # If the experiment already exists, raise an error
            if existing is not None:
                raise HandlerError(f"Experiment '{name}' already exists")

            # Create evaluator instances
            evaluators = self._create_evaluators(evaluators)

            # Set the task to "sentiment_classification"
            # TODO(Sherwood): Make this configurable
            task = SentimentClassificationTask()

            # Run experiment
            logger.info(f"Running experiment: {name}")
            result = arize.run_experiment(
                experiment=name,
                dataset=dataset,
                task=task,
                evaluators=evaluators if evaluators else None,
            )

            # Log the result for debugging
            logger.debug(f"Experiment result: {result}")

            # Print the result of the experiment
            if hasattr(result, "success"):
                if result.success:
                    click.secho(f"\nSuccessfully ran experiment '{name}'", fg="green")
                else:
                    click.secho(
                        f"\nExperiment '{name}' failed: {result.error}", fg="red"
                    )
            else:
                # Handle raw Arize API result
                click.secho(
                    f"\nExperiment '{name}' completed. Result: {result}", fg="green"
                )

        except Exception as e:
            raise HandlerError(f"Unexpected error: {str(e)}")

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

    def _get_dataset(self) -> str:
        """Get the dataset name.

        Returns:
            Dataset name
        """
        dataset = os.getenv("ARIZE_DATASET")

        # If the dataset is not set, generate a random dataset name
        if dataset is None:
            return f"dataset_{os.urandom(4).hex()}"

        return dataset

    def _parse_tags(self, tag_list: Optional[List[str]]) -> Optional[Dict[str, str]]:
        """Parse tag strings into a dictionary.

        Args:
            tag_list: Optional list of key=value strings

        Returns:
            Dictionary of parsed tags or None

        Raises:
            HandlerError: If tag format is invalid
        """
        if not tag_list:
            return None

        tags = {}
        for tag in tag_list:
            try:
                key, value = tag.split("=", 1)
                tags[key.strip()] = value.strip()
            except ValueError:
                raise HandlerError(f"Invalid tag format: {tag}. Use key=value format.")

        return tags

    def _create_sentiment_classification_accuracy_evaluator(
        self,
    ) -> SentimentClassificationAccuracyEvaluator:
        """Create a sentiment classification accuracy evaluator.

        Returns:
            SentimentClassificationAccuracyEvaluator instance
        """
        try:
            api_key = self._get_required_env("OPENAI_API_KEY")
            return SentimentClassificationAccuracyEvaluator(
                api_key=api_key,
            )
        except Exception as e:
            raise HandlerError(
                f"Failed to create sentiment classification accuracy evaluator: {str(e)}"
            )

    def _create_evaluators(
        self, evaluator_names: Optional[List[str]]
    ) -> List[BaseEvaluator]:
        """Create evaluator instances from names.

        Args:
            evaluator_names: Optional list of evaluator names

        Returns:
            List of evaluator instances

        Raises:
            HandlerError: If an evaluator cannot be created
        """
        if not evaluator_names:
            return []

        evaluators = []

        # For each evaluator name, create an evaluator instance
        for name in evaluator_names:
            if name == "sentiment_classification_accuracy":
                evaluator = self._create_sentiment_classification_accuracy_evaluator()
            else:
                raise HandlerError(f"Unknown evaluator: {name}")

            evaluators.append(evaluator)

        return evaluators

    def _get_required_env(self, name: str) -> str:
        """Get a required environment variable.

        Args:
            name: Name of the environment variable

        Returns:
            Value of the environment variable

        Raises:
            HandlerError: If the variable is not set
        """
        value = os.getenv(name)

        # If the variable is not set, raise an error
        if not value:
            error_msg = (
                f"{name} environment variable is not set.\n"
                f"Please set it in your .env file:\n"
                f"{name}=your_value_here"
            )
            logger.error(error_msg)
            raise HandlerError(error_msg)

        return value
