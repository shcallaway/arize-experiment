"""
Command handlers for the CLI interface.
"""

import logging
from typing import Dict, List, Optional

import click

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.tasks.echo import EchoTask
from arize_experiment.tasks.sentiment_classification import SentimentClassificationTask
from arize_experiment.core.experiment import Experiment
from arize_experiment.infrastructure.arize_client import ArizeClient, ArizeClientError
from arize_experiment.infrastructure.config import (
    ConfigManager,
    ConfigurationError,
    ExperimentConfig,
)
from arize_experiment.services.evaluator_service import EvaluatorService
from arize_experiment.services.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class HandlerError(Exception):
    """Raised when command handling fails."""
    pass


class CommandHandler:
    """Handles CLI command execution.
    
    This class coordinates between the CLI interface and the application's
    core services. It handles:
    1. Parameter validation and processing
    2. Service coordination
    3. Error handling and user feedback
    """

    def __init__(self):
        """Initialize the command handler."""
        self.config_manager = ConfigManager()
        self.evaluator_service = EvaluatorService()
        self._init_evaluators()

    def _init_evaluators(self):
        """Initialize and register available evaluators."""
        # Import evaluators here to avoid circular imports
        from arize_experiment.evaluators.sentiment_classification_accuracy import SentimentClassificationAccuracyEvaluator

        # Register available evaluator types
        self.evaluator_service.register_type(SentimentClassificationAccuracyEvaluator)

    def run_experiment(
        self,
        name: str,
        dataset: Optional[str],
        description: Optional[str],
        tags: Optional[List[str]],
        evaluator_names: Optional[List[str]],
    ) -> None:
        """Handle the run experiment command.
        
        Args:
            name: Name of the experiment
            dataset: Optional dataset name
            description: Optional experiment description
            tags: Optional list of key=value tag strings
            evaluator_names: Optional list of evaluator names
        
        Raises:
            HandlerError: If command execution fails
        """
        try:
            # Get Arize configuration
            logger.debug("Loading Arize configuration")
            arize_config = self.config_manager.get_arize_config()

            # Initialize client
            logger.debug("Initializing Arize client")
            client = ArizeClient(arize_config)

            # Parse tags
            parsed_tags = self._parse_tags(tags)
            if parsed_tags:
                logger.info(f"Using tags: {parsed_tags}")

            # Get dataset name
            dataset_name = dataset or arize_config.default_dataset
            if not dataset_name:
                raise HandlerError(
                    "No dataset specified. Either provide --dataset option "
                    "or set DATASET in your .env file"
                )

            # Check if experiment already exists
            existing = client.get_experiment(experiment_name=name, dataset_name=dataset_name)
            if existing is not None:
                raise HandlerError(f"Experiment '{name}' already exists")

            # Create experiment configuration
            logger.debug(f"Creating experiment configuration: {name}")
            config = self.config_manager.create_experiment_config(
                name=name,
                dataset=dataset_name,
                description=description,
                tags=parsed_tags,
                evaluators=evaluator_names,
            )

            # Create evaluator instances
            evaluators = self._create_evaluators(evaluator_names)

            # Use sentiment classification as default task
            task = SentimentClassificationTask()

            # Create experiment
            experiment = Experiment(
                name=config.name,
                dataset=config.dataset,
                task=task,
                evaluators=evaluators,
                description=config.description,
                tags=config.tags or {},
            )

            # Initialize runner
            runner = ExperimentRunner(client)

            # Run experiment
            logger.info(f"Running experiment: {experiment}")
            result = runner.run(experiment)

            # Log the result for debugging
            logger.debug(f"Experiment result: {result}")
            
            if hasattr(result, 'success'):
                if result.success:
                    click.secho(
                        f"\nSuccessfully ran experiment '{name}'",
                        fg="green"
                    )
                else:
                    click.secho(
                        f"\nExperiment '{name}' failed: {result.error}",
                        fg="red"
                    )
            else:
                # Handle raw Arize API result
                click.secho(
                    f"\nExperiment '{name}' completed. Result: {result}",
                    fg="green"
                )

        except (ConfigurationError, ArizeClientError) as e:
            raise HandlerError(f"Configuration error: {str(e)}")
        except Exception as e:
            raise HandlerError(f"Unexpected error: {str(e)}")

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
                raise HandlerError(
                    f"Invalid tag format: {tag}. Use key=value format."
                )

        return tags

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
        for name in evaluator_names:
            evaluator = self.evaluator_service.get_evaluator(name)
            if evaluator is None:
                raise HandlerError(f"Unknown evaluator: {name}")
            evaluators.append(evaluator)

        return evaluators
