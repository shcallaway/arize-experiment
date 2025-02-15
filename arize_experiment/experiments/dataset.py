"""
Dataset-based experiment implementation.
"""

import logging
from arize.pandas.logger import Client

from arize_experiment.config import ExperimentConfig
from arize_experiment.experiments.base import Experiment, ExperimentError

logger = logging.getLogger(__name__)


class DatasetExperiment(Experiment):
    """Experiment implementation for dataset-based experiments."""

    def __init__(self, client: Client, config: ExperimentConfig):
        """Initialize dataset experiment.

        Args:
            client: Configured Arize client instance
            config: Experiment configuration
        """
        logger.debug("Initializing dataset experiment")
        super().__init__(client, config)
        logger.debug("Dataset experiment initialized")

    def run(self) -> str:
        """Run the dataset experiment.

        Creates a new experiment on Arize using the configured dataset.

        Returns:
            str: ID of the created experiment

        Raises:
            ExperimentError: If experiment creation fails
        """
        try:
            logger.info(
                f"Creating experiment '{self.config.name}' with dataset '{self.config.dataset}'"
            )
            logger.debug(f"Full experiment config: {self.config.to_dict()}")

            # Convert config to dict for client
            config_dict = self.config.to_dict()

            # Create experiment using Arize client
            logger.debug("Calling create_experiment on Arize client")
            try:
                self.experiment_id = self.client.create_experiment(**config_dict)
            except Exception as e:
                logger.error(f"Arize client error: {str(e)}")
                raise ExperimentError(f"Arize client error: {str(e)}") from e

            if not self.experiment_id:
                logger.error("No experiment ID returned from Arize")
                raise ExperimentError("No experiment ID returned from Arize")

            logger.info(
                f"Successfully created experiment with ID: {self.experiment_id}"
            )
            return self.experiment_id

        except Exception as e:
            error_msg = f"Failed to create experiment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ExperimentError(error_msg) from e
