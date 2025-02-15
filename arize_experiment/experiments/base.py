"""
Base experiment class for arize-experiment.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from arize.pandas.logger import Client
from arize_experiment.config.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentError(Exception):
    """Base class for experiment-related errors."""

    pass


class Experiment(ABC):
    """Base class for all experiment types."""

    def __init__(self, client: Client, config: ExperimentConfig):
        """Initialize experiment with Arize client and configuration.

        Args:
            client: Configured Arize client instance
            config: Experiment configuration
        """
        self.client = client
        self.config = config
        self.experiment_id: Optional[str] = None

    @abstractmethod
    def run(self) -> str:
        """Run the experiment.

        This method must be implemented by concrete experiment classes
        to define the specific experiment logic.

        Returns:
            str: ID of the created experiment

        Raises:
            ExperimentError: If experiment creation fails
        """
        pass

    def get_status(self) -> Optional[str]:
        """Get the current status of the experiment.

        Returns:
            Optional[str]: Current experiment status if available
        """
        if not self.experiment_id:
            return None

        try:
            return self.client.get_experiment_status(self.experiment_id)
        except Exception as e:
            logger.error(f"Failed to get experiment status: {e}")
            return None
