"""
Experiment configuration management for arize-experiment.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an Arize experiment."""

    name: str
    dataset: str
    description: Optional[str] = None
    tags: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for Arize client.

        Returns:
            Dict representation of experiment configuration
        """
        config = {
            "name": self.name,
            "dataset": self.dataset,
        }

        if self.description:
            config["description"] = self.description

        if self.tags:
            config["tags"] = self.tags

        logger.debug(f"Created experiment config: {config}")
        return config


def create_experiment_config(
    name: str,
    dataset: str,
    description: Optional[str] = None,
    tags: Optional[dict] = None,
) -> ExperimentConfig:
    """Create a new experiment configuration.

    Args:
        name: Name of the experiment
        dataset: Name of the dataset to use
        description: Optional experiment description
        tags: Optional key-value pairs for experiment metadata

    Returns:
        ExperimentConfig instance with provided parameters
    """
    logger.debug(
        f"Creating experiment config: name={name}, dataset={dataset}, "
        f"description={description}, tags={tags}"
    )

    config = ExperimentConfig(
        name=name,
        dataset=dataset,
        description=description,
        tags=tags,
    )

    logger.debug("Successfully created experiment configuration")
    return config
