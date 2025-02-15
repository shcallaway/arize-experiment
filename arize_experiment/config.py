"""
Configuration management for arize-experiment.

This module combines environment variable and experiment configuration management.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

from arize_experiment.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArizeConfig:
    """Configuration for Arize API credentials."""

    api_key: str
    space_id: str
    developer_key: str


@dataclass
class ExperimentConfig:
    """Configuration for an Arize experiment."""

    name: str
    dataset: str
    description: Optional[str] = None
    tags: Optional[dict] = None
    evaluators: Optional[list[str]] = None

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
            
        if self.evaluators:
            config["evaluators"] = self.evaluators

        logger.debug(f"Created experiment config: {config}")
        return config


class EnvironmentError(Exception):
    """Raised when required environment variables are missing."""

    pass


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    logger.debug("Environment variables loaded from .env file")


def get_arize_config() -> ArizeConfig:
    """Get Arize API configuration from environment.

    Returns:
        ArizeConfig with API credentials

    Raises:
        EnvironmentError: If required variables are not set
    """
    # Load environment variables first
    load_environment()

    api_key = os.getenv("ARIZE_API_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")
    developer_key = os.getenv("ARIZE_DEVELOPER_KEY")

    logger.debug(f"Retrieved API key (length: {len(api_key) if api_key else 0})")
    logger.debug(f"Retrieved space ID: {space_id}")
    logger.debug(f"Retrieved developer key: {developer_key}")
    if not api_key:
        msg = (
            "ARIZE_API_KEY environment variable is not set.\n"
            "Please set your Arize API key in the .env file:\n"
            "ARIZE_API_KEY=your_api_key_here"
        )
        logger.error(msg)
        raise EnvironmentError(msg)

    if not space_id:
        msg = (
            "ARIZE_SPACE_ID environment variable is not set.\n"
            "Please set your Arize space ID in the .env file:\n"
            "ARIZE_SPACE_ID=your_space_id_here"
        )
        logger.error(msg)
        raise EnvironmentError(msg)

    if not developer_key:
        msg = (
            "ARIZE_DEVELOPER_KEY environment variable is not set.\n"
            "Please set your Arize developer key in the .env file:\n"
            "ARIZE_DEVELOPER_KEY=your_developer_key_here"
        )
        logger.error(msg)
        raise EnvironmentError(msg)

    logger.debug("Successfully loaded Arize configuration")
    return ArizeConfig(
        api_key=api_key,
        space_id=space_id,
        developer_key=developer_key
    )


def create_experiment_config(
    name: str,
    dataset: str,
    description: Optional[str] = None,
    tags: Optional[dict] = None,
    evaluators: Optional[list[str]] = None,
) -> ExperimentConfig:
    """Create a new experiment configuration.

    Args:
        name: Name of the experiment
        dataset: Name of the dataset to use
        description: Optional experiment description
        tags: Optional key-value pairs for experiment metadata
        evaluators: Optional list of evaluator names to use

    Returns:
        ExperimentConfig instance with provided parameters
    """
    logger.debug(
        f"Creating experiment config: name={name}, dataset={dataset}, "
        f"description={description}, tags={tags}, evaluators={evaluators}"
    )
    return ExperimentConfig(
        name=name,
        dataset=dataset,
        description=description,
        tags=tags,
        evaluators=evaluators
    )
