"""
CLI command implementations for arize-experiment.
"""

import sys
from typing import Tuple

import click

from arize_experiment.client.arize import create_client, ClientError
from arize_experiment.config.env import get_arize_config, EnvironmentError
from arize_experiment.config.experiment import create_experiment_config
from arize_experiment.experiments.base import ExperimentError
from arize_experiment.experiments.dataset import DatasetExperiment
from arize_experiment.cli.options import experiment_options, parse_tags
from arize_experiment.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    """arize-experiment: A tool for running experiments on Arize.

    This CLI provides commands for interacting with the Arize platform
    to create and run experiments. Use the --help flag with any command
    for more information.
    """
    # Configure logging early
    configure_logging()
    print("CLI initialized")  # Basic print for testing


@cli.command()
def test():
    """Test command to verify CLI functionality."""
    print("Test command executed")
    click.echo("Click echo test")
    click.secho("Click secho test", fg="green")
    logger.debug("Debug log test")
    logger.info("Info log test")
    logger.warning("Warning log test")
    logger.error("Error log test")


@cli.command()
@experiment_options
def run(name: str, dataset: str, description: str, tag: Tuple[str, ...]):
    """Run an experiment on Arize.

    This command creates and runs a new experiment on the Arize platform
    using the specified dataset.

    Example:
        $ arize-experiment run --name my-experiment --dataset my-dataset

    Tags can be added using the --tag option multiple times:
        $ arize-experiment run -n exp-1 -d data-1 -t type=test -t env=prod
    """
    print("Starting experiment run...")  # Basic print for testing

    try:
        # Parse tags if provided
        tags = parse_tags(tag)
        if tags:
            print(f"Using tags: {tags}")  # Basic print for testing

        # Get Arize configuration
        print("Loading Arize configuration...")  # Basic print for testing
        arize_config = get_arize_config()

        # Create Arize client
        print("Initializing Arize client...")  # Basic print for testing
        client = create_client(arize_config)

        # Create experiment configuration
        print("Creating experiment configuration...")  # Basic print for testing
        config = create_experiment_config(
            name=name,
            dataset=dataset,
            description=description,
            tags=tags,
        )

        # Create and run experiment
        print(f"Running experiment '{name}'...")  # Basic print for testing
        experiment = DatasetExperiment(client, config)
        experiment_id = experiment.run()

        # Success output
        print(f"\nSuccessfully started experiment '{name}'")  # Basic print for testing
        print(f"Experiment ID: {experiment_id}")  # Basic print for testing
        print("View results in the Arize UI")  # Basic print for testing

    except (EnvironmentError, ClientError) as e:
        print(
            f"\nConfiguration error: {str(e)}", file=sys.stderr
        )  # Basic print for testing
        sys.exit(1)
    except ExperimentError as e:
        print(
            f"\nExperiment error: {str(e)}", file=sys.stderr
        )  # Basic print for testing
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", exc_info=True)
        print(
            f"\nUnexpected error: {str(e)}", file=sys.stderr
        )  # Basic print for testing
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)  # Basic print for testing
        sys.exit(1)


if __name__ == "__main__":
    main()
