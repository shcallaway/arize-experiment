"""
CLI implementation for arize-experiment.
"""

import sys
from typing import Tuple
import click
import logging

from arize_experiment.arize import create_client, ClientError
from arize_experiment.config import get_arize_config, EnvironmentError
from arize_experiment.config import create_experiment_config
from arize_experiment.experiments.base import ExperimentError
from arize_experiment.experiments.dataset import DatasetExperiment

logger = logging.getLogger(__name__)


def experiment_options(f):
    """Common options for experiment commands."""
    f = click.option(
        "--name",
        "-n",
        required=True,
        help="Name of the experiment to create",
    )(f)

    f = click.option(
        "--dataset",
        "-d",
        required=True,
        help="Name of the dataset to use for the experiment",
    )(f)

    f = click.option(
        "--description",
        help="Optional description of the experiment",
    )(f)

    f = click.option(
        "--tag",
        "-t",
        multiple=True,
        help="Optional tags in key=value format (can be used multiple times)",
    )(f)

    return f


def parse_tags(tag_list):
    """Parse tag options into a dictionary.

    Args:
        tag_list: List of strings in key=value format

    Returns:
        Dict of parsed tags

    Raises:
        click.BadParameter: If tag format is invalid
    """
    if not tag_list:
        return None

    tags = {}
    for tag in tag_list:
        try:
            key, value = tag.split("=", 1)
            tags[key.strip()] = value.strip()
        except ValueError:
            raise click.BadParameter(
                f"Invalid tag format: {tag}. Use key=value format."
            )

    return tags


@click.group()
def cli():
    """arize-experiment: A tool for running experiments on Arize.

    This CLI provides commands for interacting with the Arize platform
    to create and run experiments. Use the --help flag with any command
    for more information.
    """
    # Configure logging early
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.debug("CLI initialized")


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
    logger.info("Starting experiment run")

    try:
        # Parse tags if provided
        tags = parse_tags(tag)
        if tags:
            logger.info(f"Using tags: {tags}")

        # Get Arize configuration
        logger.info("Loading Arize configuration")
        arize_config = get_arize_config()

        # Create Arize client
        logger.info("Initializing Arize client")
        client = create_client(arize_config)

        # Create experiment configuration
        logger.info("Creating experiment configuration")
        config = create_experiment_config(
            name=name,
            dataset=dataset,
            description=description,
            tags=tags,
        )

        # Create and run experiment
        logger.info(f"Running experiment '{name}'")
        experiment = DatasetExperiment(client, config)
        experiment_id = experiment.run()

        # Success output
        click.secho(f"\nSuccessfully started experiment '{name}'", fg="green")
        click.echo(f"Experiment ID: {experiment_id}")
        click.echo("View results in the Arize UI")

    except (EnvironmentError, ClientError) as e:
        click.secho(f"\nConfiguration error: {str(e)}", fg="red", err=True)
        sys.exit(1)
    except ExperimentError as e:
        click.secho(f"\nExperiment error: {str(e)}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", exc_info=True)
        click.secho(f"\nUnexpected error: {str(e)}", fg="red", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        click.secho(f"Critical error: {str(e)}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
