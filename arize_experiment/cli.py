"""
CLI implementation for arize-experiment.
"""

import sys
from typing import Tuple
import click

from arize_experiment.arize import create_client, ClientError
from arize_experiment.config import get_arize_config, EnvironmentError
from arize_experiment.config import create_experiment_config
from arize_experiment.logging import configure_logging, get_logger

logger = get_logger(__name__)


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

    f = click.option(
        "--evaluator",
        "-e",
        multiple=True,
        help="Name of an evaluator to use (can be used multiple times)",
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
    # Initialize logging with DEBUG level for development
    configure_logging("DEBUG")
    logger.debug("CLI initialized")


@cli.command()
@experiment_options
def run(
    name: str,
    dataset: str,
    description: str,
    tag: Tuple[str, ...],
    evaluator: Tuple[str, ...],
):
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
        logger.debug("Starting run command")

        # Parse tags if provided
        tags = parse_tags(tag)
        if tags:
            logger.info(f"Using tags: {tags}")

        # Get Arize configuration
        logger.debug("Loading Arize configuration")
        arize_config = get_arize_config()
        logger.debug(f"Loaded config: {arize_config}")

        # TODO: Uncomment when client implementation is ready
        # logger.debug("Initializing Arize client")
        # client = create_client(arize_config)
        # logger.debug("Client initialized successfully")

        # Create experiment configuration
        logger.debug("Creating experiment configuration")
        # Convert evaluator tuple to list if provided
        evaluators = list(evaluator) if evaluator else None
        if evaluators:
            logger.info(f"Using evaluators: {evaluators}")

        config = create_experiment_config(
            name=name,
            dataset=dataset,
            description=description,
            tags=tags,
            evaluators=evaluators,
        )
        logger.debug(f"Created experiment config: {config}")

        # Create and run experiment
        logger.info(f"Starting experiment '{name}'")
        # TODO: Use client to run experiment once implemented
        click.secho(f"\nSuccessfully started experiment '{name}'", fg="green")

    except (EnvironmentError, ClientError) as e:
        click.secho(f"\nConfiguration error: {str(e)}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", exc_info=True)
        click.secho(f"\nUnexpected error: {str(e)}", fg="red", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        # Set up exception hook to print full traceback
        def handle_exception(exc_type, exc_value, exc_traceback):
            import traceback

            print(
                "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            )

        sys.excepthook = handle_exception

        cli()
    except Exception as e:
        click.secho(f"Critical error: {str(e)}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
