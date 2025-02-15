"""
CLI implementation for arize-experiment.
"""

import sys
import click
from typing import Optional, Tuple
from arize_experiment.client import ClientError, create_client
from arize_experiment.task import example_task
from arize_experiment.config import get_arize_config, EnvironmentError
from arize_experiment.config import create_experiment_config
from arize_experiment.logging import get_logger, configure_logging

logger = get_logger(__name__)


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


@cli.command()
@click.option(
    "--name",
    "-n",
    required=True,
    help="Name of the experiment to create",
)
@click.option(
    "--dataset",
    "-d",
    required=False,
    help="Name of the dataset to use for the experiment (uses DATASET from .env if not provided)",
)
@click.option(
    "--description",
    help="Optional description of the experiment",
)
@click.option(
    "--tag",
    "-t",
    multiple=True,
    help="Optional tags in key=value format (can be used multiple times)",
)
@click.option(
    "--evaluator",
    "-e",
    multiple=True,
    help="Name of an evaluator to use (can be used multiple times)",
)
def run(
    name: str,
    dataset: Optional[str],
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

        logger.debug("Initializing Arize client")
        client = create_client(arize_config)
        logger.debug("Client initialized successfully")

        # Create experiment configuration
        logger.debug("Creating experiment configuration")
        # Convert evaluator tuple to list if provided
        evaluators = list(evaluator) if evaluator else None
        if evaluators:
            logger.info(f"Using evaluators: {evaluators}")

        # Use default dataset if none provided
        dataset_id = dataset or arize_config.default_dataset
        if not dataset_id:
            raise click.UsageError(
                "No dataset specified. Either provide --dataset option "
                "or set DATASET in your .env file"
            )
        logger.debug(f"Using dataset: {dataset_id}")

        config = create_experiment_config(
            name=name,
            dataset=dataset_id,
            description=description,
            tags=tags,
            evaluators=evaluators,
        )
        logger.debug(f"Created experiment config: {config}")

        # Create and run experiment
        logger.info(f"Starting experiment '{name}'")
        # Get values from config
        experiment_dict = config.to_dict()
        logger.debug(f"Running experiment with config: {experiment_dict}")
        
        # Run experiment with named arguments matching API signature
        dataset_id = experiment_dict['dataset']
        logger.debug(f"Using dataset ID: {dataset_id}")
        
        client.run_experiment(
            space_id=arize_config.space_id,
            dataset_id=dataset_id,
            task=example_task,
            evaluators=experiment_dict.get('evaluators'),
            experiment_name=experiment_dict['name']
        )
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
