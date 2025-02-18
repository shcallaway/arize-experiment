"""
CLI command definitions for arize-experiment.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import click

from arize_experiment.cli.handler import Handler

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """arize-experiment: A tool for running experiments on Arize.

    This CLI provides commands for interacting with the Arize platform
    to create and run experiments. Use the --help flag with any command
    for more information.
    """
    # Initialize logging
    log_level = os.getenv("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug("CLI initialized with log level: %s", log_level)


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
    required=True,
    help="Name of the dataset to use",
)
@click.option(
    "--task",
    "-t",
    required=True,
    help="Name of the task to use",
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
    task: Optional[str],
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

    Available tasks:
        sentiment_classification: Classifies the sentiment of a text
        execute_agent: Executes an agent by calling a web server

    Available evaluators:
        sentiment_classification_accuracy: Evaluates whether the sentiment
            classification is accurate
        execute_agent: Executes an agent by calling a web server
    """
    try:
        logger.info("Running experiment")

        # Initialize the command handler
        handler = Handler()

        # Run the experiment
        handler.run(
            experiment_name=name,
            dataset_name=dataset,
            task_name=task,
            raw_tags=list(tag) if tag else None,
            evaluator_names=list(evaluator) if evaluator else None,
        )
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
