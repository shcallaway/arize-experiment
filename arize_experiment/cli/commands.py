"""
CLI command definitions for arize-experiment.
"""

import logging
import os
import sys
from typing import Tuple

import click
from dotenv import load_dotenv

from arize_experiment.cli.handler import Handler
from arize_experiment.core.errors import pretty_print_error
from arize_experiment.core.evaluator_registry import EvaluatorRegistry

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """arize-experiment: A tool for running experiments on Arize.

    This CLI provides commands for interacting with the Arize platform
    to create and run experiments. Use the --help flag with any command
    for more information.
    """
    # Load environment variables
    try:
        load_dotenv()
    except Exception as e:
        error_msg = pretty_print_error(e)
        click.secho(f"\nError: {error_msg}", fg="red", err=True)
        sys.exit(1)

    # Initialize logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
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
    type=click.Choice(
        [
            "sentiment_classification",
            "execute_agent",
        ]
    ),
    help="Name of the task to use",
)
@click.option(
    "--evaluator",
    "-e",
    multiple=True,
    required=True,
    type=click.Choice(EvaluatorRegistry.list()),
    help="Name of an evaluator to use (can be used multiple times)",
)
@click.option(
    "--tag",
    multiple=True,
    help="Optional tags in key=value format (can be used multiple times)",
)
def run(
    name: str,
    dataset: str,
    task: str,
    tag: Tuple[str, ...],
    evaluator: Tuple[str, ...],
) -> None:
    """Run an experiment on Arize.

    This command creates and runs a new experiment on the Arize platform
    using the specified dataset.

    Example:
        $ arize-experiment run \
            --name <experiment-name> \
            --dataset <dataset-name> \
            --task <task-name> \
            --evaluator <evaluator-name> \
            --tag <tag-key>=<tag-value> \
            --tag <tag-key>=<tag-value>

    Available tasks:
        sentiment_classification: Classifies the sentiment of a text
            Requires OPENAI_API_KEY environment variable
        execute_agent: Executes an agent by calling a web server
            Optional AGENT_SERVER_URL environment variable\
            (default: http://localhost:8080)

    Available evaluators:
        Run with --help to see the current list of registered evaluators
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
        error_msg = pretty_print_error(e)
        click.secho(f"\nError: {error_msg}", fg="red", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        click.secho(f"Critical error: {str(e)}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
