"""
CLI command definitions for arize-experiment.
"""

import logging
import os
import sys
from typing import List

import click
from dotenv import load_dotenv

from arize_experiment.cli.commands.create_dataset import CreateDatasetCommand
from arize_experiment.cli.commands.run import RunCommand
from arize_experiment.core.errors import pretty_print_error
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


def register_all_evaluators() -> None:
    """Register all evaluators."""
    import arize_experiment.evaluators.chatbot_response_is_acceptable  # noqa
    import arize_experiment.evaluators.sentiment_classification_is_accurate  # noqa


def register_all_tasks() -> None:
    """Register all tasks."""
    import arize_experiment.tasks.call_chatbot_server  # noqa
    import arize_experiment.tasks.classify_sentiment  # noqa
    import arize_experiment.tasks.delegate  # noqa
    import arize_experiment.tasks.echo  # noqa


def register() -> None:
    """Register all evaluators and tasks."""
    register_all_evaluators()
    register_all_tasks()


register()


def validate_task(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate task name."""
    if value not in TaskRegistry.list():
        raise click.BadParameter(
            f"Task '{value}' not found. Available tasks: "
            f"{', '.join(TaskRegistry.list())}"
        )
    return value


def validate_evaluator(
    ctx: click.Context, param: click.Parameter, value: List[str]
) -> List[str]:
    """Validate evaluator names."""
    available_evaluators = EvaluatorRegistry.list()
    for evaluator in value:
        if evaluator not in available_evaluators:
            raise click.BadParameter(
                f"Evaluator '{evaluator}' not found. Available evaluators: "
                f"{', '.join(available_evaluators)}"
            )
    return value


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


@cli.command(
    help="""Run an experiment on Arize.

This command creates and runs a new experiment on the Arize platform
using the specified dataset, task, and evaluator(s). The experiment
results can be viewed in the Arize web dashboard."""
)
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
    callback=validate_task,
    help="Name of the task to use",
)
@click.option(
    "--evaluator",
    "-e",
    multiple=True,
    required=True,
    callback=validate_evaluator,
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
    tag: List[str],
    evaluator: List[str],
) -> None:
    """Run an experiment on Arize.

    Args:
        name: Name of the experiment to create
        dataset: Name of the dataset to use
        task: Name of the task to use
        tag: Optional tags in key=value format
        evaluator: Name of an evaluator to use
    """
    try:
        logger.info("Running experiment")

        # Initialize the command handler
        command = RunCommand()

        # Run the experiment
        command.execute(
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


@cli.command(
    help="""Create a new dataset from a CSV file.

This command creates a new dataset on the Arize platform from a local CSV file."""
)
@click.option(
    "--name",
    "-n",
    required=True,
    help="Name of the dataset to create",
)
@click.option(
    "--path-to-csv",
    required=True,
    help="Path to the CSV file to upload",
)
def create_dataset(name: str, path_to_csv: str) -> None:
    """Create a new dataset from a CSV file.

    Args:
        name: Name of the dataset to create
        path_to_csv: Path to the CSV file to upload
    """
    try:
        logger.info("Creating dataset")

        # Initialize the command handler
        command = CreateDatasetCommand()

        # Create the dataset
        command.execute(
            dataset_name=name,
            path_to_csv=path_to_csv,
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
