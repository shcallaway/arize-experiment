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
from arize_experiment.evaluators.is_positive import is_positive

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

        # Check if dataset exists
        dataset_name = dataset or arize_config.default_dataset
        if not dataset_name:
            raise click.UsageError(
                "No dataset specified. Either provide --dataset option "
                "or set DATASET in your .env file"
            )
        
        try:
            logger.debug(f"Checking if dataset {dataset_name} exists")
            client.get_dataset(space_id=arize_config.space_id, dataset_name=dataset_name)
            logger.debug(f"Dataset {dataset_name} exists")
        except Exception as e:
            logger.error(f"Dataset {dataset_name} does not exist", exc_info=True)
            raise click.UsageError(
                f"Dataset '{dataset_name}' does not exist in space '{arize_config.space_id}'. "
                "Please create the dataset first or use an existing dataset."
            )

        # Create experiment configuration
        logger.debug("Creating experiment configuration")

        # Map evaluator names to functions
        evaluator_map = {
            'is_positive': is_positive
        }

        # Convert evaluator names to functions
        evaluators = []
        if evaluator:
            for eval_name in evaluator:
                if eval_name in evaluator_map:
                    evaluators.append(evaluator_map[eval_name])
                else:
                    raise click.UsageError(f"Unknown evaluator: {eval_name}")
            logger.info(f"Using evaluators: {[e.__name__ for e in evaluators]}")

        logger.debug(f"Using dataset: {dataset_name}")

        config = create_experiment_config(
            name=name,
            dataset=dataset_name,
            description=description,
            tags=tags,
            evaluators=[e.__name__ for e in evaluators] if evaluators else None,
        )
        
        logger.debug(f"Created experiment config: {config}")

        # Check if experiment exists
        logger.debug(f"Checking if experiment '{name}' exists")
        try:
            client.get_experiment(
                space_id=arize_config.space_id,
                experiment_name=name,
                dataset_name=dataset_name
            )
            # If we get here, experiment exists
            logger.error(f"Experiment '{name}' already exists")
            raise click.UsageError(
                f"Experiment '{name}' already exists with dataset '{dataset_name}'. "
                "Please use a different experiment name."
            )
        except RuntimeError as e:
            # Get the original error message from the cause chain
            cause = e.__cause__
            if cause and isinstance(cause, Exception):
                error_msg = str(cause).lower()
            else:
                error_msg = str(e).lower()
                
            if "already exists" in error_msg:
                logger.error(f"Experiment '{name}' already exists", exc_info=True)
                raise click.UsageError(
                    f"Experiment '{name}' already exists with dataset '{dataset_name}'. "
                    "Please use a different experiment name."
                )
            elif "not found" in error_msg or "does not exist" in error_msg:
                # This is expected - experiment doesn't exist
                logger.info(f"Creating new experiment '{name}'")
                
                # Get values from config
                experiment_dict = config.to_dict()
                logger.debug(f"Running experiment with config: {experiment_dict}")
                
                logger.info("Attempting to run experiment")
                try:
                    logger.debug("Calling run_experiment with parameters:")
                    logger.debug(f"  space_id: {arize_config.space_id}")
                    logger.debug(f"  dataset_name: {dataset_name}")
                    logger.debug(f"  experiment_name: {experiment_dict['name']}")
                    logger.debug(f"  evaluators: {experiment_dict.get('evaluators')}")
                    
                    try:
                        client.run_experiment(
                            space_id=arize_config.space_id,
                            dataset_name=dataset_name,
                            task=example_task,
                            evaluators=evaluators,  # Pass the actual function list
                            experiment_name=experiment_dict['name']
                        )
                    except RuntimeError as e:
                        if "already exists" in str(e).lower():
                            # Race condition - experiment was created between our check and creation
                            logger.warning("Race condition detected - experiment already exists")
                            raise click.UsageError(
                                f"Experiment '{name}' already exists with dataset '{dataset_name}'. "
                                "Please use a different experiment name."
                            )
                        raise  # Re-raise other runtime errors
                    logger.info("Successfully called run_experiment")
                    click.secho(f"\nSuccessfully started experiment '{name}'", fg="green")
                except RuntimeError as e:
                    if "already exists" in str(e).lower():
                        # Race condition - experiment was created between our check and creation
                        logger.warning("Race condition detected - experiment already exists")
                        raise click.UsageError(
                            f"Experiment '{name}' already exists with dataset '{dataset_name}'. "
                            "Please use a different experiment name."
                        )
                    logger.error(f"Failed to run experiment: {str(e)}", exc_info=True)
                    raise click.UsageError(f"Failed to run experiment: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to run experiment: {str(e)}", exc_info=True)
                    raise click.UsageError(f"Failed to run experiment: {str(e)}")
            else:
                # Re-raise unexpected errors
                logger.error(f"Unexpected error checking experiment: {error_msg}")
                raise
        except Exception as e:
            # Log and re-raise any other unexpected errors
            logger.error(f"Unexpected error checking experiment existence: {e}", exc_info=True)
            raise

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
