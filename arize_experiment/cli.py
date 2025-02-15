"""
Command-line interface for arize-experiment.
"""

import os
import sys
import logging
from typing import Optional
import click
from arize.pandas.logger import Client
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@click.group()
def cli():
    """arize-experiment: A tool for running experiments on Arize.

    This CLI provides commands for interacting with the Arize platform
    to create and run experiments. Use the --help flag with any command
    for more information.
    """
    pass


def init_client() -> Client:
    """Initialize Arize client with API and space ID."""
    api_key = os.getenv("ARIZE_API_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")
    
    if not api_key:
        click.secho("Error: ARIZE_API_KEY environment variable is not set", fg="red")
        click.echo("Please set your Arize API key in the .env file:")
        click.echo("ARIZE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    if not space_id:
        click.secho("Error: ARIZE_SPACE_ID environment variable is not set", fg="red")
        click.echo("Please set your Arize space ID in the .env file:")
        click.echo("ARIZE_SPACE_ID=your_space_id_here")
        sys.exit(1)

    logger.debug(f"Initializing client with API key: {api_key[:5]}...")
    return Client(api_key=api_key, space_id=space_id)


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
    help="Name of the dataset to use for the experiment",
)
def run(name: str, dataset: str):
    """Run an experiment on Arize.
    
    This command creates and runs a new experiment on the Arize platform
    using the specified dataset.
    
    Example:
        $ arize-experiment run --name my-experiment --dataset my-dataset
    """
    try:
        # Initialize client
        client = init_client()
        
        # Create experiment configuration
        config = {
            "name": name,
            "dataset": dataset
        }
        
        # Start experiment
        try:
            logger.debug(f"Creating experiment with config: {config}")
            experiment_id = client.create_experiment(**config)
            click.secho(f"Successfully started experiment '{name}'", fg="green")
            click.echo(f"Experiment ID: {experiment_id}")
            click.echo("View results in the Arize UI")
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}", exc_info=True)
            click.secho(f"Error creating experiment: {str(e)}", fg="red")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Unexpected error", exc_info=True)
        click.secho(f"Unexpected error: {str(e)}", fg="red")
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
