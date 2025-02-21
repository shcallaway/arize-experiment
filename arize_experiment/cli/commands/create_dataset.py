"""
Command handler for creating datasets.

This module provides the handler for creating datasets in the arize-experiment
framework. It handles parameter validation, service coordination, error handling,
and user feedback.
"""

import logging

import click
import pandas as pd

from arize_experiment.core.base_command import BaseCommand
from arize_experiment.core.errors import HandlerError

logger = logging.getLogger(__name__)


class CreateDatasetCommand(BaseCommand):
    """Handles dataset creation command execution."""

    def execute(
        self,
        dataset_name: str,
        path_to_csv: str,
    ) -> None:
        """Create a new dataset from a CSV file.

        Args:
            dataset_name: Name of the dataset to create
            path_to_csv: Path to the CSV file to upload

        Raises:
            HandlerError: If command execution fails
            ConfigurationError: If command configuration fails
        """
        self._verify_dataset_does_not_exist(dataset_name)

        logger.info(f"Creating dataset '{dataset_name}' from {path_to_csv}")

        try:
            # Read the CSV file using pandas with appropriate settings
            df = pd.read_csv(path_to_csv, quoting=1, escapechar="\\")

            # Create the dataset using the Arize client
            dataset_id = self._arize_client.create_dataset(
                dataset_name=dataset_name, data=df
            )

            if not dataset_id:
                raise HandlerError(
                    f"Failed to create dataset '{dataset_name}'",
                    details={"error": "No dataset ID returned"},
                )

            logger.debug(f"Created dataset with ID: {dataset_id}")
            click.secho(f"\nSuccessfully created dataset '{dataset_name}'", fg="green")

        except pd.errors.EmptyDataError:
            raise HandlerError(
                f"Failed to create dataset '{dataset_name}'",
                details={"error": "CSV file is empty"},
            )
        except pd.errors.ParserError as e:
            raise HandlerError(
                f"Failed to create dataset '{dataset_name}'",
                details={"error": f"Failed to parse CSV file: {str(e)}"},
            )
        except Exception as e:
            raise HandlerError(
                f"Failed to create dataset '{dataset_name}'", details={"error": str(e)}
            )
