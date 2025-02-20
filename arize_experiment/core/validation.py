"""
Schema validation utilities for Arize datasets and tasks.

This module provides utilities for validating dataset schemas against task
requirements. It ensures that datasets contain the required columns with
appropriate data types before task execution.

The validation framework:
1. Validates column presence
2. Checks data types
3. Verifies required fields
4. Handles nested schemas
5. Provides detailed error reporting

Example:
    ```python
    from arize_experiment.core.validation import SchemaValidator
    from arize_experiment.core.task import Task

    validator = SchemaValidator()
    errors = validator.validate(dataset_name, task, arize_client)
    if errors:
        print("Validation failed:", errors)
    ```
"""

import logging
from typing import Any, List

import pandas as pd

from arize_experiment.core.arize import ArizeClientProtocol
from arize_experiment.core.errors import ConfigurationError
from arize_experiment.core.schema import ValidationError
from arize_experiment.core.task import Task

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates dataset schemas against task requirements."""

    def validate(  # noqa: C901
        self,
        dataset_name: str,
        task: Task,
        arize_client: ArizeClientProtocol,
    ) -> List[ValidationError]:
        """Validate a dataset against a task's schema requirements.

        Args:
            dataset_name: Name of the dataset to validate
            task: Task instance containing schema requirements
            arize_client: Arize client instance for accessing dataset

        Returns:
            List[ValidationError]: List of validation errors, empty if validation passes

        Raises:
            ConfigurationError: If dataset cannot be accessed or schema validation fails
        """
        try:
            # Get dataset metadata from Arize
            dataset = arize_client.get_dataset(dataset_name)
            if dataset is None:
                raise ConfigurationError(
                    f"Dataset {dataset_name} not found",
                    details={"dataset": dataset_name},
                )

            # Handle the case where dataset is already a DataFrame
            if isinstance(dataset, pd.DataFrame):
                if dataset.empty:
                    raise ConfigurationError(
                        f"Empty DataFrame in dataset {dataset_name}",
                        details={"dataset": dataset_name},
                    )
                validation_data = dataset.iloc[0].to_dict()
            else:
                # Get sample data to validate schema
                sample = dataset.get_sample(limit=1)
                if sample is None or not hasattr(sample, "data") or sample.data is None:
                    raise ConfigurationError(
                        f"Could not get sample data from dataset {dataset_name}",
                        details={"dataset": dataset_name},
                    )

                # Get the data to validate
                sample_data: Any = sample.data[0]
                if isinstance(sample_data, pd.DataFrame):
                    if sample_data.empty:
                        raise ConfigurationError(
                            f"Empty DataFrame in dataset {dataset_name}",
                            details={"dataset": dataset_name},
                        )
                    records = sample_data.to_dict(orient="records")
                    if not records:
                        raise ConfigurationError(
                            f"No records found in dataset {dataset_name}",
                            details={"dataset": dataset_name},
                        )
                    validation_data = records[0]
                else:
                    validation_data = sample_data

            # Validate sample against task schema
            return task.required_schema.validate_data(validation_data)

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise e
            raise ConfigurationError(
                "Failed to validate dataset schema",
                details={"dataset": dataset_name, "task": task.name, "error": str(e)},
            ) from e
