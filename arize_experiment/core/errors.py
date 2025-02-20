"""
Core error types and error handling utilities for the Arize experiment framework.

This module defines the base error types and utilities used throughout the
framework. It provides a consistent approach to error handling with:
1. Detailed error messages
2. Structured error details
3. Hierarchical error types
4. Logging integration

Example:
    ```python
    from arize_experiment.core.errors import TaskError

    try:
        result = process_data()
    except Exception as e:
        raise TaskError(
            "Failed to process data",
            details={"error": str(e)}
        )
    ```
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ArizeExperimentError(Exception):
    """Base exception for all Arize experiment errors.

    This class provides a consistent base for all framework errors with support
    for detailed error messages and structured error details.

    Attributes:
        message (str): The main error message
        details (Dict[str, Any]): Additional error context and details

    Example:
        ```python
        raise ArizeExperimentError(
            "Operation failed",
            details={
                "operation": "data_processing",
                "error": "Invalid format"
            }
        )
        ```
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the error.

        Args:
            message: The main error message
            details: Optional dictionary of additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with details if available.

        Returns:
            str: The formatted error message including any details
        """
        msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg = f"{msg} ({details_str})"
        return msg


class ArizeClientError(ArizeExperimentError):
    """Error related to Arize API client operations.

    This error type is raised for issues related to:
    - Authentication failures
    - Network connectivity problems
    - API rate limiting
    - Invalid API responses
    - Dataset access issues
    - Experiment creation/retrieval failures

    Example:
        ```python
        try:
            client.create_dataset(data)
        except Exception as e:
            raise ArizeClientError(
                "Failed to create dataset",
                details={"error": str(e)}
            )
        ```
    """


class ConfigurationError(ArizeExperimentError):
    """Error related to configuration issues.

    This error is raised when there are problems with the configuration
    of tasks, evaluators, or the framework itself. Common cases include:
    - Missing required environment variables
    - Invalid configuration file format
    - Invalid parameter values
    - Missing required configuration options

    Example:
        ```python
        if not config.get("api_key"):
            raise ConfigurationError("API key is required in configuration")
        ```
    """


class HandlerError(ArizeExperimentError):
    """Error related to command handling issues.

    This error is raised when there are problems executing CLI commands
    or handling user input. Common cases include:
    - Invalid command syntax
    - Missing required arguments
    - Failed task or evaluator creation
    - Invalid file paths or formats

    Example:
        ```python
        if not os.path.exists(input_path):
            raise HandlerError(f"Input file not found: {input_path}")
        ```
    """


class EvaluatorError(ArizeExperimentError):
    """Error related to evaluator issues.

    This error is raised when there are problems during evaluation
    execution. Common cases include:
    - Invalid evaluation input format
    - Failed metric calculation
    - Missing ground truth data
    - Evaluation timeout or resource limits

    Example:
        ```python
        if not isinstance(predictions, list):
            raise EvaluatorError("Predictions must be a list")
        ```
    """


class TaskError(ArizeExperimentError):
    """Error related to task execution issues.

    This error is raised when there are problems during task
    execution, including:
    - Invalid input data format
    - Failed API calls or external service errors
    - Resource constraints or timeouts
    - Internal processing errors
    - Task configuration issues

    Example:
        ```python
        try:
            response = api.process(input_data)
        except ApiError as e:
            raise TaskError(
                "API call failed",
                details={
                    "api": "openai",
                    "error": str(e)
                }
            )
        ```
    """


def pretty_print_error(error: Exception) -> str:
    """Format an error into a user-friendly message.

    Args:
        error: The exception to format

    Returns:
        A user-friendly error message
    """
    if isinstance(error, ArizeExperimentError):
        return str(error)

    # Map common error types to user-friendly messages
    if isinstance(error, ValueError):
        return f"Invalid input: {str(error)}"
    if isinstance(error, KeyError):
        return f"Missing required value: {str(error)}"
    if isinstance(error, FileNotFoundError):
        return f"File not found: {str(error)}"

    # Generic error message for unknown error types
    return f"An unexpected error occurred: {str(error)}"
