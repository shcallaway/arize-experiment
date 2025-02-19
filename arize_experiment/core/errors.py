"""
Core error types and error handling utilities for the Arize experiment framework.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ArizeExperimentError(Exception):
    """Base exception for all Arize experiment errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with details if available."""
        msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg = f"{msg} ({details_str})"
        return msg


class ArizeClientError(ArizeExperimentError):
    """Base class for client-related errors.

    This includes:
    - Authentication issues
    - Network connectivity problems
    - Rate limiting
    - API errors
    - Dataset access issues
    - Experiment creation/retrieval issues
    """


class ConfigurationError(ArizeExperimentError):
    """Error related to configuration issues."""


class HandlerError(ArizeExperimentError):
    """Error related to command handling issues."""


class EvaluatorError(ArizeExperimentError):
    """Error related to evaluator issues."""


class TaskError(ArizeExperimentError):
    """Error related to task issues."""


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
