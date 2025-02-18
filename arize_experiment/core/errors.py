"""
Core error types and error handling utilities for the Arize experiment framework.
"""

from typing import Optional, Dict, Any
import logging

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

    def format_message(self) -> str:
        """Format the error message with troubleshooting tips based on error type."""
        base_msg = super().format_message()
        error_msg = str(self.details.get("error", "")).lower()

        # Add specific troubleshooting tips based on error message
        if "unauthorized" in error_msg or "invalid key" in error_msg:
            return (
                f"{base_msg}\nTip: Check your API key and credentials in the .env file"
            )
        if "connection" in error_msg or "timeout" in error_msg:
            return f"{base_msg}\nTip: Check your internet connection and try again"
        if "rate limit" in error_msg:
            return f"{base_msg}\nTip: Please wait a moment and try again"
        if "already exists" in error_msg:
            return (
                f"{base_msg}\n"
                "Tip: Use a different experiment name or delete the existing experiment"
            )
        if "not found" in error_msg or "does not exist" in error_msg:
            return (
                f"{base_msg}\n"
                "Tip: Verify that the resource exists and you have access to it"
            )

        return base_msg


class ConfigurationError(ArizeExperimentError):
    """Error related to configuration issues."""

    def format_message(self) -> str:
        """Format the error message with troubleshooting tips based on error type."""
        base_msg = super().format_message()
        return (
            f"{base_msg}\n"
            "Tip: Check your configuration file and ensure all required values are present"
        )


class HandlerError(ArizeExperimentError):
    """Error related to command handling issues."""

    def format_message(self) -> str:
        """Format the error message with troubleshooting tips based on error type."""
        base_msg = super().format_message()
        return f"{base_msg}\nTip: Check your command and ensure all required values are present"


class EvaluatorError(ArizeExperimentError):
    """Error related to evaluator issues."""

    def format_message(self) -> str:
        """Format the error message with troubleshooting tips based on error type."""
        base_msg = super().format_message()
        return f"{base_msg}\nTip: Check your evaluator and ensure all required values are present"


class TaskError(ArizeExperimentError):
    """Error related to task issues."""

    def format_message(self) -> str:
        """Format the error message with troubleshooting tips based on error type."""
        base_msg = super().format_message()
        return f"{base_msg}\nTip: Check your task and ensure all required values are present"


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
