"""
Core task interface and base implementation for arize-experiment.

This module defines the base Task interface and TaskResult data structure
that all tasks in the framework must implement. It provides a standardized
way to execute tasks and handle their results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from arize_experiment.core.errors import TaskError
from arize_experiment.core.schema import DatasetSchema


@dataclass
class TaskResult:
    """Standardized result type for all tasks.

    This dataclass encapsulates the input, output, and metadata for a task execution.
    It also tracks any errors that occurred during execution.

    Attributes:
        input: The original input data provided to the task
        output: The task's output data
        metadata: Optional metadata about the execution (e.g., timing, model info)
        error: Optional error message if the task failed
    """

    input: Dict[str, Any]  # The input data for the task
    output: Any  # The task's output
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution
    error: Optional[str] = None  # Error message if task failed

    @property
    def success(self) -> bool:
        """Check if the task executed successfully.

        Returns:
            bool: True if no error occurred, False otherwise
        """
        return self.error is None


class Task(ABC):
    """Base class for all tasks.

    All tasks in the framework must inherit from this class and implement
    its abstract methods. This ensures consistent behavior and return types
    across all tasks.

    A task represents a discrete unit of work that can be executed with
    some input data and produces a result. Tasks should be stateless and
    thread-safe where possible.
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the task with any required parameters."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the task's name.

        Returns:
            str: The task's name
        """
        pass

    @property
    @abstractmethod
    def required_schema(self) -> DatasetSchema:
        """Get the dataset schema required by this task.

        Returns:
            DatasetSchema: The required schema for input data
        """
        pass

    @abstractmethod
    def execute(
        self,
        Input: Dict[str, Any],
    ) -> TaskResult:
        """Execute the task with the given input.

        Args:
            Input: Dictionary containing the task's input data

        Returns:
            TaskResult: The result of the task execution

        Raises:
            TaskError: If the task fails to execute
        """
        pass

    def __str__(self) -> str:
        """Get a string representation of the task.

        Returns:
            str: A human-readable string describing this task
        """
        return f"{self.__class__.__name__}(name={self.name})"

    def __call__(self, Input: Dict[str, Any]) -> Any:
        """Make the task callable by delegating to execute.

        This allows tasks to be used directly as functions. It unwraps
        the TaskResult and returns just the output, or raises an error
        if execution failed.

        Args:
            Input: The input data for the task

        Returns:
            The TaskResult object returned by the task's execute method

        Raises:
            TaskError: If the task fails to execute
        """
        result = self.execute(Input)
        if result.error:
            raise TaskError(result.error)
        return result
