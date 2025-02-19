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

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the task.

        Returns:
            str: The unique identifier for this task. This should be a
                 lowercase string with underscores, e.g. 'sentiment_analysis'
        """
        pass

    @abstractmethod
    def execute(
        self,
        Input: Dict[str, Any],
    ) -> TaskResult:
        """Execute the task with the given input.

        Args:
            Input: Dictionary containing the input data for the task.
                       Tasks should document their expected input format.

        Returns:
            TaskResult containing:
                - input: The original input data
                - output: The task's output data
                - metadata: Optional execution metadata
                - error: Any error message if task failed

        Raises:
            TaskError: If the input is invalid or task execution fails
            ValueError: If the input format is incorrect
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
