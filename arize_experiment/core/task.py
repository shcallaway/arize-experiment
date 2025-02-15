"""
Core task interface and base implementation for arize-experiment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TaskResult:
    """Standardized result type for all tasks."""

    output: Any  # The task's output
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution
    error: Optional[str] = None  # Error message if task failed

    @property
    def success(self) -> bool:
        """Check if the task executed successfully."""
        return self.error is None


class Task(ABC):
    """Base class for all tasks.
    
    All tasks must inherit from this class and implement its abstract methods.
    This ensures consistent behavior and return types across all tasks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the task.
        
        Returns:
            str: The unique name of this task
        """
        pass

    @abstractmethod
    def execute(self, input_data: Any) -> TaskResult:
        """Execute the task with the given input.
        
        Args:
            input_data: The input data for the task. Tasks should document
                       their expected input types.
        
        Returns:
            TaskResult containing the output and optional metadata or error.
        
        Raises:
            ValueError: If the input is not of the expected type or format
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the task is properly configured.
        
        This method should check that any required configuration or dependencies
        are available and valid.
        
        Returns:
            bool: True if the task is properly configured
        
        Raises:
            ValueError: If the task is not properly configured
        """
        pass

    def __str__(self) -> str:
        """Get a string representation of the task."""
        return f"{self.__class__.__name__}(name={self.name})"
