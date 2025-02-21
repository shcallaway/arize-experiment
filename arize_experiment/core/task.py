"""
Core task interface and base implementation for arize-experiment.

This module defines the base Task interface and TaskResult data structure
that all tasks in the framework must implement. It provides a standardized
way to execute tasks and handle their results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, final

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
    @final
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

    Implementation Guidelines:
        1. Tasks should be stateless to ensure thread safety
        2. All configuration should be done in __init__
        3. The execute method should handle all error cases gracefully
        4. Input validation should be done using the required_schema
        5. Results should be returned in a TaskResult object

    Example:
        ```python
        from arize_experiment.core.task import Task
        from arize_experiment.core.schema import DatasetSchema, ColumnSchema, DataType

        class MyTask(Task):
            def __init__(self, param1: str = "default") -> None:
                self.param1 = param1

            @property
            def name(self) -> str:
                return "my_task"

            @property
            def required_schema(self) -> DatasetSchema:
                return DatasetSchema(
                    columns={
                        "input": ColumnSchema(
                            name="input",
                            types=[DataType.STRING],
                            required=True
                        )
                    }
                )

            def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
                try:
                    result = self._process_input(dataset_row["input"])
                    return TaskResult(
                        input=dataset_row,
                        output=result,
                        metadata={"param1": self.param1}
                    )
                except Exception as e:
                    return TaskResult(
                        input=dataset_row,
                        output=None,
                        error=str(e)
                    )
        ```
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
        dataset_row: Dict[str, Any],
    ) -> TaskResult:
        """Execute the task with the given input.

        Args:
            dataset_row: Dictionary containing the task's input data

        Returns:
            TaskResult: The result of the task execution

        Raises:
            TaskError: If the task fails to execute
        """
        pass

    @final
    def __str__(self) -> str:
        """Get a string representation of the task.

        Returns:
            str: A human-readable string describing this task
        """
        return f"{self.__class__.__name__}(name={self.name})"

    @final
    def __call__(
        self, dataset_row: Dict[str, Any], _return_task_result: bool = False
    ) -> Any:
        """Make the task callable by delegating to execute.

        This allows tasks to be used directly as functions. It unwraps
        the TaskResult and returns just the output, or raises an error
        if execution failed.

        Args:
            dataset_row: The input data for the task
            _return_task_result: If True, returns the entire TaskResult object

        Returns:
            The TaskResult object returned by the task's execute method

        Raises:
            TaskError: If the task fails to execute
        """
        result = self.execute(dataset_row)

        if result.error:
            if _return_task_result:
                return result
            raise TaskError(result.error)

        return result if _return_task_result else result.output
