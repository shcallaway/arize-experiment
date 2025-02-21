"""
Example task implementation using the Task base class.
"""

from typing import Any, Dict

from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry


@TaskRegistry.register("echo")
class EchoTask(Task):
    """Simple echo task that returns its input.

    This is a basic example task that demonstrates the Task interface.
    It simply returns whatever input it receives.
    """

    def __init__(self) -> None:
        """Initialize the echo task."""
        self._validated = False

    @property
    def name(self) -> str:
        """Get the task name."""
        return "echo"

    def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
        """Execute the echo task.

        Args:
            dataset_row: Dictionary containing:
                - input: The input data to echo

        Returns:
            TaskResult containing:
            - The input data as output
            - Optional metadata about the input type
            - No error (this task cannot fail)
        """
        return TaskResult(
            dataset_row=dataset_row,
            output=dataset_row["input"],
            metadata={},
        )
