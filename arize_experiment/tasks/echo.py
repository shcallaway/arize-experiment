"""
Example task implementation using the Task base class.
"""

import logging
from typing import Any

from arize_experiment.core.task import Task, TaskResult

logger = logging.getLogger(__name__)


class EchoTask(Task):
    """Simple echo task that returns its input.

    This is a basic example task that demonstrates the Task interface.
    It simply returns whatever input it receives.
    """

    def __init__(self):
        """Initialize the echo task."""
        self._validated = False

    @property
    def name(self) -> str:
        """Get the task name."""
        return "echo"

    def execute(self, Input: Any) -> TaskResult:
        """Execute the echo task.

        Args:
            Input: Any input data

        Returns:
            TaskResult containing:
            - The input data as output
            - Optional metadata about the input type
            - No error (this task cannot fail)
        """
        return TaskResult(output=Input, metadata={})
