"""
Example task implementation using the Task base class.
"""

import logging
from typing import Any, Dict, Optional

from arize_experiment.core.task import Task, TaskResult

logger = logging.getLogger(__name__)


class EchoTask(Task):
    """Simple echo task that returns its input.
    
    This is a basic example task that demonstrates the Task interface.
    It simply returns whatever input it receives, optionally with some
    metadata about the input type.
    """

    def __init__(self, add_type_info: bool = False):
        """Initialize the echo task.
        
        Args:
            add_type_info: Whether to include input type information in metadata
        """
        self._add_type_info = add_type_info
        self._validated = False

    @property
    def name(self) -> str:
        """Get the task name."""
        return "echo"

    def validate(self) -> bool:
        """Validate the task configuration.
        
        This task doesn't require any special configuration,
        so validation always succeeds.
        
        Returns:
            True
        """
        self._validated = True
        return True

    def execute(self, input_data: Any) -> TaskResult:
        """Execute the echo task.
        
        Args:
            input_data: Any input data
        
        Returns:
            TaskResult containing:
            - The input data as output
            - Optional metadata about the input type
            - No error (this task cannot fail)
        """
        metadata: Optional[Dict[str, Any]] = None
        
        if self._add_type_info:
            metadata = {
                "input_type": type(input_data).__name__,
                "input_length": len(str(input_data))
            }

        return TaskResult(
            output=input_data,
            metadata=metadata
        )

