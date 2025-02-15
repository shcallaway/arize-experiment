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


class TextProcessingTask(Task):
    """Task that performs basic text processing.
    
    This example task demonstrates more complex processing by:
    1. Validating input
    2. Performing text transformations
    3. Handling errors
    4. Providing detailed metadata
    """

    def __init__(
        self,
        strip: bool = True,
        lower: bool = True,
        max_length: Optional[int] = None
    ):
        """Initialize the text processing task.
        
        Args:
            strip: Whether to strip whitespace
            lower: Whether to convert to lowercase
            max_length: Optional maximum length limit
        """
        self._strip = strip
        self._lower = lower
        self._max_length = max_length
        self._validated = False

    @property
    def name(self) -> str:
        """Get the task name."""
        return "text_processor"

    def validate(self) -> bool:
        """Validate the task configuration.
        
        Checks that:
        1. max_length is positive if set
        
        Returns:
            True if validation succeeds
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self._max_length is not None and self._max_length <= 0:
            raise ValueError("max_length must be positive")
        
        self._validated = True
        return True

    def execute(self, input_data: Any) -> TaskResult:
        """Execute the text processing task.
        
        Args:
            input_data: The text to process
        
        Returns:
            TaskResult containing:
            - Processed text as output
            - Metadata about the processing
            - Error message if processing failed
        """
        try:
            # Validate input
            if not isinstance(input_data, str):
                raise ValueError(f"Expected string input, got {type(input_data)}")

            # Process text
            text = input_data
            
            if self._strip:
                text = text.strip()
            
            if self._lower:
                text = text.lower()
            
            # Check length
            if (
                self._max_length is not None and
                len(text) > self._max_length
            ):
                text = text[:self._max_length]

            # Create metadata
            metadata = {
                "original_length": len(input_data),
                "processed_length": len(text),
                "was_stripped": self._strip,
                "was_lowered": self._lower,
                "was_truncated": len(text) < len(input_data)
            }

            return TaskResult(
                output=text,
                metadata=metadata
            )

        except Exception as e:
            error_msg = f"Text processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return TaskResult(
                output=None,
                error=error_msg
            )
