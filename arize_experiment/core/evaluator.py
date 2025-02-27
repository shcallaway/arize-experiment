"""
Core evaluator interface and result types for arize-experiment.

This module defines the base evaluator interface that all evaluators must
implement. Evaluators are responsible for assessing the quality or performance
of task outputs against specific criteria.

The evaluator framework is designed to be:
1. Extensible - New evaluators can be easily added
2. Composable - Multiple evaluators can be used together
3. Consistent - All evaluators follow the same interface
4. Reusable - Evaluators can be used across different tasks

Example:
    ```python
    from arize_experiment.core.evaluator import BaseEvaluator
    from arize.experimental.datasets.experiments.types import EvaluationResult

    class AccuracyEvaluator(BaseEvaluator):
        def __init__(self, threshold: float = 0.8):
            self.threshold = threshold

        @property
        def name(self) -> str:
            return "accuracy_evaluator"

        def evaluate(self, output: TaskResult) -> EvaluationResult:
            accuracy = self._calculate_accuracy(output)
            return EvaluationResult(
                score=accuracy,
                passed=accuracy >= self.threshold,
                metadata={"threshold": self.threshold}
            )

        def __call__(self, output: TaskResult) -> EvaluationResult:
            return self.evaluate(output)
    ```
"""

from abc import ABC, abstractmethod
from typing import Any, final
import logging

from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.task import TaskResult


class BaseEvaluator(ABC):
    """Base class for all evaluators.

    All evaluators in the framework must inherit from this class and implement
    its abstract methods. This ensures consistent behavior and return types
    across all evaluators.

    An evaluator is responsible for:
    1. Taking a task output
    2. Assessing its quality or performance against defined criteria
    3. Returning a standardized evaluation result

    Implementation Guidelines:
        1. Evaluators should be stateless where possible
        2. Configuration should be done in __init__
        3. The evaluate method should handle all error cases gracefully
        4. Results should be returned in an EvaluationResult object
        5. Metadata should include any relevant configuration or context
        6. Names should be lowercase with underscores
        7. Error handling should be comprehensive

    Attributes:
        name (str): The unique identifier for this evaluator
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the evaluator.

        Subclasses should override this with their own initialization parameters
        and document them appropriately.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()

    @final
    def __str__(self) -> str:
        """Return a string representation of the evaluator.

        Returns:
            str: The evaluator's name
        """
        return self.name

    @final
    def __repr__(self) -> str:
        """Return a detailed string representation of the evaluator.

        Returns:
            str: The evaluator's name
        """
        return self.name

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the evaluator.

        Returns:
            str: The unique identifier for this evaluator. This should be a
                lowercase string with underscores, e.g. 'accuracy_evaluator'
        """
        pass

    @abstractmethod
    def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate the given output and return a standardized result.

        Args:
            task_result: TaskResult containing the task execution result with input,
                   output, metadata and any error information. Evaluators should
                   document their expected TaskResult structure in their implementation.

        Returns:
            EvaluationResult containing:
                - score: Normalized score between 0 and 1
                - passed: Boolean indicating if the evaluation passed
                - metadata: Optional dictionary with additional information
                - explanation: Optional explanation of the evaluation result

        Raises:
            EvaluatorError: If evaluation fails or output is invalid
            ValueError: If the output format is incorrect
        """
        pass

    @final
    def __call__(self, task_result: Any) -> EvaluationResult:
        """Make the evaluator callable by delegating to evaluate.
        
        Args:
            task_result: The task result to evaluate. This can be:
                - A TaskResult object
                - A dictionary representation of a TaskResult
                - The output of a task (in which case we need to wrap it)
                
        Returns:
            EvaluationResult: The evaluation result
            
        Raises:
            EvaluatorError: If evaluation fails
            ValueError: If input format is invalid
        """
        if isinstance(task_result, TaskResult):
            return self.evaluate(task_result)
        elif isinstance(task_result, dict) and all(k in task_result for k in ['dataset_row', 'output']):
            return self.evaluate(TaskResult(**task_result))
        else:
            # If we just got the output (like a string), we need to create a TaskResult
            # This is a fallback and should generate a warning
            logging.warning(
                "Evaluator received task output instead of TaskResult. "
                "This may cause issues if the evaluator needs access to metadata or input. "
                "Consider using task(dataset_row, return_full_result=True) instead."
            )
            # We can't properly reconstruct the original dataset_row
            dummy_task_result = TaskResult(
                dataset_row={"input": task_result if isinstance(task_result, str) else str(task_result)},
                output=task_result,
                metadata={}
            )
            return self.evaluate(dummy_task_result)