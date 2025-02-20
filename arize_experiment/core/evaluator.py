"""
Core evaluator interface and result types for arize-experiment.

This module defines the base evaluator interface that all evaluators must
implement. Evaluators are responsible for assessing the quality or performance
of task outputs against some criteria.

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

        def evaluate(self, result: Any) -> EvaluationResult:
            accuracy = self._calculate_accuracy(result)
            return EvaluationResult(
                score=accuracy,
                passed=accuracy >= self.threshold,
                metadata={"threshold": self.threshold}
            )
    ```
"""

from abc import ABC, abstractmethod
from typing import Any

from arize.experimental.datasets.experiments.types import EvaluationResult


class BaseEvaluator(ABC):
    """Base class for all evaluators.

    All evaluators in the framework must inherit from this class and implement
    its abstract methods. This ensures consistent behavior and return types
    across all evaluators.

    An evaluator is responsible for:
    1. Taking a task output
    2. Assessing its quality or performance
    3. Returning a standardized evaluation result

    Implementation Guidelines:
        1. Evaluators should be stateless where possible
        2. Configuration should be done in __init__
        3. The evaluate method should handle all error cases
        4. Results should be returned in an EvaluationResult object
        5. Metadata should include any relevant configuration

    Attributes:
        name (str): The unique identifier for this evaluator
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the evaluator.

        Subclasses can override this with their own initialization parameters.
        """
        super().__init__()

    # @abstractmethod
    # def __str__(self) -> str:
    #     """Return the simple name of the evaluator."""
    #     pass

    # @abstractmethod
    # def __repr__(self) -> str:
    #     """Return a string representation of the evaluator."""
    #     pass

    def __str__(self) -> str:
        """Return the simple name of the evaluator."""
        return self.name

    def __repr__(self) -> str:
        """Return a string representation of the evaluator."""
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
    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate the given output and return a standardized result.

        Args:
            output: The output to evaluate. Evaluators should document
                   their expected input types and formats.

        Returns:
            EvaluationResult containing:
                - score: Normalized score between 0 and 1
                - label: Classification label or category
                - explanation: Optional explanation of the evaluation

        Raises:
            EvaluatorError: If evaluation fails or output is invalid
            ValueError: If the output format is incorrect
        """
        pass

    @abstractmethod
    def __call__(self, output: Any) -> EvaluationResult:
        """Make the evaluator callable by delegating to evaluate.

        This allows evaluators to be used directly as functions.

        Args:
            output: The output to evaluate

        Returns:
            The evaluation result

        Raises:
            EvaluatorError: If evaluation fails
        """
        pass
