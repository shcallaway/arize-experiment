"""
Core evaluator interface and result types for arize-experiment.
"""

from abc import ABC, abstractmethod
from typing import Any

from arize.experimental.datasets.experiments.types import EvaluationResult

class BaseEvaluator(ABC):
    """Base class for all evaluators.

    All evaluators must inherit from this class and implement its abstract methods.
    This ensures consistent behavior and return types across all evaluators.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the evaluator.

        Returns:
            str: The unique name of this evaluator
        """
        pass

    @abstractmethod
    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate the given output and return a standardized result.

        Args:
            output: The output to evaluate. Can be any type, but evaluators
                   should document their expected input types.

        Returns:
            EvaluationResult containing:
            - score: Normalized score between 0 and 1
            - label: Classification label
            - explanation: Optional explanation of the result

        Raises:
            EvaluatorError: If the evaluator fails to evaluate the output
        """
        pass

    @abstractmethod
    def __call__(self, output: Any) -> Any:
        """Make the evaluator callable by delegating to evaluate.

        This allows evaluators to be used directly as functions.
        """
        pass
