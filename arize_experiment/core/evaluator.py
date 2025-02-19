"""
Core evaluator interface and result types for arize-experiment.

This module defines the base evaluator interface that all evaluators must
implement. Evaluators are responsible for assessing the quality or performance
of task outputs against some criteria.
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
    """

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
