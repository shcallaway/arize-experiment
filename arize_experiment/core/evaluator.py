"""
Core evaluator interface and result types for arize-experiment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvaluationResult:
    """Standardized result type for all evaluators."""

    score: float  # Normalized score between 0 and 1
    label: str  # Classification label
    explanation: Optional[str] = None  # Optional explanation of the result

    def __post_init__(self):
        """Validate the evaluation result."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if not self.label:
            raise ValueError("Label cannot be empty")


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
            EvaluationResult containing the normalized score, label, and optional
            explanation.
        
        Raises:
            ValueError: If the output is not of the expected type or format
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the evaluator is properly configured.
        
        This method should check that any required configuration or dependencies
        are available and valid.
        
        Returns:
            bool: True if the evaluator is properly configured
        
        Raises:
            ValueError: If the evaluator is not properly configured
        """
        pass
