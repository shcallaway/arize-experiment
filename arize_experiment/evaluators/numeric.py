"""
Numeric value evaluator.
"""

import logging
from typing import Any, Union

import pandas as pd

from arize_experiment.core.evaluator import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class NumericEvaluator(BaseEvaluator):
    """Evaluates whether values are numeric.
    
    This evaluator checks if values can be converted to numbers and
    determines their specific numeric type (integer or float).
    """

    def __init__(self):
        """Initialize the evaluator."""
        self._validated = False

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "numeric"

    def validate(self) -> bool:
        """Validate the evaluator configuration.
        
        This evaluator doesn't require external dependencies or
        configuration, so validation always succeeds.
        
        Returns:
            True
        """
        self._validated = True
        return True

    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate whether the output is numeric.
        
        Args:
            output: The value to evaluate
        
        Returns:
            EvaluationResult with:
            - score: 1.0 for numeric values, 0.0 for non-numeric
            - label: "numeric" or "non-numeric"
            - explanation: Description of the numeric analysis
        """
        # Convert input to string for consistent handling
        str_output = str(output).strip()

        # Handle empty string case
        if not str_output:
            return EvaluationResult(
                score=0.0,
                label="non-numeric",
                explanation="Input is empty"
            )

        try:
            # Attempt to convert to numeric using pandas
            numeric_value = pd.to_numeric(str_output)

            # Determine if it's an integer
            is_integer = float(numeric_value).is_integer()
            value_type = "integer" if is_integer else "float"

            return EvaluationResult(
                score=1.0,
                label="numeric",
                explanation=f"Input '{str_output}' is a valid {value_type}"
            )

        except (ValueError, TypeError):
            return EvaluationResult(
                score=0.0,
                label="non-numeric",
                explanation=f"Input '{str_output}' cannot be converted to a number"
            )

    def _is_integer(self, value: Union[int, float]) -> bool:
        """Check if a numeric value is effectively an integer.
        
        Args:
            value: The numeric value to check
        
        Returns:
            True if the value is effectively an integer
        """
        if isinstance(value, int):
            return True
        return float(value).is_integer()
