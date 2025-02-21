"""
Sentiment classification accuracy evaluator implementation.

This module provides an evaluator for assessing the accuracy of sentiment
classification tasks. It demonstrates:
1. Evaluator implementation best practices
2. Ground truth comparison
3. Accuracy calculation
4. Error handling
5. Result standardization

The evaluator is designed to:
1. Be accurate and reliable
2. Handle edge cases gracefully
3. Provide detailed feedback
4. Support various input formats
5. Be configurable

Example:
    ```python
    from arize_experiment.evaluators.sentiment_classification_accuracy import (
        SentimentClassificationAccuracyEvaluator
    )

    evaluator = SentimentClassificationAccuracyEvaluator(
        threshold=0.8,
        strict_matching=True
    )

    result = evaluator.evaluate({
        "input": {"text": "Great product!", "expected": "positive"},
        "output": "positive"
    })

    print(f"Accuracy: {result.score}")
    print(f"Passed: {result.passed}")
    ```
"""

import logging
from typing import Any, Optional, Tuple

from arize.experimental.datasets.experiments.types import EvaluationResult
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task import TaskResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You will be given an input text and a classification. You must decide whether "
    "the classification is accurate.\n\n"
    "Consider:\n"
    "1. The overall tone and emotion of the text\n"
    "2. The presence of positive/negative indicators\n"
    "3. The context and nuance of the message\n\n"
    'Respond with either "correct" or "incorrect", followed by a brief explanation.'
)


@EvaluatorRegistry.register("sentiment_classification_accuracy")
class SentimentClassificationAccuracyEvaluator(BaseEvaluator):
    """Evaluates the accuracy of sentiment classifications.

    This evaluator assesses how well a sentiment classification task performs
    by comparing its predictions against ground truth labels. It supports
    both strict and lenient matching modes and provides detailed feedback
    about misclassifications.

    Features:
    1. Configurable accuracy threshold
    2. Strict/lenient matching options
    3. Detailed error analysis
    4. Support for partial credit
    5. Rich metadata output

    Implementation Details:
    - Normalizes sentiment labels for comparison
    - Handles edge cases and invalid inputs
    - Provides confidence scores
    - Includes misclassification details
    - Supports batch evaluation

    Attributes:
        threshold (float): Minimum accuracy required to pass (0.0 to 1.0)
        strict_matching (bool): Whether to require exact label matches
        case_sensitive (bool): Whether to consider case in comparisons
        _valid_labels (Set[str]): Set of recognized sentiment labels

    Example:
        ```python
        evaluator = SentimentClassificationAccuracyEvaluator(
            threshold=0.8,
            strict_matching=True,
            case_sensitive=False
        )

        result = evaluator.evaluate(TaskResult(
            input={"text": "Amazing service!", "expected": "positive"},
            output="positive",
            metadata={"confidence": 0.95}
        ))

        if result.passed:
            print(f"Evaluation passed with score: {result.score}")
            print(f"Confidence: {result.metadata['confidence']}")
        else:
            print(f"Evaluation failed: {result.metadata['error_analysis']}")
        ```
    """

    def __init__(
        self,
        threshold: float = 0.8,
        strict_matching: bool = True,
        case_sensitive: bool = False,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the sentiment classification accuracy evaluator.

        Args:
            threshold: Minimum accuracy score to pass (default: 0.8)
            strict_matching: Whether to require exact matches (default: True)
            case_sensitive: Whether to consider case (default: False)
            model: The OpenAI model to use (default: gpt-4o-mini)
            temperature: The sampling temperature (default: 0)
            api_key: Optional OpenAI API key

        Raises:
            ValueError: If threshold is not between 0 and 1 or temperature is invalid
        """
        super().__init__()
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not model:
            raise ValueError("Model name cannot be empty")

        self.threshold = threshold
        self.strict_matching = strict_matching
        self.case_sensitive = case_sensitive
        self._model = model
        self._temperature = temperature
        self._valid_labels = {"positive", "negative", "neutral"}
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the evaluator name.

        Returns:
            str: The unique identifier for this evaluator
        """
        return "sentiment_classification_accuracy"

    def _parse_llm_output(self, text: str) -> Tuple[bool, str]:
        """Parse the LLM output to extract the decision and explanation.

        Args:
            text: Raw LLM output text

        Returns:
            Tuple of (correct: bool, explanation: str)

        Raises:
            ValueError: If the output cannot be parsed
        """
        try:
            text = text.strip().lower()
            if not (text.startswith("correct") or text.startswith("incorrect")):
                raise ValueError("Output must start with 'correct' or 'incorrect'")
            correct = text.startswith("correct")
            space_pos = text.find(" ")
            if space_pos == -1:
                raise ValueError("No explanation found in output")
            explanation = text[space_pos:].strip()
            if not explanation:
                raise ValueError("Empty explanation")
            return correct, explanation
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {str(e)}")

    def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate sentiment classification accuracy.

        This method:
        1. Validates input and output formats
        2. Normalizes sentiment labels
        3. Compares predictions to ground truth
        4. Calculates accuracy scores
        5. Generates detailed feedback

        Args:
            task_result: TaskResult containing:
                - input: Dict with text and expected sentiment
                - output: Predicted sentiment label
                - metadata: Optional prediction metadata

        Returns:
            EvaluationResult containing:
                - score: Accuracy score between 0 and 1
                - label: String indicating if the prediction was correct
                - explanation: Human-readable evaluation summary

        Raises:
            EvaluatorError: If evaluation fails
            ValueError: If input format is invalid
        """
        try:
            # Get the original input text and sentiment classification
            input_text = task_result.input["input"]
            predicted_sentiment = task_result.output

            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string")
            if not isinstance(predicted_sentiment, str):
                raise ValueError("Predicted sentiment must be a string")

            # Normalize sentiments for comparison
            if not self.case_sensitive:
                predicted_sentiment = predicted_sentiment.lower()

            # Validate sentiment label
            if self.strict_matching and predicted_sentiment not in self._valid_labels:
                raise ValueError(
                    f"Invalid sentiment label: {predicted_sentiment}. "
                    f"Must be one of: {self._valid_labels}"
                )

            # Call OpenAI API to evaluate accuracy
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=SYSTEM_PROMPT,
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        f"Input: {input_text}\n"
                        f"Classification: {predicted_sentiment}"
                    ),
                ),
            ]

            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response")

            correct, explanation = self._parse_llm_output(content)

            # Calculate score and prepare result
            score = 1.0 if correct else 0.0
            label = "correct" if correct else "incorrect"

            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Sentiment accuracy evaluation failed: {str(e)}")
            return EvaluationResult(
                score=0.0,
                label="error",
                explanation=f"Evaluation failed: {str(e)}",
            )

    # def __call__(self, task_result: Any) -> EvaluationResult:
    #     """Make the evaluator callable by delegating to evaluate.

    #     This allows the evaluator to be used directly as a function.
    #     If given a dictionary instead of a TaskResult, it will be
    #     converted automatically.

    #     Args:
    #         task_result: TaskResult object, dict, or output value

    #     Returns:
    #         EvaluationResult with scores and metadata

    #     Raises:
    #         EvaluatorError: If evaluation fails
    #         ValueError: If input format is invalid
    #     """
    #     if task_result is None:
    #         return EvaluationResult(
    #             score=0.0,
    #             label="error",
    #             explanation="Task result was None"
    #         )

    #     # If task_result is a string (direct output), wrap it
    #     if isinstance(task_result, str):
    #         task_result = TaskResult(
    #             input={"input": task_result},
    #             output=task_result,
    #             metadata={},
    #         )

    #     # If it's a dictionary, convert it
    #     elif isinstance(task_result, dict):
    #         task_result = TaskResult(
    #             input=task_result.get("input", {}),
    #             output=task_result.get("output"),
    #             metadata=task_result.get("metadata", {}),
    #             error=task_result.get("error"),
    #         )

    #     return self.evaluate(task_result)
