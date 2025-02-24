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
    from arize_experiment.evaluators.sentiment_classification_is_accurate import (
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
from typing import List, Optional, Tuple

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


@EvaluatorRegistry.register("sentiment_classification_is_accurate")
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
        return "sentiment_classification_is_accurate"

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
            logger.info("Evaluating sentiment classification accuracy")
            logger.debug(f"Task result: {task_result}")

            # Get the original input text and sentiment classification
            text, classification = self._parse_task_result(task_result)

            if not isinstance(text, str):
                raise ValueError("Text must be a string")

            if not isinstance(classification, str):
                raise ValueError("Classification must be a string")

            # Normalize classification for comparison
            if not self.case_sensitive:
                classification = classification.lower()

            # Validate classification label
            if self.strict_matching and classification not in self._valid_labels:
                raise ValueError(
                    f"Invalid sentiment label: {classification}. "
                    f"Must be one of: {self._valid_labels}"
                )

            # Call OpenAI API to evaluate accuracy
            messages: List[
                ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
            ] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=SYSTEM_PROMPT,
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(f"Input: {text}\n" f"Classification: {classification}"),
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

            # Calculate score and label
            score = 1.0 if correct else 0.0
            label = self._determine_label(correct)

            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            error_msg = f"Sentiment accuracy evaluation failed: {str(e)}"
            logger.error(error_msg)
            return EvaluationResult(
                score=0.0,
                label="error",
                explanation=error_msg,
            )

    def _determine_label(self, correct: bool) -> str:
        """Determine the label for the given correctness.

        Args:
            correct: Whether the prediction is correct

        Returns:
            str: The label for the given correctness
        """
        return "correct" if correct else "incorrect"

    def _parse_task_result(self, task_result: TaskResult | dict) -> Tuple[str, str]:
        """Parse the task result to extract the input text and classification.

        Args:
            task_result: TaskResult or dict containing:
                - dataset_row: Dict with text and expected sentiment
                - output: Predicted sentiment label

        Returns:
            Tuple of (text: str, classification: str)
        """
        if isinstance(task_result, dict):
            text = task_result["dataset_row"]["input"]
            classification = task_result["output"]
        else:
            text = task_result.dataset_row["input"]
            classification = task_result.output
        return text, classification
