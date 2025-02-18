"""
Evaluator for judging the accuracy of sentiment classifications using OpenAI's API.
"""

import logging
from typing import Any, Optional, Tuple

from arize_experiment.core.task import TaskResult
from openai import OpenAI
from openai.types.chat import ChatCompletion

from arize_experiment.core.evaluator import BaseEvaluator
from arize.experimental.datasets.experiments.types import EvaluationResult

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


class SentimentClassificationAccuracyEvaluator(BaseEvaluator):
    """Evaluates the accuracy of sentiment classifications using OpenAI's API.

    This evaluator uses GPT-4o-mini to analyze whether a given sentiment
    classification (positive, neutral, negative) is appropriate for input text.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (0-1)
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "sentiment_classification_accuracy"

    def _parse_output(self, text: str) -> Tuple[bool, str]:
        """Parse the LLM output to extract the decision and explanation.

        Args:
            text: Raw LLM output text

        Returns:
            Tuple of (correct: bool, explanation: str)
        """
        try:
            text = text.strip().lower()
            correct = text.startswith("correct")
            explanation = text[text.find(" ")].strip()
            return correct, explanation
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {str(e)}")

    def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate whether the sentiment classification is accurate.

        Args:
            output: The sentiment classification (positive/neutral/negative)

        Returns:
            EvaluationResult containing:
            - score: 1.0 if correct, 0.0 if incorrect
            - label: "correct" or "incorrect"
            - explanation: Detailed explanation of the accuracy assessment

        Raises:
            ValueError: If output format is invalid or API call fails
        """
        logger.info("Evaluating sentiment classification accuracy")
        logger.debug(f"Task result: {task_result}")

        # Get the original input text and sentiment classification from the task result
        input = task_result.input["input"]
        sentiment = task_result.output

        logger.info(f"Input: {input}")
        logger.info(f"Sentiment: {sentiment}")

        try:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (f"Input: {input}\n" f"Classification: {sentiment}"),
                },
            ]

            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            correct, explanation = self._parse_output(
                response.choices[0].message.content
            )

            label = "correct" if correct else "incorrect"
            score = 1.0 if correct else 0.0

            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            raise ValueError(f"Sentiment accuracy evaluation failed: {str(e)}")

    def __call__(self, task_result: Any) -> Any:
        """Make the evaluator callable by delegating to evaluate.

        This allows evaluators to be used directly as functions.

        Args:
            task_result: The task result to evaluate

        Returns:
            The evaluation result
        """
        # If task_result is a dictionary, convert it to a TaskResult
        if isinstance(task_result, dict):
            task_result = TaskResult(
                input=task_result["input"],
                output=task_result["output"],
                metadata=task_result.get("metadata", {}),
                error=task_result.get("error", None),
            )

        return self.evaluate(task_result)
