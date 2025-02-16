"""
Evaluator for judging the accuracy of sentiment classifications using OpenAI's API.
"""

import logging
from typing import Any, Optional, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion

from arize_experiment.core.evaluator import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a sentiment analysis accuracy checker.

You will be given an input text and a classification. You must decide whether the classification is accurate.

Consider:
1. The overall tone and emotion of the text
2. The presence of positive/negative indicators
3. The context and nuance of the message

Respond with either "correct" or "incorrect", followed by a brief explanation.
"""


class SentimentClassificationAccuracyEvaluator(BaseEvaluator):
    """Evaluates the accuracy of sentiment classifications using OpenAI's API.
    
    This evaluator uses GPT-4o-mini to analyze whether a given sentiment classification
    (positive, neutral, negative) is appropriate for a given input text.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None
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
            explanation = text[text.find(" "):].strip()
            return correct, explanation
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {str(e)}")

    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate whether the sentiment classification is accurate.
        
        Args:
            output: Dict containing:
                - input: The original text being classified
                - classification: The sentiment classification (positive/neutral/negative)
        
        Returns:
            EvaluationResult with:
            - score: 1.0 if correct, 0.0 if incorrect
            - label: "correct" or "incorrect"
            - explanation: Detailed explanation of the accuracy assessment
        
        Raises:
            ValueError: If output format is invalid or API call fails
        """
        if not isinstance(output, dict):
            raise ValueError(f"Expected dict output, got {type(output)}")
        
        required_keys = {"input", "classification"}
        if not all(key in output for key in required_keys):
            raise ValueError(
                f"Output dict must contain keys {required_keys}, "
                f"got {set(output.keys())}"
            )

        try:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Input: {output['input']}\nClassification: {output['classification']}",
                },
            ]

            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            # Parse response
            correct, explanation = self._parse_output(
                response.choices[0].message.content
            )

            return EvaluationResult(
                score=1.0 if correct else 0.0,
                label="correct" if correct else "incorrect",
                explanation=explanation
            )
        except Exception as e:
            error_msg = f"Sentiment accuracy evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) 
