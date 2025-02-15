"""
Sentiment analysis evaluator using OpenAI's API.
"""

import logging
from typing import Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from arize_experiment.core.evaluator import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


SENTIMENT_PROMPT = """
You are evaluating whether tone is positive, neutral, or negative

[Message]: {output}

Respond with either "positive", "neutral", or "negative"
"""


class SentimentEvaluator(BaseEvaluator):
    """Evaluates text sentiment using OpenAI's API.
    
    This evaluator uses GPT-4 to analyze the sentiment of text output,
    classifying it as positive, neutral, or negative.
    """

    def __init__(
        self,
        model: str = "gpt-4",
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
        self._validated = False

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "sentiment"

    def validate(self) -> bool:
        """Validate the evaluator configuration.
        
        This checks that:
        1. OpenAI client is properly configured
        2. Model is available
        
        Returns:
            True if validation succeeds
        
        Raises:
            ValueError: If validation fails
        """
        if self._validated:
            return True

        try:
            # Test API access with a minimal request
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": "test"}
                ],
                max_tokens=1
            )
            self._validated = True
            return True

        except Exception as e:
            error_msg = (
                f"Failed to validate OpenAI configuration: {str(e)}\n"
                "Please check your OpenAI API key and model access."
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate the sentiment of the output text.
        
        Args:
            output: The text to evaluate
        
        Returns:
            EvaluationResult with:
            - score: 1.0 for positive, 0.5 for neutral, 0.0 for negative
            - label: "positive", "neutral", or "negative"
            - explanation: Description of the sentiment analysis
        
        Raises:
            ValueError: If output is not a string or API call fails
        """
        if not isinstance(output, str):
            raise ValueError(f"Expected string output, got {type(output)}")

        try:
            # Format prompt with output text
            prompt = SENTIMENT_PROMPT.format(output=output)

            # Make API call
            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis assistant. "
                        "Respond only with 'positive', 'neutral', or 'negative'.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
            )

            # Get sentiment label
            label = response.choices[0].message.content.strip().lower()

            # Convert label to score and create explanation
            score = {
                "positive": 1.0,
                "neutral": 0.5,
                "negative": 0.0
            }.get(label, 0.0)

            explanation = f"Text was evaluated as having {label} sentiment"

            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation
            )

        except Exception as e:
            error_msg = f"Sentiment evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
