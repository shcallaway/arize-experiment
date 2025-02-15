"""
Evaluator for judging the accuracy of sentiment classifications using OpenAI's API.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion

from arize_experiment.core.evaluator import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


SENTIMENT_ACCURACY_PROMPT = """
You are evaluating whether a sentiment classification is accurate for a given text.

[Input Text]: {input_text}
[Given Classification]: {classification}

Determine if this classification is accurate. Consider:
1. The overall tone and emotion of the text
2. The presence of positive/negative indicators
3. The context and nuance of the message

Respond with either "correct" or "incorrect", followed by a brief explanation.
"""


class SentimentClassificationAccuracyEvaluator(BaseEvaluator):
    """Evaluates the accuracy of sentiment classifications using OpenAI's API.
    
    This evaluator uses GPT-4 to analyze whether a given sentiment classification
    (positive, neutral, negative) is appropriate for a given input text.
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
        return "sentiment_classification_accuracy"

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

    def _parse_output(self, text: str) -> Tuple[bool, str]:
        """Parse the LLM output to extract the decision and explanation.
        
        Args:
            text: Raw LLM output text
        
        Returns:
            Tuple of (is_correct: bool, explanation: str)
        """
        text = text.strip().lower()
        is_correct = text.startswith("correct")
        explanation = text[text.find(" "):].strip()
        return is_correct, explanation

    def evaluate(self, output: Any) -> EvaluationResult:
        """Evaluate whether the sentiment classification is accurate.
        
        Args:
            output: Dict containing:
                - input_text: The original text being classified
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
        
        required_keys = {"input_text", "classification"}
        if not all(key in output for key in required_keys):
            raise ValueError(
                f"Output dict must contain keys {required_keys}, "
                f"got {set(output.keys())}"
            )

        try:
            # Format prompt with input text and classification
            prompt = SENTIMENT_ACCURACY_PROMPT.format(
                input_text=output["input_text"],
                classification=output["classification"]
            )

            # Make API call
            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis accuracy checker. "
                        "Respond with 'correct' or 'incorrect' followed by a brief explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
            )

            # Parse response
            is_correct, explanation = self._parse_output(
                response.choices[0].message.content
            )

            return EvaluationResult(
                score=1.0 if is_correct else 0.0,
                label="correct" if is_correct else "incorrect",
                explanation=explanation
            )

        except Exception as e:
            error_msg = f"Sentiment accuracy evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) 
