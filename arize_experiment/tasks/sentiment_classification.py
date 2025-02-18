from arize_experiment.core.task import Task
from arize_experiment.core.task import TaskResult
from arize_experiment.core.errors import TaskError
from openai import OpenAI
from openai.types.chat import ChatCompletion
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a sentiment analyzer. Classify the following text as either 'positive','negative',
or 'neutral'. Respond with just one word.
"""


class SentimentClassificationTask(Task):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: str = None,
    ):
        """Initialize the sentiment classification task.

        Args:
            model: Name of the OpenAI model to use
            temperature: Temperature parameter for model inference
            api_key: OpenAI API key (optional if set in environment)
        """
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the task name."""
        return "sentiment_classification"

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        """Execute sentiment classification on input text.

        Args:
            Input: A dictionary containing the input text to classify.

        Returns:
            TaskResult containing:
                output: Classification result ('positive', 'negative', or 'neutral')
                metadata: Processing information including model used
                error: Any error message if task failed
        """
        try:
            logger.info("Executing sentiment classification task")
            logger.info(f"Input: {Input}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    # Get the actual input text from off the input dict.
                    # This is necessary b/c the input param is actually a dict,
                    # and the input text is nested inside that dict.
                    "content": Input["input"],
                },
            ]

            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            logger.info(f"OpenAI chat completion request successful")

            sentiment = self._parse_llm_output(response.choices[0].message.content)

            # Ensure we only get valid labels
            if sentiment not in ["positive", "negative", "neutral"]:
                sentiment = "neutral"

            logger.info(f"Sentiment: {sentiment}")
            logger.info(f"Finishing sentiment classification task")

            return TaskResult(
                input=Input,
                output=sentiment,
                metadata={
                    "model": self._model,
                },
            )
        except Exception as e:
            logger.error(f"Sentiment classification failed: {str(e)}")
            return TaskResult(
                output=None,
                error=f"Sentiment classification failed: {str(e)}",
            )

    def _parse_llm_output(
        self,
        text: str,
    ) -> str:
        """Parse the output of the sentiment classification task.

        Args:
            text: Raw LLM output text

        Returns:
            Classification result
        """
        try:
            return text.strip().lower()
        except Exception as e:
            raise TaskError(f"Failed to parse LLM output: {str(e)}")
