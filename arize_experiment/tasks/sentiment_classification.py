"""Sentiment classification task using OpenAI's API.

This task uses OpenAI's language models to classify text sentiment
as positive, negative, or neutral.
"""

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
    """Task for classifying text sentiment using OpenAI's API.
    
    This task uses OpenAI's language models to analyze text and classify
    its sentiment as either positive, negative, or neutral.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: str = None,
    ):
        """Initialize the sentiment classification task.

        Args:
            model: Name of the OpenAI model to use
            temperature: Temperature parameter for model inference (0-1)
            api_key: OpenAI API key (optional if set in environment)
        """
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the task name.
        
        Returns:
            str: The unique identifier for this task
        """
        return "sentiment_classification"

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        """Execute sentiment classification on input text.

        Args:
            Input: A dictionary containing the input text to classify.

        Returns:
            TaskResult containing:
                output: The sentiment classification result (positive/negative/neutral)
                metadata: Processing information including model used
                error: Any error message if classification failed

        Raises:
            TaskError: If the input is invalid or classification fails
        """
        try:
            # Extract input from Input.
            # This is a little confusing, but it is necessary b/c the dataset example
            # is passed to the execute method as the "Input" param, and this param
            # contains the entire example in dict format. Within the example dict,
            # there is an "input" key which contains the text to classify.
            if not isinstance(Input, dict) or "input" not in Input:
                raise ValueError("Input must be a dictionary with 'input' key")

            input = Input["input"]
            if not isinstance(input, str):
                raise ValueError("Input must be a string")

            # Call OpenAI API
            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input},
                ],
            )

            # Parse and return result
            result = self._parse_llm_output(response.choices[0].message.content)

            return TaskResult(
                input=Input,
                output=result,
                metadata={
                    "model": self._model,
                    "temperature": self._temperature,
                },
            )

        except Exception as e:
            logger.error(f"Sentiment classification failed: {str(e)}")
            return TaskResult(
                input=Input,
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
            str: Classification result (positive/negative/neutral)

        Raises:
            TaskError: If the output cannot be parsed
        """
        try:
            return text.strip().lower()
        except Exception as e:
            raise TaskError(f"Failed to parse LLM output: {str(e)}")
