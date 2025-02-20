"""Sentiment classification task using OpenAI's API.

This task uses OpenAI's language models to classify text sentiment
as positive, negative, or neutral. The task leverages OpenAI's
advanced language understanding capabilities to provide accurate
sentiment analysis.

Example:
    ```python
    from arize_experiment.core.task_registry import TaskRegistry

    # Create task instance
    task = TaskRegistry.get("sentiment_classification")()

    # Classify sentiment
    result = task.execute({
        "input": "I really enjoyed this product!"
    })
    # result.output would be "positive"
    ```

Note:
    This task requires an OpenAI API key to be set either in the
    environment variables or passed during initialization.
"""

import logging
from typing import Any, Dict, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from arize_experiment.core.errors import TaskError
from arize_experiment.core.schema import ColumnSchema, DatasetSchema, DataType
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a sentiment analyzer. Classify the following text as either 'positive',
'negative', or 'neutral'. Respond with just one word.
"""


@TaskRegistry.register("sentiment_classification")
class SentimentClassificationTask(Task):
    """Task for classifying text sentiment using OpenAI's API.

    This task uses OpenAI's language models to analyze text and classify
    its sentiment as either positive, negative, or neutral. The task uses
    a simple prompt-based approach, instructing the model to respond with
    a single word classification.

    The task is designed to:
    1. Be simple and straightforward to use
    2. Provide consistent results
    3. Handle various text inputs gracefully
    4. Return clear, categorical classifications

    Attributes:
        model (str): The OpenAI model to use for classification
        temperature (float): The sampling temperature for generation
        api_key (Optional[str]): OpenAI API key if not set in environment
        _client (OpenAI): OpenAI client instance

    Note:
        The task uses a zero temperature by default to ensure consistent
        results, but this can be adjusted if more variation is desired.
    """

    def __init__(
        self,
        *args: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sentiment classification task.

        Args:
            model: The OpenAI model to use
            temperature: The sampling temperature for the model
            api_key: OpenAI API key (optional if set in environment)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the task name.

        Returns:
            str: The unique identifier for this task
        """
        return "sentiment_classification"

    @property
    def required_schema(self) -> DatasetSchema:
        """Get the dataset schema required by this task.

        Returns:
            DatasetSchema: The required schema for input data
        """
        return DatasetSchema(
            columns={
                "input": ColumnSchema(
                    name="input",
                    types=[DataType.STRING],
                    required=True,
                    description="The text to classify",
                )
            },
            description="Dataset containing text for sentiment classification",
        )

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        """Execute the sentiment classification task.

        Args:
            Input: A dictionary containing the input text to classify under the
                "input" key.

        Returns:
            TaskResult: The result of the classification task.

        Raises:
            TaskError: If the input is invalid or classification fails
        """
        try:
            # Extract input from Input.
            # This is a little confusing, but it is necessary b/c the dataset example
            # is passed to the execute method as the "Input" param, and this param
            # contains the entire example in dict format. Within the example dict,
            # there is an "input" key which contains the text to classify.
            if "input" not in Input:
                return TaskResult(
                    input=Input,
                    output=None,
                    error="Input must be a dictionary with 'input' key",
                    metadata={
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                )

            input = Input["input"]
            if not isinstance(input, str):
                return TaskResult(
                    input=Input,
                    output=None,
                    error="Input must be a string",
                    metadata={
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                )

            # Call OpenAI API
            response: ChatCompletion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input},
                ],
            )

            # Parse and return result
            content = response.choices[0].message.content
            if content is None:
                raise TaskError("LLM returned empty response")

            result = self._parse_llm_output(content)

            return TaskResult(
                input=Input,
                output=result,
                metadata={
                    "model": self.model,
                    "temperature": self.temperature,
                },
            )

        except Exception as e:
            logger.error(f"Sentiment classification failed: {str(e)}")
            return TaskResult(
                input=Input,
                output=None,
                error=f"Sentiment classification failed: {str(e)}",
                metadata={
                    "model": self.model,
                    "temperature": self.temperature,
                },
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
