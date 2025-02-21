"""
Sentiment classification task implementation using OpenAI's API.

This module provides a task implementation for classifying text sentiment
using OpenAI's language models. It demonstrates:
1. Task implementation best practices
2. OpenAI API integration
3. Error handling
4. Input validation
5. Result standardization

The task is designed to be:
1. Simple to use
2. Reliable
3. Configurable
4. Well-documented
5. Error-resistant

Example:
    ```python
    from arize_experiment.tasks.sentiment_classification import (
        SentimentClassificationTask
    )

    task = SentimentClassificationTask(
        model="gpt-4",
        temperature=0
    )

    result = task.execute({
        "input": "I really enjoyed this product!"
    })

    print(result.output)  # "positive"
    ```
"""

import logging
import os
from typing import Any, Dict

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


@TaskRegistry.register("classify_sentiment")
class ClassifySentimentTask(Task):
    """Classify sentiment in text using OpenAI models.

    This task uses OpenAI's language models to analyze the sentiment of text,
    classifying it as positive, negative, or neutral.
    """

    def __init__(
        self,
        *args: Any,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sentiment classification task.

        Args:
            model: The OpenAI model to use for classification
            temperature: The sampling temperature (default: 0)
            api_key: Optional OpenAI API key (defaults to env var)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @property
    def name(self) -> str:
        """Get the task name.

        Returns:
            str: The unique identifier for this task
        """
        return "classify_sentiment"

    @property
    def required_schema(self) -> DatasetSchema:
        """Get the dataset schema required by this task.

        The schema requires a single text input column that will be
        analyzed for sentiment.

        Returns:
            DatasetSchema: Schema requiring a text input column
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

    def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
        """Execute the sentiment classification task.

        This method:
        1. Validates the input text
        2. Prepares the classification prompt
        3. Makes the API request
        4. Processes the response
        5. Returns a standardized result

        Args:
            dataset_row: Dictionary containing:
                - input: The text to classify

        Returns:
            TaskResult containing:
                - dataset_row: The original dataset row
                - output: The sentiment classification (positive/negative/neutral)
                - metadata: Additional information including confidence scores
                - error: Error message if classification failed

        Raises:
            TaskError: Various reasons
        """
        try:
            # Validate dataset row format
            if "input" not in dataset_row:
                return TaskResult(
                    dataset_row=dataset_row,
                    output=None,
                    metadata={
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                    error="dataset_row must be a dictionary with 'input' key",
                )

            text = dataset_row["input"]

            if not isinstance(text, str):
                return TaskResult(
                    dataset_row=dataset_row,
                    output=None,
                    metadata={
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                    error="dataset_row['input'] must be a string",
                )

            # Call OpenAI API
            response: ChatCompletion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
            )

            # Parse and return result
            content = response.choices[0].message.content

            if content is None:
                raise TaskError("LLM returned empty response")

            result = self._parse_llm_output(content)

            return TaskResult(
                dataset_row=dataset_row,
                output=result,
                metadata={
                    "model": self.model,
                    "temperature": self.temperature,
                },
            )

        except Exception as e:
            error_msg = f"Sentiment classification failed: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                dataset_row=dataset_row,
                output=None,
                metadata={
                    "model": self.model,
                    "temperature": self.temperature,
                },
                error=error_msg,
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
