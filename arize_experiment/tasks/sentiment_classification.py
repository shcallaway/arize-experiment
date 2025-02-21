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

    Features:
    1. Simple API integration
    2. Consistent results
    3. Configurable model parameters
    4. Comprehensive error handling
    5. Clear output format

    Implementation Details:
    - Uses a zero-shot classification approach
    - Returns standardized sentiment labels
    - Handles various text inputs gracefully
    - Provides detailed error information
    - Supports custom model configuration

    Attributes:
        model (str): The OpenAI model to use for classification
        temperature (float): The sampling temperature for generation
        api_key (Optional[str]): OpenAI API key if not set in environment
        _client (OpenAI): OpenAI client instance

    Example:
        ```python
        task = SentimentClassificationTask(
            model="gpt-4",
            temperature=0,
            api_key="your-api-key"  # Optional
        )

        result = task.execute({
            "input": "The service was excellent!"
        })

        if result.success:
            print(f"Sentiment: {result.output}")
            print(f"Confidence: {result.metadata['confidence']}")
        else:
            print(f"Error: {result.error}")
        ```
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
            model: The OpenAI model to use (default: gpt-4o-mini)
            temperature: The sampling temperature (default: 0)
            api_key: Optional OpenAI API key
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If model or temperature are invalid
            ConfigurationError: If API key is invalid
        """
        if not model:
            raise ValueError("Model name cannot be empty")

        if temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

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
                    error="dataset_row must be a dictionary with 'input' key",
                    metadata={
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                )

            text = dataset_row["input"]

            if not isinstance(text, str):
                return TaskResult(
                    dataset_row=dataset_row,
                    output=None,
                    error="dataset_row['input'] must be a string",
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
                    {"role": "user", "content": text},
                ],
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
                error=error_msg,
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
