"""Call a chatbot server to process conversations."""

import json
import logging
import os
from typing import Any, Dict

import requests
from requests.exceptions import RequestException

from arize_experiment.core.schema import ColumnSchema, DatasetSchema, DataType
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("call_chatbot_server")
class CallChatbotServerTask(Task):
    """Call a chatbot server to process conversations.

    This task makes HTTP requests to a specified endpoint to process conversations
    through a chatbot server. The server is expected to be running at the
    provided URL endpoint.
    """

    def __init__(
        self, *args: Any, base_url: str = "http://localhost:8080", **kwargs: Any
    ) -> None:
        """Initialize the chatbot server task.

        Args:
            base_url: The base URL where the chatbot server is running
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        base_url = os.getenv("AGENT_EXECUTION_URL", base_url)
        self.url = f"{base_url}/execute"

    @property
    def name(self) -> str:
        """Get the task name.

        Returns:
            str: The unique identifier for this task
        """
        return "call_chatbot_server"

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
                    description="A JSON string containing the conversation history",
                )
            },
            description="Dataset containing text inputs for agent execution",
        )

    def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
        """Execute the agent on input text.

        Args:
            dataset_row: Dictionary containing:
                - input: String containing the input text for the agent

        Returns:
            TaskResult containing:
                - dataset_row: The original dataset row
                - output: The agent's response from the server
                - metadata: Processing information including request details
                - error: Any error message if the request or processing failed

        Raises:
            TaskError: Various reasons
        """
        try:
            # Validate dataset row format
            if "input" not in dataset_row:
                return TaskResult(
                    dataset_row=dataset_row,
                    output=None,
                    metadata={"url": self.url},
                    error="dataset_row must be a dictionary with 'input' key",
                )

            conversation: Any = json.loads(dataset_row["input"])

            if not isinstance(conversation, list):
                return TaskResult(
                    dataset_row=dataset_row,
                    output=None,
                    metadata={"url": self.url},
                    error="dataset_row['input'] must be a list",
                )

            # Make the API request
            response = requests.post(
                self.url, json={"agent_id": "test", "conversation": conversation}
            )

            response.raise_for_status()

            output = response.json()

            # Return successful result
            return TaskResult(
                dataset_row=dataset_row,
                output=output,
                metadata={
                    "url": self.url,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                },
            )

        except RequestException as e:
            error_msg = f"Failed to execute agent request: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                dataset_row=dataset_row,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        except ValueError as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                dataset_row=dataset_row,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                dataset_row=dataset_row,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )
