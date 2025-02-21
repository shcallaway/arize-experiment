"""Execute an agent by calling a web server."""

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


@TaskRegistry.register("execute_agent")
class ExecuteAgentTask(Task):
    """Execute an agent by calling a web server.

    This task makes HTTP requests to a specified endpoint to execute an agent
    and process its response. The agent is expected to be running at the
    provided URL endpoint.
    """

    def __init__(
        self, *args: Any, base_url: str = "http://localhost:8080", **kwargs: Any
    ) -> None:
        """Initialize the execute agent task.

        Args:
            base_url: The base URL where the agent is running
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
        return "execute_agent"

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
                output: The agent's response from the server
                metadata: Processing information including request details
                error: Any error message if the request or processing failed

        Raises:
            TaskError: If the HTTP request fails or agent processing fails
        """
        # Create input from dataset row.
        # If there is no input inside the dataset row, default to row itself.
        input = {"input": dataset_row.get("input", dataset_row)}

        # Parse the input JSON string
        conversation = None
        try:
            conversation = json.loads(input["input"])
        except Exception as e:
            error_msg = f"Failed to parse input: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                input=input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        # Validate that the conversation data is a list
        if not isinstance(conversation, list):
            error_msg = "Conversation is not a list"
            logger.error(error_msg)
            return TaskResult(
                input=input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        # Make the API request
        response = None
        try:
            response = requests.post(
                self.url, json={"agent_id": "test", "conversation": conversation}
            )
            response.raise_for_status()
        except Exception as e:
            error_msg = f"Failed to execute agent request: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                input=input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        # Parse the response body as JSON
        output = None
        try:
            output = response.json()
        except Exception as e:
            error_msg = f"Failed to parse response: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                input=input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )

        # Return successful result
        return TaskResult(
            input=input,
            output=output,
            metadata={
                "url": self.url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            },
        )
