"""Execute an agent by calling a web server."""

import logging
from typing import Any

import requests
from requests.exceptions import RequestException

from arize_experiment.core.task import Task, TaskResult

logger = logging.getLogger(__name__)


class ExecuteAgentTask(Task):
    """Execute an agent by calling a web server.

    This task makes HTTP requests to a specified endpoint to execute an agent
    and process its response. The agent is expected to be running at the
    provided URL endpoint.
    """

    def __init__(self, url: str = "http://localhost:8080"):
        """Initialize the execute agent task.

        Args:
            url: The endpoint URL where the agent is running
        """
        self.url = url

    @property
    def name(self) -> str:
        """Get the task name.

        Returns:
            str: The unique identifier for this task
        """
        return "execute_agent"

    # flake8: noqa: C901
    def execute(self, Input: Any) -> TaskResult:
        """Execute the agent on input text.

        Args:
            Input: Dictionary containing the input data for the task including:
                - agent_id: string identifier for the agent
                - conversation: list of message dictionaries with role and content

        Returns:
            TaskResult containing:
                output: The agent's response from the server
                metadata: Processing information including request details
                error: Any error message if the request or processing failed

        Raises:
            TaskError: If the HTTP request fails or agent processing fails
        """
        try:
            # Validate input format
            if not isinstance(Input, dict):
                return TaskResult(
                    input=Input,
                    output=None,
                    metadata={"url": self.url},
                    error="Input must be a dictionary",
                )

            # Validate required fields
            if "agent_id" not in Input:
                return TaskResult(
                    input=Input,
                    output=None,
                    metadata={"url": self.url},
                    error="Input must contain 'agent_id' field",
                )

            if "conversation" not in Input:
                return TaskResult(
                    input=Input,
                    output=None,
                    metadata={"url": self.url},
                    error="Input must contain 'conversation' field",
                )

            # Validate conversation format
            conversation = Input["conversation"]
            if not isinstance(conversation, list):
                return TaskResult(
                    input=Input,
                    output=None,
                    metadata={"url": self.url},
                    error="Conversation must be a list",
                )

            for message in conversation:
                if not isinstance(message, dict):
                    return TaskResult(
                        input=Input,
                        output=None,
                        metadata={"url": self.url},
                        error="Each conversation message must be a dictionary",
                    )
                if "role" not in message or "content" not in message:
                    return TaskResult(
                        input=Input,
                        output=None,
                        metadata={"url": self.url},
                        error=(
                            "Each conversation message must have "
                            "'role' and 'content' fields"
                        ),
                    )

            # Make the API request
            response = requests.post(self.url, json=Input)
            response.raise_for_status()
            output = response.json()

            # Return successful result
            return TaskResult(
                input=Input,
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
                input=Input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            return TaskResult(
                input=Input,
                output=None,
                metadata={"url": self.url},
                error=error_msg,
            )
