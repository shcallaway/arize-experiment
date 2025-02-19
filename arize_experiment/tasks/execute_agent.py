"""Execute an agent by calling a web server."""

import logging
from typing import Any, Dict

import requests
from requests.exceptions import RequestException

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
        self, *args: Any, url: str = "http://localhost:8080", **kwargs: Any
    ) -> None:
        """Initialize the execute agent task.

        Args:
            url: The endpoint URL where the agent is running
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.url = url

    @property
    def name(self) -> str:
        """Get the task name.

        Returns:
            str: The unique identifier for this task
        """
        return "execute_agent"

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        """Execute the agent on input text.

        Args:
            Input: Dictionary containing:
                - messages: List of message dictionaries, where each message has:
                    - role: string indicating the speaker (e.g. "user" or "assistant")
                    - content: string containing the message content

        Returns:
            TaskResult containing:
                output: The agent's response from the server
                metadata: Processing information including request details
                error: Any error message if the request or processing failed

        Raises:
            TaskError: If the HTTP request fails or agent processing fails
        """
        try:
            messages = Input.get("messages", [])

            # Validate input format
            if not isinstance(messages, list):
                return TaskResult(
                    input=Input,
                    output=None,
                    metadata={"url": self.url},
                    error="messages must be a list",
                )

            # Validate message format
            for message in messages:
                if not isinstance(message, dict):
                    return TaskResult(
                        input=Input,
                        output=None,
                        metadata={"url": self.url},
                        error="Each message must be a dictionary",
                    )
                if "role" not in message or "content" not in message:
                    return TaskResult(
                        input=Input,
                        output=None,
                        metadata={"url": self.url},
                        error="Each message must have 'role' and 'content' fields",
                    )

            # Prepare request payload
            payload = {"agent_id": "test", "conversation": messages}

            # Make the API request
            response = requests.post(self.url, json=payload)
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
