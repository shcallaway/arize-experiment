"""Execute an agent by calling a web server."""

from typing import Any

from arize_experiment.core.task import Task, TaskResult


class ExecuteAgentTask(Task):
    """Execute an agent by calling a web server.

    This task makes HTTP requests to a specified endpoint to execute an agent
    and process its response. The agent is expected to be running at the
    provided URL endpoint.
    """

    def __init__(self, url: str):
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

    def execute(self, Input: Any) -> TaskResult:
        """Execute the agent on input text.

        Args:
            Input: Dictionary containing the input data for the task

        Returns:
            TaskResult containing:
                output: The agent's response (currently unimplemented)
                metadata: Processing information including request details
                error: Any error message if the request or processing failed

        Raises:
            TaskError: If the HTTP request fails or agent processing fails
        """
        if not isinstance(Input, dict):
            return TaskResult(
                input=Input,
                output=None,
                metadata={"url": self.url},
                error="Input must be a dictionary",
            )

        return TaskResult(input=Input, output=None, metadata={"url": self.url})
